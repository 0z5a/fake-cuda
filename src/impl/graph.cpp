#include "impl/graph.h"
#include "impl/capture.h"
#include "impl/core.h"
#include "impl/device.h"
#include "impl/scheduler.h"
#include "impl/virtual_work.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace fake_cuda::detail {
thread_local CUstreamCaptureMode thread_capture_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
Graph::Graph(CUcontext context, std::vector<GraphNode> nodes, size_t lanes) noexcept
    : context_(context), nodes_(std::move(nodes)), lanes_(lanes) {}

CUresult Graph::supports(unsigned long long flags) const noexcept {
    // Default-priority nodes need no priority arbitration. Nonzero captured
    // kernel priorities require a scheduling policy this model does not provide.
    if ((flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY) &&
        std::any_of(nodes_.begin(), nodes_.end(),
                    [](const GraphNode &entry) {
                        return entry.node.kind == Kind::kernel && entry.node.priority != 0;
                    }))
        return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_SUCCESS;
}

CUresult Graph::launch(Scheduler &s, Key key, OpPtr &completion) const {
    const CUcontext ctx = key.context;
    // Preflight all handles and bounds before committing any work. Temporal
    // validity is checked at each node's dependency-adjusted start.
    for (const GraphNode &entry : nodes_) {
        const Node &node = entry.node;
        if (node.event && (s.events.find(node.event) == s.events.end() ||
                           s.events.at(node.event).context != ctx))
            return CUDA_ERROR_INVALID_HANDLE;
        if ((node.first && s.memory.check_bounds(ctx, node.first, node.bytes) != CUDA_SUCCESS) ||
            (node.second && s.memory.check_bounds(ctx, node.second, node.bytes) != CUDA_SUCCESS))
            return CUDA_ERROR_INVALID_VALUE;
    }
    auto *device = fake_cuda::virtual_core_device(s.queue.device_for(ctx));
    std::vector<KernelQuery> queries;
    std::vector<size_t> prediction_index(nodes_.size(), no_node);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto &node = nodes_[i].node;
        if (node.kind != Kind::kernel) continue;
        prediction_index[i] = queries.size();
        queries.push_back({ctx, device->ordinal(), &device->profile(), node.launch.get(), LaunchMode::graph_replay});
    }
    std::vector<PredictorResult> predictions;
    CUresult status = s.predict(queries, predictions);
    if (status != CUDA_SUCCESS) return status;
    auto prediction_for = [&](size_t i) -> const PredictorResult * {
        return prediction_index[i] == no_node ? nullptr : &predictions[prediction_index[i]];
    };
    std::vector<Key> lanes(lanes_);
    for (size_t i = 0; i < lanes_; ++i)
        lanes[i] = {ctx, reinterpret_cast<CUstream>(std::numeric_limits<std::uintptr_t>::max() - s.new_handle()), 0, false};

    // Use a private copy of the resource timelines for preflight so a failed
    // replay does not submit any work. Event waits can delay async allocation use.
    Time launch_time = VirtualClock::now();
    std::vector<Time> predicted(nodes_.size());
    std::vector<Time> lane_end(lanes_, s.queue.earliest(key, Kind::marker, completion, launch_time));
    Time h2d_available = device->h2d_queue.available_at();
    Time d2h_available = device->d2h_queue.available_at();
    Time compute_available = device->compute_queue.available_at();
    auto resource_time = [&](Kind kind) -> Time * {
        switch (kind) {
        case Kind::h2d: return &h2d_available;
        case Kind::d2h: return &d2h_available;
        case Kind::compute:
        case Kind::kernel: return &compute_available;
        default: return nullptr;
        }
    };
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const GraphNode &entry = nodes_[i];
        const Node &node = entry.node;
        Time at = std::max(launch_time, lane_end[entry.lane]);
        Time *available = resource_time(node.kind);
        if (available) at = std::max(at, *available);
        if (entry.event_dependency != no_node) at = std::max(at, predicted[entry.event_dependency]);
        else if (node.kind == Kind::wait) {
            auto record = s.events.at(node.event).record;
            if (record) at = std::max(at, record->end);
        }
        if ((node.first && s.memory.check_pointer(ctx, node.first, node.bytes, at) != CUDA_SUCCESS) ||
            (node.second && s.memory.check_pointer(ctx, node.second, node.bytes, at) != CUDA_SUCCESS))
            return CUDA_ERROR_INVALID_VALUE;
        const auto *prediction = prediction_for(i);
        predicted[i] = at + (prediction ? prediction->service_time : QueueScheduler::duration(node.kind, node.bytes));
        lane_end[entry.lane] = predicted[i];
        if (available) *available = predicted[i];
    }
    std::vector<OpPtr> replay;
    replay.reserve(nodes_.size());
    const OpPtr entry = s.queue.schedule(key, Kind::marker, 0, completion, launch_time);
    for (Key lane : lanes) s.queue.schedule(lane, Kind::marker, 0, entry, launch_time);
    std::vector<OpPtr> exits(lanes_, entry);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const GraphNode &entry = nodes_[i];
        const Node &node = entry.node;
        OpPtr dependency;
        if (entry.event_dependency != no_node) dependency = replay[entry.event_dependency];
        else if (node.kind == Kind::wait) dependency = s.events.at(node.event).record;
        Key lane = lanes[entry.lane];
        OpPtr op = s.queue.schedule(lane, node.kind, node.bytes, dependency, launch_time, prediction_for(i));
        op->launch = node.launch;
        if (node.kind == Kind::record) s.events.at(node.event).record = op;
        exits[entry.lane] = op;
        replay.push_back(std::move(op));
    }
    // Join all nodes without importing legacy/default-stream barriers into the DAG.
    for (const OpPtr &exit : exits)
        completion = s.queue.schedule(key, Kind::marker, 0, exit, launch_time);
    for (Key lane : lanes) s.queue.retire_stream(lane);
    s.commit_predictions(predictions);
    return CUDA_SUCCESS;
}

GraphManager::GraphManager() = default;
GraphManager::~GraphManager() = default;

bool GraphManager::capturing(Key key) const {
    auto it = captures_.find(key.context);
    return it != captures_.end() && it->second->contains(key);
}
CUresult GraphManager::append(Key key, Node node, CUcontext event_context,
                              CUgraphNode handle, bool &captured) {
    captured = false;
    if (auto capture = captures_.find(key.context); capture != captures_.end()) {
        if (capture->second->contains(key) && capture->second->invalidated())
            return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
        if (node.kind == Kind::wait && event_context != key.context && capture->second->contains(key))
            return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        captured = capture->second->append(key, node, handle);
    }
    return CUDA_SUCCESS;
}
CUresult GraphManager::begin_capture(Scheduler &s, Key key, CUstreamCaptureMode mode) {
    if (captures_.contains(key.context))
        return capturing(key) ? CUDA_ERROR_STREAM_CAPTURE_UNMATCHED : CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    captures_.emplace(key.context, std::make_unique<Capture>(key, s.opaque<CUgraph>(), mode));
    return CUDA_SUCCESS;
}
CUresult GraphManager::invalidate(Key key) {
    auto it = captures_.find(key.context);
    if (it == captures_.end() || !it->second->contains(key)) return CUDA_SUCCESS;
    it->second->invalidate();
    return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
}
CUresult GraphManager::invalidate_context(CUcontext ctx) {
    auto it = captures_.find(ctx);
    if (it == captures_.end()) return CUDA_SUCCESS;
    it->second->invalidate();
    return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
}
CUresult GraphManager::check_unsafe_call() {
    if (thread_capture_mode == CU_STREAM_CAPTURE_MODE_RELAXED) return CUDA_SUCCESS;
    bool prohibited = false;
    for (auto &[context, capture] : captures_) {
        if (capture->mode() == CU_STREAM_CAPTURE_MODE_RELAXED) continue;
        if (capture->owned_by_thread() ||
            (thread_capture_mode == CU_STREAM_CAPTURE_MODE_GLOBAL &&
             capture->mode() == CU_STREAM_CAPTURE_MODE_GLOBAL)) {
            capture->invalidate();
            prohibited = true;
        }
    }
    return prohibited ? CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED : CUDA_SUCCESS;
}
CUresult GraphManager::end_capture(Key key, CUgraph *graph) {
    auto it = captures_.find(key.context);
    if (it == captures_.end() || !it->second->contains(key)) return CUDA_ERROR_ILLEGAL_STATE;
    if (!it->second->matches_origin(key)) return CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
    Capture &session = *it->second;
    if (session.mode() != CU_STREAM_CAPTURE_MODE_RELAXED && !session.owned_by_thread()) {
        captures_.erase(it);
        *graph = nullptr;
        return CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
    }
    if (session.invalidated()) {
        captures_.erase(it);
        *graph = nullptr;
        return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    }
    if (!session.joined()) {
        captures_.erase(it);
        *graph = nullptr;
        return CUDA_ERROR_STREAM_CAPTURE_UNJOINED;
    }
    CUgraph handle = session.graph();
    graphs_.emplace(handle, std::make_shared<Graph>(session.finish(key.context)));
    captures_.erase(it);
    *graph = handle;
    return CUDA_SUCCESS;
}
void GraphManager::capture_info(Key key, CUstreamCaptureStatus *status, cuuint64_t *id,
                                CUgraph *graph, const CUgraphNode **dependencies,
                                const CUgraphEdgeData **edges, size_t *count) const {
    auto it = captures_.find(key.context);
    const Capture *session = it != captures_.end() && it->second->contains(key) ? it->second.get() : nullptr;
    *status = !session ? CU_STREAM_CAPTURE_STATUS_NONE : session->invalidated() ?
        CU_STREAM_CAPTURE_STATUS_INVALIDATED : CU_STREAM_CAPTURE_STATUS_ACTIVE;
    if (id) *id = session ? reinterpret_cast<std::uintptr_t>(session->graph()) : 0;
    if (graph) *graph = session ? session->graph() : nullptr;
    const CUgraphNode *dependency = session ? session->dependency(key) : nullptr;
    if (dependencies) *dependencies = dependency;
    if (edges) *edges = dependency ? session->dependency_edge(key) : nullptr;
    if (count) *count = dependency ? 1 : 0;
}
CUresult GraphManager::get_nodes(CUcontext ctx, CUgraph graph, CUgraphNode *nodes, size_t *count) const {
    const std::vector<GraphNode> *entries = nullptr;
    if (auto it = graphs_.find(graph); it != graphs_.end() && it->second->context() == ctx)
        entries = &it->second->nodes();
    else if (auto active = captures_.find(ctx);
             active != captures_.end() && active->second->graph() == graph)
        entries = &active->second->graph_nodes();
    if (!entries) return CUDA_ERROR_INVALID_HANDLE;
    if (!nodes) { *count = entries->size(); return CUDA_SUCCESS; }
    const size_t requested = *count;
    const size_t copied = std::min(requested, entries->size());
    for (size_t i = 0; i < copied; ++i) nodes[i] = (*entries)[i].handle;
    for (size_t i = copied; i < requested; ++i) nodes[i] = nullptr;
    *count = copied;
    return CUDA_SUCCESS;
}
CUresult GraphManager::instantiate(Scheduler &s, CUcontext ctx, CUgraph graph,
                                   unsigned long long flags, CUgraphExec *exec) {
    auto it = graphs_.find(graph);
    if (it == graphs_.end() || it->second->context() != ctx) return CUDA_ERROR_INVALID_HANDLE;
    if (CUresult result = it->second->supports(flags); result != CUDA_SUCCESS) return result;
    CUgraphExec handle = s.opaque<CUgraphExec>();
    executables_.emplace(handle, Executable{it->second, {}});
    *exec = handle;
    return CUDA_SUCCESS;
}
CUresult GraphManager::launch(Scheduler &s, CUgraphExec exec, Key key) {
    auto it = executables_.find(exec);
    if (it == executables_.end() || it->second.graph->context() != key.context) return CUDA_ERROR_INVALID_HANDLE;
    if (capturing(key)) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    return it->second.graph->launch(s, key, it->second.completion);
}
CUresult GraphManager::destroy(CUcontext ctx, CUgraph graph) {
    auto it = graphs_.find(graph);
    if (it == graphs_.end() || it->second->context() != ctx) return CUDA_ERROR_INVALID_HANDLE;
    graphs_.erase(it);
    return CUDA_SUCCESS;
}
CUresult GraphManager::destroy_exec(CUcontext ctx, CUgraphExec exec) {
    auto it = executables_.find(exec);
    if (it == executables_.end() || it->second.graph->context() != ctx) return CUDA_ERROR_INVALID_HANDLE;
    executables_.erase(it);
    return CUDA_SUCCESS;
}
void GraphManager::retire_stream(CUcontext ctx, Key key) {
    if (auto it = captures_.find(ctx); it != captures_.end() && it->second->contains(key))
        captures_.erase(it);
}
void GraphManager::retire_context(CUcontext ctx) {
    captures_.erase(ctx);
    for (auto it = graphs_.begin(); it != graphs_.end();)
        if (it->second->context() == ctx) it = graphs_.erase(it); else ++it;
    for (auto it = executables_.begin(); it != executables_.end();)
        if (it->second.graph->context() == ctx) it = executables_.erase(it); else ++it;
}
} // namespace fake_cuda::detail

using fake_cuda::detail::Scheduler;
using fake_cuda::detail::scheduler;

extern "C" {
CUresult virtual_thread_exchange_capture_mode(CUstreamCaptureMode *mode) {
    if (!mode || (*mode != CU_STREAM_CAPTURE_MODE_GLOBAL &&
                  *mode != CU_STREAM_CAPTURE_MODE_THREAD_LOCAL &&
                  *mode != CU_STREAM_CAPTURE_MODE_RELAXED)) return CUDA_ERROR_INVALID_VALUE;
    int count = 0;
    CUresult result = core_count(&count);
    if (result != CUDA_SUCCESS) return result;
    std::swap(*mode, fake_cuda::detail::thread_capture_mode);
    return CUDA_SUCCESS;
}
CUresult virtual_stream_begin_capture(CUstream stream, CUstreamCaptureMode mode) {
    if (mode != CU_STREAM_CAPTURE_MODE_GLOBAL && mode != CU_STREAM_CAPTURE_MODE_THREAD_LOCAL &&
        mode != CU_STREAM_CAPTURE_MODE_RELAXED) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext, fake_cuda::detail::Key key) {
        return s.graph.begin_capture(s, key, mode);
    });
}
CUresult virtual_stream_end_capture(CUstream stream, CUgraph *graph) {
    if (!graph) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext, fake_cuda::detail::Key key) {
        return s.graph.end_capture(key, graph);
    });
}
CUresult virtual_stream_is_capturing(CUstream stream, CUstreamCaptureStatus *status) {
    if (!status) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext, fake_cuda::detail::Key key) {
        s.graph.capture_info(key, status, nullptr, nullptr, nullptr, nullptr, nullptr);
        return CUDA_SUCCESS;
    });
}
CUresult virtual_stream_get_capture_info(CUstream stream, CUstreamCaptureStatus *status,
                                         cuuint64_t *id, CUgraph *graph,
                                         const CUgraphNode **dependencies,
                                         const CUgraphEdgeData **edges, size_t *count) {
    if (!status) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext, fake_cuda::detail::Key key) {
        s.graph.capture_info(key, status, id, graph, dependencies, edges, count);
        return CUDA_SUCCESS;
    });
}
CUresult virtual_graph_get_nodes(CUgraph graph, CUgraphNode *nodes, size_t *count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        return s.graph.get_nodes(ctx, graph, nodes, count);
    });
}
CUresult virtual_graph_instantiate_flags(CUgraphExec *exec, CUgraph graph, unsigned long long flags) {
    if (!exec) return CUDA_ERROR_INVALID_VALUE;
    constexpr unsigned long long supported = CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH |
                                             CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
    if (flags & ~supported) return CUDA_ERROR_NOT_SUPPORTED;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        return s.graph.instantiate(s, ctx, graph, flags, exec);
    });
}
CUresult virtual_graph_instantiate(CUgraphExec *exec, CUgraph graph) {
    return virtual_graph_instantiate_flags(exec, graph, 0);
}
CUresult virtual_graph_launch(CUgraphExec exec, CUstream stream) {
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext, fake_cuda::detail::Key key) {
        return s.graph.launch(s, exec, key);
    });
}
CUresult virtual_graph_destroy(CUgraph graph) {
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        return s.graph.destroy(ctx, graph);
    });
}
CUresult virtual_graph_exec_destroy(CUgraphExec exec) {
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        return s.graph.destroy_exec(ctx, exec);
    });
}
} // extern "C"
