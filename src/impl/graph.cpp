#include "impl/graph.h"
#include "impl/capture.h"
#include "impl/device.h"
#include "impl/scheduler.h"
#include "impl/virtual_work.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace fake_cuda::detail {
Graph::Graph(CUcontext context, std::vector<GraphNode> nodes, size_t lanes) noexcept
    : context_(context), nodes_(std::move(nodes)), lanes_(lanes) {}

CUresult Graph::supports(unsigned long long flags) const noexcept {
    // Captured graphs contain no allocation/free nodes. Per-node priorities
    // are vacuous only when there are no kernel nodes.
    if ((flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY) &&
        std::any_of(nodes_.begin(), nodes_.end(),
                    [](const GraphNode &entry) { return entry.node.kind == Kind::kernel; }))
        return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_SUCCESS;
}

CUresult Graph::launch(Scheduler &s, Key key) const {
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
    std::vector<Key> lanes(lanes_);
    lanes[0] = key;
    for (size_t i = 1; i < lanes_; ++i)
        lanes[i] = {ctx, reinterpret_cast<CUstream>(std::numeric_limits<std::uintptr_t>::max() - s.new_handle()), 0, false};

    // Use a private copy of the resource timelines for preflight so a failed
    // replay does not submit any work. Event waits can delay async allocation use.
    Time launch_time = VirtualClock::now();
    std::vector<Time> predicted(nodes_.size());
    std::vector<Time> lane_end(lanes_, s.queue.earliest(key, Kind::marker, {}, launch_time));
    auto *device = fake_cuda::virtual_core_device(s.queue.device_for(ctx));
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
        predicted[i] = at + QueueScheduler::duration(node.kind, node.bytes);
        lane_end[entry.lane] = predicted[i];
        if (available) *available = predicted[i];
    }
    std::vector<OpPtr> replay;
    replay.reserve(nodes_.size());
    s.queue.schedule(key, Kind::marker, 0, {}, launch_time);
    for (const GraphNode &entry : nodes_) {
        const Node &node = entry.node;
        OpPtr dependency;
        if (entry.event_dependency != no_node) dependency = replay[entry.event_dependency];
        else if (node.kind == Kind::wait) dependency = s.events.at(node.event).record;
        Key lane = lanes[entry.lane];
        OpPtr op = s.queue.schedule(lane, node.kind, node.bytes, dependency, launch_time);
        if (node.kind == Kind::record) s.events.at(node.event).record = op;
        replay.push_back(std::move(op));
    }
    // End-capture proved every side lane flows into the origin lane.
    s.queue.schedule(key, Kind::marker, 0, {}, launch_time);
    for (size_t i = 1; i < lanes.size(); ++i) s.queue.retire_stream(lanes[i]);
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
        if (node.kind == Kind::wait && event_context != key.context && capture->second->contains(key))
            return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        captured = capture->second->append(key, node, handle);
    }
    return CUDA_SUCCESS;
}
CUresult GraphManager::begin_capture(Scheduler &s, Key key) {
    if (captures_.contains(key.context))
        return capturing(key) ? CUDA_ERROR_STREAM_CAPTURE_UNMATCHED : CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    captures_.emplace(key.context, std::make_unique<Capture>(key, s.opaque<CUgraph>()));
    return CUDA_SUCCESS;
}
CUresult GraphManager::end_capture(Key key, CUgraph *graph) {
    auto it = captures_.find(key.context);
    if (it == captures_.end() || !it->second->matches_origin(key))
        return CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
    Capture &session = *it->second;
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
    *status = session ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
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
    executables_.emplace(handle, it->second);
    *exec = handle;
    return CUDA_SUCCESS;
}
CUresult GraphManager::launch(Scheduler &s, CUgraphExec exec, Key key) const {
    auto it = executables_.find(exec);
    if (it == executables_.end() || it->second->context() != key.context) return CUDA_ERROR_INVALID_HANDLE;
    if (capturing(key)) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    return it->second->launch(s, key);
}
CUresult GraphManager::destroy(CUcontext ctx, CUgraph graph) {
    auto it = graphs_.find(graph);
    if (it == graphs_.end() || it->second->context() != ctx) return CUDA_ERROR_INVALID_HANDLE;
    graphs_.erase(it);
    return CUDA_SUCCESS;
}
CUresult GraphManager::destroy_exec(CUcontext ctx, CUgraphExec exec) {
    auto it = executables_.find(exec);
    if (it == executables_.end() || it->second->context() != ctx) return CUDA_ERROR_INVALID_HANDLE;
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
        if (it->second->context() == ctx) it = executables_.erase(it); else ++it;
}
} // namespace fake_cuda::detail

using fake_cuda::detail::Scheduler;
using fake_cuda::detail::scheduler;

extern "C" {
CUresult virtual_stream_begin_capture(CUstream stream, CUstreamCaptureMode mode) {
    if (mode != CU_STREAM_CAPTURE_MODE_GLOBAL && mode != CU_STREAM_CAPTURE_MODE_THREAD_LOCAL &&
        mode != CU_STREAM_CAPTURE_MODE_RELAXED) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext, fake_cuda::detail::Key key) {
        return s.graph.begin_capture(s, key);
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
        *status = s.graph.capturing(key) ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
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
