#include "impl/queue_scheduler.h"
#include "impl/device.h"
#include "impl/virtual_work_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <thread>
#include <utility>

namespace fake_cuda {
Device *virtual_core_device(CUdevice ordinal);
}

namespace fake_cuda::detail {
namespace {
ExecutionQueue *queue_for(Device &device, Kind kind) {
    switch (kind) {
    case Kind::h2d: return &device.h2d_queue;
    case Kind::d2h: return &device.d2h_queue;
    case Kind::compute:
    case Kind::kernel: return &device.compute_queue;
    default: return nullptr; // Marker/event operations reserve no resource queue.
    }
}
} // namespace

Key key_for(CUcontext context, CUstream stream, unsigned int flags) {
    if (stream == CU_STREAM_LEGACY) stream = nullptr;
    static std::atomic<std::uint64_t> next_thread{1};
    thread_local std::uint64_t thread = next_thread.fetch_add(1);
    return {context, stream, stream == CU_STREAM_PER_THREAD ? thread : 0,
            stream && stream != CU_STREAM_PER_THREAD && !(flags & CU_STREAM_NON_BLOCKING)};
}

static double bandwidth(const char *name, double fallback) {
    const char *text = std::getenv(name);
    if (!text || !*text) return fallback;
    char *end = nullptr;
    double value = std::strtod(text, &end);
    return end != text && *end == '\0' && std::isfinite(value) && value > 0 ? value : fallback;
}
VirtualClock::duration QueueScheduler::duration(Kind kind, size_t bytes) {
    if (kind == Kind::marker || kind == Kind::record || kind == Kind::wait) return VirtualClock::duration::zero();
    if (kind == Kind::kernel) return synthetic_kernel_prediction().service_time;
    double bw = kind == Kind::h2d ? bandwidth("FAKE_CUDA_H2D_BW_GBPS", 358.0) :
                kind == Kind::d2h ? bandwidth("FAKE_CUDA_D2H_BW_GBPS", 296.2) :
                kind == Kind::peer ? bandwidth("FAKE_CUDA_P2P_BW_GBPS", 50.0) : // modeling parameter, not measured
                bandwidth("FAKE_CUDA_HBM_BW_GBPS", 4800.0);
    // Decimal GB/s; round up to the next steady_clock tick.
    long double seconds = static_cast<long double>(bytes) / (static_cast<long double>(bw) * 1.0e9L);
    long double ticks = seconds / std::chrono::duration<long double>(VirtualClock::duration(1)).count();
    auto limit = static_cast<long double>(std::numeric_limits<VirtualClock::rep>::max() / 4);
    return VirtualClock::duration(static_cast<VirtualClock::rep>(std::ceil(std::min(ticks, limit))));
}

void QueueScheduler::register_context(CUcontext ctx, CUdevice ordinal) {
    devices.emplace(ctx, ordinal);
}
Time QueueScheduler::earliest_on(Key key, const ExecutionQueue *resource_queue,
                                 const OpPtr &dependency, Time current) const {
    Time start = current;
    auto it = tails.find(key);
    if (it != tails.end()) start = std::max(start, it->second->end);
    if (resource_queue) start = std::max(start, resource_queue->available_at());
    if (!key.stream) {
        auto barrier = blocking_end.find(key.context);
        if (barrier != blocking_end.end()) start = std::max(start, barrier->second);
    } else if (key.blocking) {
        auto legacy = tails.find(key_for(key.context, nullptr));
        if (legacy != tails.end()) start = std::max(start, legacy->second->end);
    }
    if (dependency) start = std::max(start, dependency->end);
    return start;
}
Time QueueScheduler::earliest(Key key, Kind kind, const OpPtr &dependency, Time current) const {
    auto *resource_queue = queue_for(*virtual_core_device(device_for(key.context)), kind);
    return earliest_on(key, resource_queue, dependency, current);
}
Time QueueScheduler::earliest_peer(Key key, CUdevice src, CUdevice dst,
                                   const OpPtr &dependency, Time current) const {
    device_for(key.context);
    if (src == dst)
        return earliest_on(key, &virtual_core_device(dst)->compute_queue, dependency, current);
    return earliest_on(key, &virtual_core_device(src)->p2p_queues[dst], dependency, current);
}
OpPtr QueueScheduler::schedule_on(Key key, Kind kind, size_t bytes, const OpPtr &dependency,
                                  Time current, ExecutionQueue *resource_queue, const PredictorResult *prediction) {
    auto op = std::make_shared<Op>();
    op->context = key.context;
    op->stream = key.stream;
    auto previous = tails.find(key);
    if (previous != tails.end()) op->dependencies.push_back(previous->second);
    if (dependency) op->dependencies.push_back(dependency);
    if (!key.stream) {
        if (auto prior = blocking_last[key.context].lock()) op->dependencies.push_back(prior);
    } else if (key.blocking) {
        auto legacy = tails.find(key_for(key.context, nullptr));
        if (legacy != tails.end()) op->dependencies.push_back(legacy->second);
    }
    if (resource_queue) {
        if (auto prior = resource_queue->last_operation())
            if (prior != (previous == tails.end() ? OpPtr{} : previous->second))
                op->dependencies.push_back(std::move(prior));
    }
    op->start = earliest_on(key, resource_queue, dependency, current);
    if (prediction) op->prediction = *prediction;
    op->end = op->start + (prediction ? prediction->service_time : duration(kind, bytes));
    pending.emplace(op->end, std::make_pair(key, op));
    include_context(key.context, op->end);
    tails[key] = op;
    if (key.blocking && blocking_end[key.context] <= op->end) {
        blocking_end[key.context] = op->end;
        blocking_last[key.context] = op;
    }
    if (resource_queue) resource_queue->reserve(op);
    return op;
}
OpPtr QueueScheduler::schedule(Key key, Kind kind, size_t bytes, const OpPtr &dependency,
                               Time current, const PredictorResult *prediction) {
    auto *resource_queue = queue_for(*virtual_core_device(device_for(key.context)), kind);
    return schedule_on(key, kind, bytes, dependency, current, resource_queue, prediction);
}
OpPtr QueueScheduler::schedule_peer(Key key, CUdevice src, CUdevice dst, size_t bytes,
                                    const OpPtr &dependency, Time current) {
    device_for(key.context);
    if (src == dst)
        return schedule_on(key, Kind::compute, bytes, dependency, current,
                           &virtual_core_device(dst)->compute_queue);
    return schedule_on(key, Kind::peer, bytes, dependency, current,
                       &virtual_core_device(src)->p2p_queues[dst]);
}
void QueueScheduler::include_context(CUcontext ctx, Time end) {
    device_for(ctx);
    outstanding[ctx] = std::max(outstanding[ctx], end);
}
void QueueScheduler::reap(Time now) {
    while (!pending.empty() && pending.begin()->first <= now) {
        auto it = pending.begin();
        auto tail = tails.find(it->second.first);
        if (tail != tails.end() && tail->second == it->second.second) tails.erase(tail);
        pending.erase(it);
    }
}
Time QueueScheduler::stream_end(Key key) const {
    if (auto it = tails.find(key); it != tails.end()) return it->second->end;
    return {};
}
Time QueueScheduler::context_end(CUcontext ctx) const {
    if (auto it = outstanding.find(ctx); it != outstanding.end()) return it->second;
    return {};
}
void QueueScheduler::retire_context(CUcontext ctx) {
    for (auto it = tails.begin(); it != tails.end();)
        if (it->first.context == ctx) it = tails.erase(it); else ++it;
    for (auto it = pending.begin(); it != pending.end();)
        if (it->second.second->context == ctx) it = pending.erase(it); else ++it;
    outstanding.erase(ctx);
    devices.erase(ctx);
    blocking_end.erase(ctx);
    blocking_last.erase(ctx);
}
} // namespace fake_cuda::detail
