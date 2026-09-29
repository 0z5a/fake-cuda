#include "impl/scheduler.h"

namespace fake_cuda::detail {
Scheduler &scheduler() {
    // CUDA Runtime teardown can run after C++ function-local statics are destroyed.
    static Scheduler &instance = *new Scheduler;
    return instance;
}

void Scheduler::reap() {
    const Time now = VirtualClock::now();
    queue.reap(now);
    memory.reap(now);
}
CUresult Scheduler::enqueue(Key key, Node node, OpPtr *result) {
    auto event = events.end();
    if (node.kind == Kind::wait || node.kind == Kind::record) {
        event = events.find(node.event);
        if (event == events.end() ||
            (node.kind == Kind::record && event->second.context != key.context))
            return CUDA_ERROR_INVALID_HANDLE;
    }
    if (graph.active(key.context)) {
        bool captured = false;
        CUresult status = graph.append(key, node,
                                       event == events.end() ? key.context : event->second.context,
                                       opaque<CUgraphNode>(), captured);
        if (status != CUDA_SUCCESS) return status;
        if (captured) return CUDA_SUCCESS;
    }
    OpPtr dependency = node.kind == Kind::wait ? event->second.record : OpPtr{};
    OpPtr op = queue.schedule(key, node.kind, node.bytes, dependency);
    if (node.kind == Kind::record) event->second.record = op;
    if (result) *result = std::move(op);
    return CUDA_SUCCESS;
}

CUresult Scheduler::begin_retire_context(CUcontext ctx) {
    return protect([&] {
        std::scoped_lock lock(mutex);
        return retired.insert(ctx).second ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
    });
}
void Scheduler::retire_stream(CUcontext ctx, CUstream stream) {
    destroyed_streams.emplace(stream, ctx);
    graph.retire_stream(ctx, key_for(ctx, stream));
    queue.retire_stream(key_for(ctx, stream));
}
void Scheduler::retire_context(CUcontext ctx) {
    retired.insert(ctx);
    memory.retire_context(ctx);
    for (auto it = destroyed_streams.begin(); it != destroyed_streams.end();)
        if (it->second == ctx) it = destroyed_streams.erase(it); else ++it;
    queue.retire_context(ctx);
    graph.retire_context(ctx);
    for (auto it = events.begin(); it != events.end();)
        if (it->second.context == ctx) it = events.erase(it); else ++it;

    for (auto it = modules.begin(); it != modules.end();)
        if (it->second == ctx) { auto mod = it->first;
            for (auto f = functions.begin(); f != functions.end();)
                if (f->second == mod) f = functions.erase(f); else ++f;
            it = modules.erase(it);
        } else ++it;
    for (auto &[handle, library] : libraries) { (void)handle; library.modules.erase(ctx); }
    for (auto it = library_functions.begin(); it != library_functions.end();)
        if (it->first.second == ctx) it = library_functions.erase(it); else ++it;
}
} // namespace fake_cuda::detail
