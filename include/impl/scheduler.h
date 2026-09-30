#ifndef FAKE_CUDA_IMPL_SCHEDULER_H
#define FAKE_CUDA_IMPL_SCHEDULER_H

#include "graph.h"
#include "queue_scheduler.h"
#include "virtual_memory.h"
#include "virtual_work_internal.h"

#include <mutex>
#include <new>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace fake_cuda::detail {
class Scheduler {
public:
    // Mutable scheduling state and device resource queues are accessed under mutex.
    std::mutex mutex;
    QueueScheduler queue;
    std::shared_ptr<PerformanceModel> performance_model = std::make_shared<SyntheticConstant>();
    TimingLedger timing;
    std::string prediction_error;
    VirtualMemory memory;
    GraphManager graph;
    std::unordered_map<CUevent, Event> events;
    std::unordered_map<CUmodule, ModuleRecord> modules;
    std::unordered_map<CUfunction, FunctionRecord> functions;
    struct Library {
        std::unordered_map<std::string, CUkernel> kernels;
        std::unordered_map<CUcontext, CUmodule> modules;
    };
    // Libraries and kernel handles are process-wide; modules/functions are per-context.
    std::unordered_map<CUlibrary, Library> libraries;
    std::unordered_map<CUkernel, KernelRecord> kernels;
    std::map<std::pair<CUkernel, CUcontext>, CUfunction> library_functions;
    std::set<CUcontext> retired;
    std::map<CUstream, CUcontext> destroyed_streams;


    void reap();
    CUresult enqueue(Key key, Node node, OpPtr *result = nullptr);
    CUresult predict(std::span<const KernelQuery> queries, std::vector<PredictorResult> &results);
    void commit_predictions(std::span<const PredictorResult> results);

    CUresult begin_retire_context(CUcontext ctx);
    void retire_stream(CUcontext ctx, CUstream stream);
    void retire_context(CUcontext ctx);

    // Handles are monotonically generated identities, not owned pointers.
    std::uintptr_t new_handle() noexcept { return ++next_handle_; }
    std::uint64_t new_invocation() noexcept { return ++next_invocation_; }
    template <typename T> T opaque() noexcept { return reinterpret_cast<T>(new_handle()); }

    template <typename F> CUresult in_context(F &&f) {
        CUcontext context = nullptr;
        CUresult r = virtual_core_current(&context);
        if (r != CUDA_SUCCESS) return r;
        return protect([&] {
            std::scoped_lock lock(mutex);
            if (retired.contains(context)) return CUDA_ERROR_INVALID_CONTEXT;
            reap();
            return std::forward<F>(f)(*this, context);
        });
    }
    template <typename F> CUresult in_stream(CUstream stream, F &&f) {
        CUcontext context = nullptr;
        unsigned int flags = 0;
        CUresult r = virtual_core_resolve_flags(stream, &context, &flags);
        if (r != CUDA_SUCCESS) return r;
        return protect([&] {
            std::scoped_lock lock(mutex);
            if (retired.contains(context)) return CUDA_ERROR_INVALID_CONTEXT;
            if (stream && stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD &&
                destroyed_streams.contains(stream)) return CUDA_ERROR_INVALID_HANDLE;
            reap();
            return std::forward<F>(f)(*this, context, key_for(context, stream, flags));
        });
    }

private:
    std::uintptr_t next_handle_ = 0x100000;
    std::uint64_t next_invocation_ = 0;
    template <typename F> static CUresult protect(F &&f) {
        try { return std::forward<F>(f)(); }
        catch (const std::bad_alloc &) { return CUDA_ERROR_OUT_OF_MEMORY; }
    }
};

Scheduler &scheduler();
} // namespace fake_cuda::detail

#endif
