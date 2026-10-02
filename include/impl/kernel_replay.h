#ifndef FAKE_CUDA_IMPL_KERNEL_REPLAY_H
#define FAKE_CUDA_IMPL_KERNEL_REPLAY_H

#include "perf_model.h"

namespace fake_cuda {
// Fixed, topologically ordered trace: one host submission lane and one compute
// queue per device. Dependencies must name earlier invocations in this run.
struct ReplayInvocation {
    KernelQuery query;
    std::uint64_t stream;
    Nanoseconds arrival, host_service;
    std::vector<std::uint64_t> dependencies;
};
struct KernelInterval {
    std::uint64_t invocation_id;
    Nanoseconds host_start, submitted, ready, start, completion;
    PredictorResult prediction;
};
struct KernelReplayResult {
    std::vector<KernelInterval> intervals;
    TimingLedger timing;
    Nanoseconds makespan{};
    std::string error;
};
KernelReplayResult replay_kernel_trace(PerformanceModel &model, std::span<const ReplayInvocation> trace);
// Same fixed trace and conservative one-kernel/device admission, but resource
// order follows dependency readiness rather than future submission reservations.
KernelReplayResult replay_resource_trace(PerformanceModel &model, std::span<const ReplayInvocation> trace);
} // namespace fake_cuda
#endif
