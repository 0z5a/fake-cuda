#ifndef FAKE_CUDA_IMPL_EXECUTION_QUEUE_H
#define FAKE_CUDA_IMPL_EXECUTION_QUEUE_H

#include "kernel_launch.h"

#include <chrono>
#include <memory>
#include <vector>
#include <cuda.h>

namespace fake_cuda {
using VirtualClock = std::chrono::steady_clock;
struct VirtualOperation {
    std::shared_ptr<const KernelLaunch> launch;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    // Dependency edges are observational; the scheduler retains unfinished
    // operations separately, so completed chains need not remain alive.
    std::vector<std::weak_ptr<VirtualOperation>> dependencies;
    VirtualClock::time_point start, end;
};
// Accessed only while holding the virtual scheduler's mutex.
class ExecutionQueue {
public:
    VirtualClock::time_point available_at() const noexcept { return available_; }
    std::shared_ptr<VirtualOperation> last_operation() const noexcept { return last_.lock(); }
    void reserve(const std::shared_ptr<VirtualOperation> &operation) noexcept {
        available_ = operation->end;
        last_ = operation;
    }

private:
    // A queue tracks ordering but does not own the operation's lifetime.
    VirtualClock::time_point available_{};
    std::weak_ptr<VirtualOperation> last_;
};
} // namespace fake_cuda

#endif
