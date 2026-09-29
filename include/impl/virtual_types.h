#ifndef FAKE_CUDA_IMPL_VIRTUAL_TYPES_H
#define FAKE_CUDA_IMPL_VIRTUAL_TYPES_H

#include "execution_queue.h"

#include <cstddef>
#include <cstdint>


#include <memory>
#include <tuple>


namespace fake_cuda::detail {
using Time = VirtualClock::time_point;
using Op = VirtualOperation;
using OpPtr = std::shared_ptr<Op>;

struct Key {
    // Opaque core handles are borrowed identities; the core owns their lifetime.
    CUcontext context;
    CUstream stream;
    std::uint64_t thread;
    bool blocking = false; // immutable stream property, not part of its identity
    bool operator<(const Key &rhs) const {
        return std::tie(context, stream, thread) < std::tie(rhs.context, rhs.stream, rhs.thread);
    }
};
Key key_for(CUcontext context, CUstream stream, unsigned int flags = 0);

enum class Kind { h2d, d2h, compute, kernel, marker, record, wait, peer };
struct Node {
    Kind kind;
    size_t bytes = 0;
    CUevent event = nullptr;
    CUdeviceptr first = 0, second = 0;
    int priority = 0;
    std::shared_ptr<const KernelLaunch> launch = nullptr;
};

struct Event {
    CUcontext context;
    unsigned int flags;
    OpPtr record;
};

} // namespace fake_cuda::detail

#endif
