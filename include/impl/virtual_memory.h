#ifndef FAKE_CUDA_IMPL_VIRTUAL_MEMORY_H
#define FAKE_CUDA_IMPL_VIRTUAL_MEMORY_H

#include "virtual_types.h"

#include <array>
#include <map>

namespace fake_cuda::detail {
struct Allocation {
    CUcontext context;
    CUdevice device;
    size_t size;
    Time ready;
    Time released = Time::max();
    bool freeing = false;
};

class VirtualMemory {
public:
    size_t free_bytes(CUdevice ordinal) const;
    bool can_allocate(CUdevice ordinal, size_t bytes) const;
    CUresult allocate(CUcontext ctx, CUdevice ordinal, CUdeviceptr *ptr, size_t bytes, Time ready);
    CUresult check_bounds(CUcontext ctx, CUdeviceptr ptr, size_t bytes) const;
    CUresult check_pointer(CUcontext ctx, CUdeviceptr ptr, size_t bytes, Time when) const;
    CUresult attribute(CUcontext ctx, void *data, CUpointer_attribute attribute, CUdeviceptr ptr) const;
    CUresult free(CUcontext ctx, CUdeviceptr ptr, Time &finish);
    CUresult can_free_async(CUcontext ctx, CUdeviceptr ptr, Time earliest) const;
    void free_async(CUdeviceptr ptr, Time finish);
    void reap(Time now);
    void retire_context(CUcontext ctx);
private:
    std::multimap<Time, CUdeviceptr> releases;
    std::map<CUdeviceptr, Allocation> allocations;
    std::uint64_t next_address = 0x100000000000ULL;
    std::array<size_t, 8> used{};
};
} // namespace fake_cuda::detail

#endif
