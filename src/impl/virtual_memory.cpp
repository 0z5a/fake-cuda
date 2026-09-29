#include "impl/virtual_memory.h"
#include "profile/gh200.h"

#include <algorithm>
#include <limits>

namespace fake_cuda::detail {
size_t VirtualMemory::free_bytes(CUdevice ordinal) const { return gh200::memory_bytes - used[ordinal]; }
bool VirtualMemory::can_allocate(CUdevice ordinal, size_t bytes) const {
    return bytes <= free_bytes(ordinal) &&
           bytes <= std::numeric_limits<std::uint64_t>::max() - next_address - 256;
}
void VirtualMemory::reap(Time now) {
    while (!releases.empty() && releases.begin()->first <= now) {
        auto it = releases.begin();
        auto allocation = allocations.find(it->second);
        if (allocation != allocations.end() && allocation->second.released == it->first) {
            used[allocation->second.device] -= allocation->second.size;
            allocations.erase(allocation);
        }
        releases.erase(it);
    }
}

CUresult VirtualMemory::check_bounds(CUcontext ctx, CUdeviceptr ptr, size_t bytes) const {
    auto it = allocations.upper_bound(ptr);
    if (it == allocations.begin()) return CUDA_ERROR_INVALID_VALUE;
    --it;
    const Allocation &a = it->second;
    return a.context == ctx && ptr >= it->first && ptr - it->first <= a.size &&
           bytes <= a.size - static_cast<size_t>(ptr - it->first) && !a.freeing
           ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult VirtualMemory::check_pointer(CUcontext ctx, CUdeviceptr ptr, size_t bytes, Time when) const {
    auto it = allocations.upper_bound(ptr);
    if (it == allocations.begin()) return CUDA_ERROR_INVALID_VALUE;
    --it;
    const Allocation &a = it->second;
    if (a.context != ctx || ptr < it->first || ptr - it->first > a.size ||
        bytes > a.size - static_cast<size_t>(ptr - it->first) || when < a.ready || a.freeing || when >= a.released)
        return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}
CUresult VirtualMemory::allocate(CUcontext ctx, CUdevice ordinal, CUdeviceptr *ptr, size_t bytes, Time ready) {
    if (!ptr || !bytes) return CUDA_ERROR_INVALID_VALUE;
    if (bytes > free_bytes(ordinal)) return CUDA_ERROR_OUT_OF_MEMORY;
    constexpr std::uint64_t alignment = 256;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - next_address - alignment)
        return CUDA_ERROR_OUT_OF_MEMORY;
    auto address = next_address;
    allocations.emplace(static_cast<CUdeviceptr>(address), Allocation{ctx, ordinal, bytes, ready});
    next_address += (static_cast<std::uint64_t>(bytes) + alignment - 1) / alignment * alignment + alignment;
    used[ordinal] += bytes;
    *ptr = static_cast<CUdeviceptr>(address);
    return CUDA_SUCCESS;
}
CUresult VirtualMemory::attribute(CUcontext ctx, void *data, CUpointer_attribute attribute,
                                  CUdeviceptr ptr) const {
    auto it = allocations.upper_bound(ptr);
    if (it == allocations.begin()) return CUDA_ERROR_INVALID_VALUE;
    --it;
    const Allocation &a = it->second;
    if (a.context != ctx || a.freeing || ptr < it->first || ptr - it->first >= a.size)
        return CUDA_ERROR_INVALID_VALUE;
    switch (attribute) {
    case CU_POINTER_ATTRIBUTE_CONTEXT: *static_cast<CUcontext *>(data) = a.context; break;
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: *static_cast<CUmemorytype *>(data) = CU_MEMORYTYPE_DEVICE; break;
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: *static_cast<CUdeviceptr *>(data) = ptr; break;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: *static_cast<int *>(data) = a.device; break;
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
    case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
    case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
        *static_cast<unsigned int *>(data) = 0; break;
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR: *static_cast<CUdeviceptr *>(data) = it->first; break;
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE: *static_cast<size_t *>(data) = a.size; break;
    case CU_POINTER_ATTRIBUTE_MAPPED: *static_cast<unsigned int *>(data) = 1; break;
    default: return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}
CUresult VirtualMemory::free(CUcontext ctx, CUdeviceptr ptr, Time &finish) {
    auto it = allocations.find(ptr);
    if (it == allocations.end() || it->second.context != ctx || it->second.freeing)
        return CUDA_ERROR_INVALID_VALUE;
    finish = std::max(finish, it->second.ready);
    releases.emplace(finish, ptr);
    it->second.freeing = true;
    it->second.released = finish;
    return CUDA_SUCCESS;
}
CUresult VirtualMemory::can_free_async(CUcontext ctx, CUdeviceptr ptr, Time earliest) const {
    auto it = allocations.find(ptr);
    return it == allocations.end() || it->second.context != ctx || it->second.freeing ||
           earliest < it->second.ready ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
void VirtualMemory::free_async(CUdeviceptr ptr, Time finish) {
    releases.emplace(finish, ptr);
    auto &allocation = allocations.at(ptr);
    allocation.released = finish;
    allocation.freeing = true;
}
void VirtualMemory::retire_context(CUcontext ctx) {
    for (auto it = releases.begin(); it != releases.end();)
        if (auto allocation = allocations.find(it->second);
            allocation != allocations.end() && allocation->second.context == ctx)
            it = releases.erase(it);
        else ++it;
    for (auto it = allocations.begin(); it != allocations.end();)
        if (it->second.context == ctx) {
            used[it->second.device] -= it->second.size;
            it = allocations.erase(it);
        }
        else ++it;
}
} // namespace fake_cuda::detail
