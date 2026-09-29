#ifndef FAKE_CUDA_DRIVER_API_MEMORY_H
#define FAKE_CUDA_DRIVER_API_MEMORY_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes);
CU_EXPORT CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *ptr, size_t bytes);
CU_EXPORT CUresult CUDAAPI cuMemFree_v2(CUdeviceptr ptr);
CU_EXPORT CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *ptr, size_t bytes, CUstream stream);
CU_EXPORT CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr ptr, CUstream stream);
CU_EXPORT CUresult CUDAAPI cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
CU_EXPORT CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes);
CU_EXPORT CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream);
CU_EXPORT CUresult CUDAAPI cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes);
CU_EXPORT CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t bytes, CUstream stream);
CU_EXPORT CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes);
CU_EXPORT CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream);
CU_EXPORT CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dst, CUcontext dst_context,
                                           CUdeviceptr src, CUcontext src_context, size_t bytes);
CU_EXPORT CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dst, CUcontext dst_context,
                                                CUdeviceptr src, CUcontext src_context,
                                                size_t bytes, CUstream stream);
CU_EXPORT CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t count);
CU_EXPORT CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dst, unsigned char value, size_t count, CUstream stream);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_MEMORY_H */
