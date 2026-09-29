#ifndef FAKE_CUDA_DRIVER_API_PRIMARY_CONTEXT_H
#define FAKE_CUDA_DRIVER_API_PRIMARY_CONTEXT_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *context, CUdevice device);
CU_EXPORT CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice device, unsigned int *flags, int *active);
CU_EXPORT CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice device);
CU_EXPORT CUresult CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice device, unsigned int flags);
CU_EXPORT CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice device);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_PRIMARY_CONTEXT_H */
