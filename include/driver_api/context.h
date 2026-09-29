#ifndef FAKE_CUDA_DRIVER_API_CONTEXT_H
#define FAKE_CUDA_DRIVER_API_CONTEXT_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuCtxCreate_v2(CUcontext *context, unsigned int flags, CUdevice device);
CU_EXPORT CUresult CUDAAPI cuCtxCreate_v4(CUcontext *context, CUctxCreateParams *params, unsigned int flags, CUdevice device);
CU_EXPORT CUresult CUDAAPI cuCtxDestroy_v2(CUcontext context);
CU_EXPORT CUresult CUDAAPI cuCtxGetCurrent(CUcontext *context);
CU_EXPORT CUresult CUDAAPI cuCtxSetCurrent(CUcontext context);
CU_EXPORT CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext context);
CU_EXPORT CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext *context);
CU_EXPORT CUresult CUDAAPI cuCtxGetDevice(CUdevice *device);
CU_EXPORT CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags);
CU_EXPORT CUresult CUDAAPI cuCtxGetApiVersion(CUcontext context, unsigned int *version);
CU_EXPORT CUresult CUDAAPI cuCtxSynchronize(void);
CU_EXPORT CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peer, unsigned int flags);
CU_EXPORT CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peer);
CU_EXPORT CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *least, int *greatest);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_CONTEXT_H */
