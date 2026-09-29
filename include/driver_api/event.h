#ifndef FAKE_CUDA_DRIVER_API_EVENT_H
#define FAKE_CUDA_DRIVER_API_EVENT_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuEventCreate(CUevent *event, unsigned int flags);
CU_EXPORT CUresult CUDAAPI cuEventDestroy_v2(CUevent event);
CU_EXPORT CUresult CUDAAPI cuEventRecord(CUevent event, CUstream stream);
CU_EXPORT CUresult CUDAAPI cuEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags);
CU_EXPORT CUresult CUDAAPI cuEventQuery(CUevent event);
CU_EXPORT CUresult CUDAAPI cuEventSynchronize(CUevent event);
CU_EXPORT CUresult CUDAAPI cuEventElapsedTime_v2(float *milliseconds, CUevent start, CUevent end);
#if CUDA_VERSION < 13000
CU_EXPORT CUresult CUDAAPI cuEventElapsedTime(float *milliseconds, CUevent start, CUevent end);
#endif

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_EVENT_H */
