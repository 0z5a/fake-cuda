#ifndef FAKE_CUDA_DRIVER_API_STREAM_H
#define FAKE_CUDA_DRIVER_API_STREAM_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuStreamCreate(CUstream *stream, unsigned int flags);
CU_EXPORT CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority);
CU_EXPORT CUresult CUDAAPI cuStreamDestroy_v2(CUstream stream);
CU_EXPORT CUresult CUDAAPI cuStreamGetCtx(CUstream stream, CUcontext *context);
CU_EXPORT CUresult CUDAAPI cuStreamGetDevice(CUstream stream, CUdevice *device);
CU_EXPORT CUresult CUDAAPI cuStreamGetFlags(CUstream stream, unsigned int *flags);
CU_EXPORT CUresult CUDAAPI cuStreamGetPriority(CUstream stream, int *priority);
CU_EXPORT CUresult CUDAAPI cuStreamQuery(CUstream stream);
CU_EXPORT CUresult CUDAAPI cuStreamSynchronize(CUstream stream);
CU_EXPORT CUresult CUDAAPI cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags);
CU_EXPORT CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream stream, CUstreamCaptureMode mode);
CU_EXPORT CUresult CUDAAPI cuStreamEndCapture(CUstream stream, CUgraph *graph);
CU_EXPORT CUresult CUDAAPI cuStreamIsCapturing(CUstream stream, CUstreamCaptureStatus *status);
CU_EXPORT CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream stream, CUstreamCaptureStatus *status,
                                                   cuuint64_t *id);
CU_EXPORT CUresult CUDAAPI cuStreamGetCaptureInfo_v2(CUstream stream, CUstreamCaptureStatus *status,
                                                      cuuint64_t *id, CUgraph *graph,
                                                      const CUgraphNode **dependencies, size_t *count);
CU_EXPORT CUresult CUDAAPI cuStreamGetCaptureInfo_v3(CUstream stream, CUstreamCaptureStatus *status,
                                                      cuuint64_t *id, CUgraph *graph,
                                                      const CUgraphNode **dependencies,
                                                      const CUgraphEdgeData **edges, size_t *count);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_STREAM_H */
