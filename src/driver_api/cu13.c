#include "bridge_internal.h"

/* CUDA 13 selects the four-argument context-create and extended capture ABI. */
CUresult CUDAAPI cuCtxCreate_v4(CUcontext *context, CUctxCreateParams *params,
                                unsigned int flags, CUdevice device) {
    if (params) return CUDA_ERROR_NOT_SUPPORTED;
    return core_context_create(context, flags, device);
}
FORWARD(cuEventElapsedTime_v2, virtual_event_elapsed,
        (float *milliseconds, CUevent start, CUevent end), (milliseconds, start, end))
FORWARD(cuStreamGetCaptureInfo_v3, virtual_stream_get_capture_info,
        (CUstream stream, CUstreamCaptureStatus *status, cuuint64_t *id, CUgraph *graph,
         const CUgraphNode **dependencies, const CUgraphEdgeData **edges, size_t *count),
        (stream, status, id, graph, dependencies, edges, count))
