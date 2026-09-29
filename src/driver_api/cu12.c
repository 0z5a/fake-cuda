#include "bridge_internal.h"

/* Legacy entry points retained for CUDA 12 clients and older Driver ABIs. */
FORWARD(cuCtxCreate_v2, core_context_create,
        (CUcontext *context, unsigned int flags, CUdevice device), (context, flags, device))
FORWARD(cuEventElapsedTime, virtual_event_elapsed,
        (float *milliseconds, CUevent start, CUevent end), (milliseconds, start, end))
FORWARD(cuStreamGetCaptureInfo, virtual_stream_get_capture_info,
        (CUstream stream, CUstreamCaptureStatus *status, cuuint64_t *id),
        (stream, status, id, NULL, NULL, NULL, NULL))
FORWARD(cuStreamGetCaptureInfo_v2, virtual_stream_get_capture_info,
        (CUstream stream, CUstreamCaptureStatus *status, cuuint64_t *id, CUgraph *graph,
         const CUgraphNode **dependencies, size_t *count),
        (stream, status, id, graph, dependencies, NULL, count))
