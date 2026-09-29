#ifndef FAKE_CUDA_DRIVER_API_BRIDGE_INTERNAL_H
#define FAKE_CUDA_DRIVER_API_BRIDGE_INTERNAL_H

#include "driver_api/all.h"
#include "impl/core.h"
#include "impl/private.h"
#include "impl/virtual_work.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Use exported Driver ABI names rather than cuda.h's version/PTDS aliases. */
#ifdef cuMemsetD8Async
#undef cuMemsetD8Async
#endif
#ifdef cuEventRecord
#undef cuEventRecord
#endif
#ifdef cuEventElapsedTime
#undef cuEventElapsedTime
#endif
#ifdef cuStreamWaitEvent
#undef cuStreamWaitEvent
#endif
#ifdef cuStreamEndCapture
#undef cuStreamEndCapture
#endif
#ifdef cuStreamIsCapturing
#undef cuStreamIsCapturing
#endif
#ifdef cuGraphInstantiateWithParams
#undef cuGraphInstantiateWithParams
#endif
#ifdef cuGraphLaunch
#undef cuGraphLaunch
#endif
#ifdef cuMemAllocAsync
#undef cuMemAllocAsync
#endif
#ifdef cuMemFreeAsync
#undef cuMemFreeAsync
#endif
#ifdef cuLaunchKernel
#undef cuLaunchKernel
#endif

/* CUDA 13 headers only declare the macro-mapped _v2 name; retain the CUDA 12 ABI. */
CU_EXPORT CUresult CUDAAPI cuEventElapsedTime(float *milliseconds, CUevent start, CUevent end);

#define FORWARD(api, core, signature, arguments) \
    CUresult CUDAAPI api signature { \
        CUresult result = core arguments; \
        if (getenv("FAKE_CUDA_TRACE_CALLS")) \
            fprintf(stderr, "fake-cuda: call %s => %d\n", #api, (int)result); \
        return result; \
    }

#endif /* FAKE_CUDA_DRIVER_API_BRIDGE_INTERNAL_H */
