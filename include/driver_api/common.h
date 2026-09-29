#ifndef FAKE_CUDA_DRIVER_API_COMMON_H
#define FAKE_CUDA_DRIVER_API_COMMON_H

#include <cuda.h>
#include <stddef.h>

/* cuda.h maps these unversioned names to versioned entry points. Remove the
 * aliases so declarations and definitions use this library's actual ABI. */
#ifdef cuCtxCreate
#undef cuCtxCreate
#endif
#ifdef cuCtxDestroy
#undef cuCtxDestroy
#endif
#ifdef cuCtxPushCurrent
#undef cuCtxPushCurrent
#endif
#ifdef cuCtxPopCurrent
#undef cuCtxPopCurrent
#endif
#ifdef cuDeviceGetUuid
#undef cuDeviceGetUuid
#endif
#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif
#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
#ifdef cuDevicePrimaryCtxSetFlags
#undef cuDevicePrimaryCtxSetFlags
#endif
#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif
#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif
#ifdef cuStreamGetCaptureInfo
#undef cuStreamGetCaptureInfo
#endif
#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

#ifndef CU_EXPORT
#if defined(__GNUC__)
#define CU_EXPORT __attribute__((visibility("default")))
#else
#define CU_EXPORT
#endif
#endif

#endif /* FAKE_CUDA_DRIVER_API_COMMON_H */
