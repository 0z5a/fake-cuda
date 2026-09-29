#ifndef FAKE_CUDA_DRIVER_API_VERSION_H
#define FAKE_CUDA_DRIVER_API_VERSION_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuDriverGetVersion(int *version);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_VERSION_H */
