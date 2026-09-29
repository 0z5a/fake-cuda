#ifndef FAKE_CUDA_DRIVER_API_INITIALIZATION_H
#define FAKE_CUDA_DRIVER_API_INITIALIZATION_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuInit(unsigned int flags);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_INITIALIZATION_H */
