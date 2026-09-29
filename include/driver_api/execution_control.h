#ifndef FAKE_CUDA_DRIVER_API_EXECUTION_CONTROL_H
#define FAKE_CUDA_DRIVER_API_EXECUTION_CONTROL_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuLaunchKernel(CUfunction function, unsigned int gridX, unsigned int gridY,
                                         unsigned int gridZ, unsigned int blockX, unsigned int blockY,
                                         unsigned int blockZ, unsigned int sharedMem, CUstream stream,
                                         void **params, void **extra);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_EXECUTION_CONTROL_H */
