#ifndef FAKE_CUDA_DRIVER_API_ERROR_H
#define FAKE_CUDA_DRIVER_API_ERROR_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuGetErrorName(CUresult error, const char **name);
CU_EXPORT CUresult CUDAAPI cuGetErrorString(CUresult error, const char **description);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_ERROR_H */
