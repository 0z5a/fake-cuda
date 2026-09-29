#ifndef FAKE_CUDA_DRIVER_API_ENTRY_POINT_ACCESS_H
#define FAKE_CUDA_DRIVER_API_ENTRY_POINT_ACCESS_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **function, int version, cuuint64_t flags);
CU_EXPORT CUresult CUDAAPI cuGetProcAddress_v2(const char *symbol, void **function, int version, cuuint64_t flags, CUdriverProcAddressQueryResult *status);
CU_EXPORT CUresult CUDAAPI cuGetExportTable(const void **table, const CUuuid *uuid);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_ENTRY_POINT_ACCESS_H */
