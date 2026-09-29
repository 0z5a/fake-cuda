#ifndef FAKE_CUDA_DRIVER_API_DEVICE_H
#define FAKE_CUDA_DRIVER_API_DEVICE_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuDeviceGetCount(int *count);
CU_EXPORT CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal);
CU_EXPORT CUresult CUDAAPI cuDeviceGetName(char *name, int size, CUdevice device);
CU_EXPORT CUresult CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice device);
CU_EXPORT CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice device);
CU_EXPORT CUresult CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice device);
CU_EXPORT CUresult CUDAAPI cuDeviceGetAttribute(int *value, CUdevice_attribute attribute, CUdevice device);
CU_EXPORT CUresult CUDAAPI cuDeviceCanAccessPeer(int *can_access, CUdevice device, CUdevice peer);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_DEVICE_H */
