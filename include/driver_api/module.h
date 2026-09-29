#ifndef FAKE_CUDA_DRIVER_API_MODULE_H
#define FAKE_CUDA_DRIVER_API_MODULE_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuModuleGetLoadingMode(CUmoduleLoadingMode *mode);
CU_EXPORT CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image);
CU_EXPORT CUresult CUDAAPI cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name);
CU_EXPORT CUresult CUDAAPI cuModuleUnload(CUmodule module);
CU_EXPORT CUresult CUDAAPI cuLibraryLoadData(CUlibrary *library, const void *code,
                                             CUjit_option *jit_options, void **jit_values,
                                             unsigned int jit_count, CUlibraryOption *library_options,
                                             void **library_values, unsigned int library_count);
CU_EXPORT CUresult CUDAAPI cuLibraryUnload(CUlibrary library);
CU_EXPORT CUresult CUDAAPI cuLibraryGetKernel(CUkernel *kernel, CUlibrary library, const char *name);
CU_EXPORT CUresult CUDAAPI cuLibraryGetModule(CUmodule *module, CUlibrary library);
CU_EXPORT CUresult CUDAAPI cuKernelGetFunction(CUfunction *function, CUkernel kernel);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_MODULE_H */
