#ifndef FAKE_CUDA_IMPL_PRIVATE_H
#define FAKE_CUDA_IMPL_PRIVATE_H

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif
/* Internal C ABI: call from cuGetExportTable in cu_api.c. Not an exported Driver API. */
CUresult private_export_table(const void **table, const CUuuid *uuid);
/* Internal teardown hook. Call exactly once per destroyed context, without
 * holding the core registry mutex; callbacks may call back into the Driver.
 * No CLS operations may race with destruction of this context. */
void private_context_destroyed(CUcontext context);
#ifdef __cplusplus
}
#endif

#endif
