#ifndef FAKE_CUDA_IMPL_CORE_H
#define FAKE_CUDA_IMPL_CORE_H

#include <cuda.h>
#include <stddef.h>

/* Stable C ABI between the exported Driver entry points and the C++ object model. */
#ifdef __cplusplus
extern "C" {
#endif
CUresult core_init(unsigned int flags);
CUresult core_version(int *version);
CUresult core_count(int *count);
CUresult core_device(CUdevice *device, int ordinal);
CUresult core_name(char *name, int size, CUdevice device);
CUresult core_uuid(CUuuid *uuid, CUdevice device);
CUresult core_memory(size_t *bytes, CUdevice device);
CUresult core_attribute(int *value, CUdevice_attribute attribute, CUdevice device);
CUresult core_can_access_peer(int *can_access, CUdevice device, CUdevice peer);
CUresult core_primary_retain(CUcontext *context, CUdevice device);
CUresult core_primary_get(CUcontext *context, CUdevice device);
CUresult core_primary_state(CUdevice device, unsigned int *flags, int *active);
CUresult core_primary_release(CUdevice device);
CUresult core_primary_flags(CUdevice device, unsigned int flags);
CUresult core_primary_reset(CUdevice device);
CUresult core_context_create(CUcontext *context, unsigned int flags, CUdevice device);
CUresult core_context_destroy(CUcontext context);
CUresult core_context_current(CUcontext *context);
CUresult core_context_set(CUcontext context);
CUresult core_context_push(CUcontext context);
CUresult core_context_pop(CUcontext *context);
CUresult core_context_device(CUdevice *device);
CUresult core_context_flags(unsigned int *flags);
CUresult core_context_version(CUcontext context, unsigned int *version);
CUresult core_context_sync(void);
CUresult core_context_enable_peer(CUcontext peer, unsigned int flags);
CUresult core_context_disable_peer(CUcontext peer);
CUresult core_stream_create(CUstream *stream, unsigned int flags, int priority);
CUresult core_stream_destroy(CUstream stream);
CUresult core_stream_context(CUstream stream, CUcontext *context);
CUresult core_stream_device(CUstream stream, CUdevice *device);
CUresult core_stream_flags(CUstream stream, unsigned int *flags);
CUresult core_stream_priority(CUstream stream, int *priority);
CUresult core_stream_sync(CUstream stream);
#ifdef __cplusplus
}

#include "execution_queue.h"
#include "device.h"
#include "context.h"
#include "stream.h"
#include "registry.h"

#endif

#endif
