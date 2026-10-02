#ifndef FAKE_CUDA_IMPL_VIRTUAL_WORK_INTERNAL_H
#define FAKE_CUDA_IMPL_VIRTUAL_WORK_INTERNAL_H

#include <cuda.h>

// Core resolves opaque handles under its registry lock. The virtual scheduler
// maintains its own lock; it never holds it while entering core.
extern "C" CUresult virtual_core_resolve(CUstream stream, CUcontext *context);
extern "C" CUresult virtual_core_resolve_flags(CUstream stream, CUcontext *context,
                                                unsigned int *flags);
extern "C" CUresult virtual_core_current(CUcontext *context);
extern "C" CUresult virtual_core_context_device(CUcontext context, CUdevice *device);
// Called with Registry's lock held; scheduler's lock is acquired inside.
extern "C" CUresult virtual_register_context(CUcontext context, CUdevice device);

namespace fake_cuda {
struct Device;
Device *virtual_core_device(CUdevice ordinal);
CUresult virtual_synchronize_stream(CUcontext context, CUstream stream, bool query);
CUresult virtual_synchronize_context(CUcontext context);
// Called under the registry lock: excludes already-resolved calls from enqueueing.
CUresult virtual_begin_retire_context(CUcontext context);
// Registry validates live contexts/capabilities before taking the scheduler lock.
CUresult virtual_enable_peer(CUcontext context, CUcontext peer);
CUresult virtual_disable_peer(CUcontext context, CUcontext peer);
void virtual_drain_context(CUcontext context);
void virtual_retire_stream(CUcontext context, CUstream stream);
void virtual_retire_context(CUcontext context);
}
#endif
