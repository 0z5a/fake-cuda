#ifndef FAKE_CUDA_IMPL_REGISTRY_H
#define FAKE_CUDA_IMPL_REGISTRY_H

#include "device.h"
#include "stream.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace fake_cuda {
// Owns all live handles. Registration, retirement, and lookup take the
// registry lock before the scheduler lock; draining and private callbacks
// happen unlocked.
class Registry {
public:
    Registry();
    Device *device(CUdevice ordinal) const noexcept {
        return Device::valid(ordinal) ? devices_[ordinal].get() : nullptr;
    }

    CUresult primary_get(CUcontext *out, CUdevice device, bool retain);
    CUresult primary_state(CUdevice device, unsigned int *flags, int *active);
    CUresult primary_release(CUdevice device);
    CUresult primary_flags(CUdevice device, unsigned int flags);
    CUresult primary_reset(CUdevice device);

    CUresult context_create(CUcontext *out, unsigned int flags, CUdevice device);
    CUresult context_destroy(CUcontext context);
    CUresult context_current(CUcontext *out) const;
    CUresult current_for_work(CUcontext *out) const;
    CUresult context_set(CUcontext context);
    CUresult context_push(CUcontext context);
    CUresult context_pop(CUcontext *out);
    CUresult context_device(CUdevice *out);
    CUresult context_flags(unsigned int *out);
    CUresult context_version(CUcontext context, unsigned int *out);
    CUresult context_device(CUcontext context, CUdevice *out);
    CUresult peer_enable(CUcontext peer, unsigned int flags);
    CUresult peer_disable(CUcontext peer);

    CUresult stream_create(CUstream *out, unsigned int flags, int priority);
    CUresult stream_destroy(CUstream stream);
    CUresult stream_context(CUstream stream, CUcontext *out);
    CUresult stream_device(CUstream stream, CUdevice *out);
    CUresult stream_flags(CUstream stream, unsigned int *out);
    CUresult stream_priority(CUstream stream, int *out);
    CUresult resolve_stream(CUstream stream, CUcontext *out, unsigned int *flags);

private:
    struct ThreadState {
        std::shared_ptr<Context> current;
        std::vector<std::shared_ptr<Context>> stack;
    };
    static thread_local ThreadState thread_;

    // These helpers require mutex_. Handles are monotonic integer tokens,
    // never dereferenced or reused when an object is retired.
    std::shared_ptr<Context> live_locked(CUcontext context) const;
    std::shared_ptr<Context> selected_locked() const;
    std::shared_ptr<Context> stream_owner_locked(CUstream stream) const;
    CUresult ensure_primary_locked(CUdevice device, std::shared_ptr<Context> *out);
    void retire_locked(const std::shared_ptr<Context>& context);

    std::mutex mutex_;
    std::uintptr_t next_handle_ = 4096;
    std::vector<std::shared_ptr<Device>> devices_;
    std::unordered_map<CUcontext, std::shared_ptr<Context>> contexts_;
    // Directed peer access: source context -> destination contexts.
    std::map<CUcontext, std::set<CUcontext>> peers_;
    std::unordered_map<CUstream, std::shared_ptr<Stream>> streams_;
};
} // namespace fake_cuda

#endif
