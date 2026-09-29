#ifndef FAKE_CUDA_IMPL_CONTEXT_H
#define FAKE_CUDA_IMPL_CONTEXT_H

#include <cuda.h>
#include <atomic>
#include <cstdint>
#include <memory>

namespace fake_cuda {
class Device;

class Context {
public:
    Context(std::shared_ptr<Device> device, bool primary, unsigned int flags,
            std::uintptr_t handle_id) noexcept;

    CUcontext handle() const noexcept { return reinterpret_cast<CUcontext>(handle_id_); }
    CUdevice device_ordinal() const noexcept;
    bool is_primary() const noexcept { return primary_; }
    unsigned int flags() const noexcept { return flags_; }
    void set_flags(unsigned int flags) noexcept { flags_ = flags; }
    unsigned int retains() const noexcept { return retains_; }
    void retain() noexcept { ++retains_; }
    void release() noexcept { --retains_; }
    bool retiring() const noexcept { return retiring_.load(std::memory_order_acquire); }
    void begin_retirement() noexcept { retiring_.store(true, std::memory_order_release); }

private:
    std::uintptr_t handle_id_;
    std::weak_ptr<Device> device_;
    bool primary_;
    unsigned int flags_;
    unsigned int retains_ = 0;
    // Other threads may still have this context in their thread-local stack.
    std::atomic<bool> retiring_{false};
};
} // namespace fake_cuda

#endif
