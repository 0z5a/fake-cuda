#ifndef FAKE_CUDA_IMPL_STREAM_H
#define FAKE_CUDA_IMPL_STREAM_H

#include "context.h"

#include <cstdint>
#include <memory>

namespace fake_cuda {
class Stream {
public:
    Stream(std::shared_ptr<Context> context, unsigned int flags, int priority,
           std::uintptr_t handle_id) noexcept;

    CUstream handle() const noexcept { return reinterpret_cast<CUstream>(handle_id_); }
    const std::shared_ptr<Context>& context() const noexcept { return context_; }
    unsigned int flags() const noexcept { return flags_; }
    int priority() const noexcept { return priority_; }

private:
    std::uintptr_t handle_id_;
    std::shared_ptr<Context> context_;
    unsigned int flags_;
    int priority_;
};
} // namespace fake_cuda

#endif
