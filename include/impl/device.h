#ifndef FAKE_CUDA_IMPL_DEVICE_H
#define FAKE_CUDA_IMPL_DEVICE_H

#include <cuda.h>
#include "execution_queue.h"

#include <array>
#include <cstddef>
#include <memory>

namespace fake_cuda {
class Context;

class Device {
public:
    explicit Device(CUdevice ordinal) noexcept : ordinal_(ordinal) {}
    static constexpr std::size_t max_devices = 8;
    static bool valid(CUdevice ordinal) noexcept;
    static int count() noexcept;
    CUdevice ordinal() const noexcept { return ordinal_; }
    CUresult name(char *out, int size) const;
    CUresult uuid(CUuuid *out) const;
    CUresult memory(size_t *out) const;
    CUresult attribute(int *out, CUdevice_attribute attribute) const;

    // All contexts on this device share these resource queues.
    ExecutionQueue h2d_queue;
    ExecutionQueue d2h_queue;
    ExecutionQueue compute_queue;
    // Outgoing transfers: p2p_queues[peer] models this device -> peer.
    std::array<ExecutionQueue, max_devices> p2p_queues;

private:
    friend class Registry;
    CUdevice ordinal_ = 0;
    std::shared_ptr<Context> primary_;
};
} // namespace fake_cuda

#endif
