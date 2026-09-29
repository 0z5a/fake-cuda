#ifndef FAKE_CUDA_IMPL_DEVICE_H
#define FAKE_CUDA_IMPL_DEVICE_H

#include <cuda.h>
#include "execution_queue.h"

#include "profile/device_profile.h"

#include <map>
#include <cstddef>
#include <memory>

namespace fake_cuda {
class Context;

class Device {
public:
    explicit Device(CUdevice ordinal) noexcept
        : ordinal_(ordinal), profile_(device_configuration().profile) {}
    const DeviceProfile &profile() const noexcept { return profile_; }
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
    std::map<CUdevice, ExecutionQueue> p2p_queues;

private:
    friend class Registry;
    CUdevice ordinal_ = 0;
    const DeviceProfile &profile_;
    std::shared_ptr<Context> primary_;
};
} // namespace fake_cuda

#endif
