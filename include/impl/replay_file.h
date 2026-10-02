#ifndef FAKE_CUDA_IMPL_REPLAY_FILE_H
#define FAKE_CUDA_IMPL_REPLAY_FILE_H

#include "perf_model.h"

#include <istream>

namespace fake_cuda {
// The caller verifies these external identities against its known code bytes,
// hardware and measurement conditions; opaque Driver handles cannot do that.
struct ReplayBinding {
    std::string id, code_identity, hardware_identity, conditions;
    CUcontext context;
    CUdevice device;
    const DeviceProfile *profile;
    KernelLaunch launch;
};
struct ReplayFile {
    std::unique_ptr<MeasuredReplay> model;
    std::string error;
};
ReplayFile load_measured_replay(std::istream &input, std::span<const ReplayBinding> bindings);
} // namespace fake_cuda
#endif
