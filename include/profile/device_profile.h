#ifndef FAKE_CUDA_PROFILE_DEVICE_PROFILE_H
#define FAKE_CUDA_PROFILE_DEVICE_PROFILE_H

#include <cuda.h>
#include <cstddef>
#include <map>
#include <memory>
#include <vector>
#include <string>

namespace fake_cuda {
struct DeviceProfile {
    std::string name;
    std::string source;
    size_t memory_bytes = 0;
    // Values use CUdevice_attribute units; missing attributes remain unknown.
    std::map<int, int> attributes;
};
struct DeviceConfiguration {
    int count = 1;
    std::vector<std::shared_ptr<const DeviceProfile>> profiles;
    std::vector<bool> peer_access;
    std::string topology_source = "synthetic all-to-all";
    bool can_access(CUdevice source, CUdevice peer) const {
        return peer_access[static_cast<size_t>(source) * count + peer];
    }
    CUresult status = CUDA_SUCCESS;
};
const DeviceConfiguration &device_configuration();
} // namespace fake_cuda

#endif
