#include "impl/kernel_launch.h"
#include "profile/device_profile.h"

#include <cstring>
#include <limits>

namespace fake_cuda {
CUresult KernelLaunch::validate(const DeviceProfile &profile) const {
    auto exceeds = [&](int attribute, std::uint64_t value) {
        const auto found = profile.attributes.find(attribute);
        return found != profile.attributes.end() &&
               (found->second < 0 || value > static_cast<std::uint64_t>(found->second));
    };
    std::uint64_t blocks = 1, threads = 1;
    for (size_t i = 0; i < 3; ++i) {
        if (!grid[i] || !block[i] || blocks > UINT64_MAX / grid[i] || threads > UINT64_MAX / block[i] ||
            exceeds(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + static_cast<int>(i), grid[i]) ||
            exceeds(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X + static_cast<int>(i), block[i]))
            return CUDA_ERROR_INVALID_VALUE;
        blocks *= grid[i];
        threads *= block[i];
    }
    // Device ceilings only: static resources and per-function opt-in are unknown.
    const int shared_limit = profile.attributes.contains(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        ? CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN : CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
    return exceeds(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, threads) || exceeds(shared_limit, dynamic_shared_bytes)
        ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
CUresult KernelLaunch::snapshot(void **extra) {
    const void *buffer = nullptr;
    size_t bytes = 0;
    bool have_buffer = false, have_size = false;
    for (size_t i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
        if (i >= 4) return CUDA_ERROR_INVALID_VALUE;
        if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER && !have_buffer) {
            buffer = extra[i + 1];
            have_buffer = true;
        } else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE && !have_size && extra[i + 1]) {
            bytes = *static_cast<const size_t *>(extra[i + 1]);
            have_size = true;
        } else return CUDA_ERROR_INVALID_VALUE;
    }
    if ((bytes && !buffer) || bytes > packed_parameters.max_size()) return CUDA_ERROR_INVALID_VALUE;
    parameters = ParameterEncoding::packed_buffer;
    packed_parameters.resize(bytes);
    if (bytes) std::memcpy(packed_parameters.data(), buffer, bytes);
    return CUDA_SUCCESS;
}
} // namespace fake_cuda
