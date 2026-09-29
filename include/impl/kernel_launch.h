#ifndef FAKE_CUDA_IMPL_KERNEL_LAUNCH_H
#define FAKE_CUDA_IMPL_KERNEL_LAUNCH_H

#include <cuda.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fake_cuda {
struct DeviceProfile;
// A load identity is local to this process, not a code hash or calibrated kernel ID.
struct KernelIdentity {
    std::uintptr_t load_id;
    std::string symbol;
};
enum class ParameterEncoding { unknown_layout, packed_buffer };
struct KernelLaunch {
    std::shared_ptr<const KernelIdentity> kernel;
    std::array<unsigned int, 3> grid, block;
    unsigned int dynamic_shared_bytes;
    ParameterEncoding parameters = ParameterEncoding::unknown_layout;
    std::vector<std::byte> packed_parameters;
    CUresult validate(const DeviceProfile &profile) const;
    CUresult snapshot(void **extra);
    // Opaque images have no decoded parameter ABI or static resource metadata.
};
struct ModuleRecord {
    CUcontext context;
    std::uintptr_t load_id;
};
struct FunctionRecord {
    CUmodule module;
    std::shared_ptr<const KernelIdentity> kernel;
};
struct KernelRecord {
    CUlibrary library;
    std::shared_ptr<const KernelIdentity> identity;
};
} // namespace fake_cuda

#endif
