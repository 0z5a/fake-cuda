#ifndef FAKE_CUDA_PROFILE_GH200_H
#define FAKE_CUDA_PROFILE_GH200_H

#include <cstddef>
#include <cstdint>

namespace fake_cuda::gh200 {
// Snapshot of device 0 on this host via cuDeviceGetAttribute(1..148),
// cuDeviceGetName and cuDeviceTotalMem_v2 (CUDA Driver 13.2). These describe
// device capabilities; the work simulator has separate scheduling parameters.
inline constexpr const char *name = "NVIDIA GH200 144G HBM3e";
inline constexpr std::size_t memory_bytes = 153008209920ULL;
// NVIDIA's published NVLink-C2C link peak (decimal bytes/s), not measured
// cudaMemcpy throughput and not the simulator's configurable copy rate.
inline constexpr std::uint64_t c2c_per_direction_peak_bytes_per_second = 450'000'000'000ULL;
inline constexpr int attributes[] = {
    0,
    /*   1- 10 */ 1024, 1024, 1024, 64, 2147483647, 65535, 65535, 49152, 65536, 32,
    /*  11- 20 */ 2147483647, 65536, 1980000, 512, 1, 132, 0, 0, 1, 0,
    /*  21- 30 */ 131072, 131072, 65536, 16384, 16384, 16384, 32768, 32768, 2048, 512,
    /*  31- 40 */ 1, 1, 1, 0, 0, 3201000, 6144, 62914560, 2048, 2,
    /*  41- 50 */ 1, 32768, 2048, 1, 32768, 32768, 8192, 8192, 32768, 9,
    /*  51- 60 */ 32, 32768, 32768, 2046, 32768, 131072, 65536, 16384, 16384, 16384,
    /*  61- 70 */ 32768, 2048, 32768, 32768, 2048, 32768, 32768, 2046, 268435456, 131072,
    /*  71- 80 */ 65000, 2097120, 32768, 32768, 9, 0, 32768, 1, 1, 1,
    /*  81- 90 */ 233472, 65536, 1, 0, 0, 1, 2, 1, 1, 1,
    /*  91-100 */ 1, 0, 0, 0, 1, 1, 232448, 1, 1, 1,
    /* 101-110 */ 1, 1, 1, 0, 0, 32, 1, 39321600, 134217728, 1,
    /* 111-120 */ 1024, 1, 0, 1, 1, 1, 3, 200, 9, 1,
    /* 121-130 */ 1, 1, 1, 1, 1, 4, 1, 1, 1, 1,
    /* 131-140 */ 2, 0, 0, 0, 0, 0, 0, 0, 591925470, 416420062,
    /* 141-148 */ 1, 1, 0, 1, 1, 0, 0, 1,
};
static_assert(sizeof(attributes) / sizeof(attributes[0]) == 149);
} // namespace fake_cuda::gh200

#endif
