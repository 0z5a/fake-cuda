// Query a real Driver without linking the simulator or changing its environment.
#include <cuda.h>
#include <dlfcn.h>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <string_view>

template<class T> T symbol(void *driver, const char *name) {
    auto function = reinterpret_cast<T>(dlsym(driver, name));
    if (!function) { std::fprintf(stderr, "missing symbol: %s\n", name); std::exit(1); }
    return function;
}
#define CALL(name, ...) do { CUresult status = symbol<decltype(&name)>(driver, #name)(__VA_ARGS__); \
    if (status != CUDA_SUCCESS) { std::fprintf(stderr, "%s: %d\n", #name, status); return 1; } \
} while (0)
int main(int argc, char **argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s /path/to/real/libcuda.so.1 ordinal\n", argv[0]); return 1; }
    const std::string_view argument(argv[2]);
    int ordinal = -1;
    auto [end, error] = std::from_chars(argument.data(), argument.data() + argument.size(), ordinal);
    if (error != std::errc{} || end != argument.data() + argument.size() || ordinal < 0) return 1;
    void *driver = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!driver) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    CALL(cuInit, 0);
    int version; CUdevice device; CUuuid uuid; char name[256]; size_t memory;
    CALL(cuDriverGetVersion, &version);
    CALL(cuDeviceGet, &device, ordinal);
    CALL(cuDeviceGetUuid_v2, &uuid, device);
    CALL(cuDeviceGetName, name, sizeof(name), device);
    CALL(cuDeviceTotalMem_v2, &memory, device);
    std::printf("schema=1\nname=%s\nsource=Driver %d device ", name, version);
    for (unsigned char byte : uuid.bytes) std::printf("%02x", byte);
    std::printf("\nmemory_bytes=%zu\n", memory);
    auto attribute = symbol<decltype(&cuDeviceGetAttribute)>(driver, "cuDeviceGetAttribute");
    for (int id = 1; id < CU_DEVICE_ATTRIBUTE_MAX; ++id) {
        int value;
        if (attribute(&value, static_cast<CUdevice_attribute>(id), device) == CUDA_SUCCESS)
            std::printf("attribute.%d=%d\n", id, value);
    }
    dlclose(driver);
}
