#include "impl/core.h"
#include "impl/virtual_work.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); std::exit(1); \
} } while (0)
#define OK(expr) CHECK((expr) == CUDA_SUCCESS)

int main() {
    OK(core_init(0));
    int count = 0; OK(core_count(&count)); CHECK(count == 3);
    std::array<CUcontext, 3> contexts;
    std::array<CUdeviceptr, 3> allocations;
    for (int device = 0; device < count; ++device) {
        const size_t capacity = device == 1 ? 524288 : 1048576;
        char name[128]; OK(core_name(name, sizeof(name), device));
        CHECK(std::strcmp(name, device == 1 ? "Synthetic smaller device" : "Synthetic contract device") == 0);
        size_t total = 0, free = 0;
        OK(core_memory(&total, device)); CHECK(total == capacity);
        OK(core_context_create(&contexts[device], 0, device));
        OK(virtual_mem_alloc(&allocations[device], 1024));
        OK(virtual_mem_get_info(&free, &total)); CHECK(total == capacity && free == capacity - 1024);
        CUdeviceptr rejected;
        CHECK(virtual_mem_alloc(&rejected, capacity) == CUDA_ERROR_OUT_OF_MEMORY);
        CUmodule module; CUfunction function;
        OK(virtual_module_load_data(&module, "opaque"));
        OK(virtual_module_get_function(&function, module, "contract"));
        CHECK(virtual_launch_kernel(function, 1, 1, 1, 128, 1, 1, 0, nullptr, nullptr, nullptr) ==
              (device == 1 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS));
        OK(virtual_module_unload(module));
    }
    for (int source = 0; source < count; ++source) {
        OK(core_context_set(contexts[source]));
        for (int peer = 0; peer < count; ++peer) {
            const bool allowed = peer == (source + 1) % count;
            int access = -1; OK(core_can_access_peer(&access, source, peer)); CHECK(access == allowed);
            if (source == peer) continue;
            if (allowed) {
                OK(core_context_enable_peer(contexts[peer], 0));
                CHECK(core_context_enable_peer(contexts[peer], 0) == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
                OK(core_context_disable_peer(contexts[peer]));
            } else CHECK(core_context_enable_peer(contexts[peer], 0) == CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
            CHECK(core_context_disable_peer(contexts[peer]) == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
        }
    }
    // A second context shares its device's accounting, not its peer's capacity.
    CUcontext second;
    OK(core_context_create(&second, 0, 1));
    size_t free = 0, total = 0; OK(virtual_mem_get_info(&free, &total));
    CHECK(total == 524288 && free == total - 1024);
    OK(core_context_destroy(second));
    for (auto context : contexts) OK(core_context_destroy(context));
    std::puts("PASS: per-device profiles, shared accounting, launch limits and directed peer access");
}
