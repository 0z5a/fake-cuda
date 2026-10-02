#include "impl/core.h"
#include "impl/scheduler.h"
#include "impl/virtual_work.h"
#include "profile/gh200.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); std::exit(1); \
} } while (0)
#define OK(expr) CHECK((expr) == CUDA_SUCCESS)

int main(int argc, char **argv) {
    CHECK(argc == 2 || argc == 3);
    const int expected = std::atoi(argv[1]);
    int count = -1;
    if (!expected) {
        CHECK(core_init(0) == CUDA_ERROR_INVALID_VALUE);
        CHECK(core_count(&count) == CUDA_ERROR_INVALID_VALUE);
        CHECK(count == 0);
        std::puts("invalid configuration rejected");
        return 0;
    }
    const bool synthetic = argc == 3;
    OK(core_init(0)); OK(core_count(&count)); CHECK(count == expected);
    CUdevice invalid;
    CHECK(core_device(&invalid, count) == CUDA_ERROR_INVALID_DEVICE);
    CHECK(core_device(&invalid, -1) == CUDA_ERROR_INVALID_DEVICE);
    const size_t capacity = synthetic ? 1048576 : fake_cuda::gh200::memory_bytes;
    std::set<std::string> uuids;
    CUcontext first = nullptr, last = nullptr;
    CUdeviceptr first_ptr = 0, last_ptr = 0;
    for (int i = 0; i < count; ++i) {
        CUuuid uuid; OK(core_uuid(&uuid, i));
        CHECK(uuids.emplace(uuid.bytes, sizeof(uuid.bytes)).second);
        size_t bytes = 0; OK(core_memory(&bytes, i)); CHECK(bytes == capacity);
        char name[128]; OK(core_name(name, sizeof(name), i));
        CHECK(std::strcmp(name, synthetic ? "Synthetic contract device" : fake_cuda::gh200::name) == 0);
        int value = 0; OK(core_attribute(&value, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, i));
        CHECK(value == (synthetic ? 128 : 1024));
        if (!synthetic && i == 0)
            for (size_t attribute = 1; attribute < std::size(fake_cuda::gh200::attributes); ++attribute) {
                OK(core_attribute(&value, static_cast<CUdevice_attribute>(attribute), i));
                CHECK(value == fake_cuda::gh200::attributes[attribute]);
            }
    }
    OK(core_context_create(&first, 0, 0));
    OK(virtual_mem_alloc(&first_ptr, 4096));
    size_t free = 0, total = 0;
    OK(virtual_mem_get_info(&free, &total)); CHECK(total == capacity && free == capacity - 4096);
    CUdeviceptr too_large = 0;
    CHECK(virtual_mem_alloc(&too_large, capacity) == CUDA_ERROR_OUT_OF_MEMORY);
    if (synthetic) {
        int unknown = 123;
        CHECK(core_attribute(&unknown, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 0) == CUDA_ERROR_NOT_SUPPORTED);
        CHECK(unknown == 123);
    }
    if (count > 1) {
        OK(core_context_create(&last, 0, count - 1));
        OK(virtual_mem_get_info(&free, &total)); CHECK(free == capacity);
        OK(virtual_mem_alloc(&last_ptr, 2048));
        OK(virtual_mem_get_info(&free, &total)); CHECK(free == capacity - 2048);
        // Exercise sparse outgoing queues at high ordinals in both directions.
        auto &scheduler = fake_cuda::detail::scheduler();
        std::scoped_lock lock(scheduler.mutex);
        const auto key = fake_cuda::detail::key_for(last, nullptr);
        auto forward = scheduler.queue.schedule_peer(key, 0, count - 1, 256);
        auto reverse = scheduler.queue.schedule_peer(key, count - 1, 0, 256);
        CHECK(fake_cuda::virtual_core_device(0)->p2p_queues.size() == 1);
        CHECK(fake_cuda::virtual_core_device(count - 1)->p2p_queues.size() == 1);
        CHECK(forward != reverse);
    }
    if (last) OK(core_context_destroy(last));
    OK(core_context_destroy(first));
    std::printf("device profile contract passed: %d devices\n", count);
}
