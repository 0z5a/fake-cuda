#include <cuda.h>
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #condition, __FILE__, __LINE__); std::abort(); \
} } while (0)
#define OK(call) do { CUresult result = (call); if (result != CUDA_SUCCESS) { \
    std::fprintf(stderr, "FAIL: %s returned %d (%s:%d)\n", #call, int(result), __FILE__, __LINE__); std::abort(); \
} } while (0)

template <class T> T load(void *lib, const char *name) {
    auto function = reinterpret_cast<T>(dlsym(lib, name));
    if (!function) { std::fprintf(stderr, "missing %s\n", name); std::abort(); }
    return function;
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    CHECK(lib);
    auto init = load<CUresult (CUDAAPI *)(unsigned int)>(lib, "cuInit");
    auto get = load<CUresult (CUDAAPI *)(CUdevice *, int)>(lib, "cuDeviceGet");
    auto retain = load<CUresult (CUDAAPI *)(CUcontext *, CUdevice)>(lib, "cuDevicePrimaryCtxRetain");
    auto release = load<CUresult (CUDAAPI *)(CUdevice)>(lib, "cuDevicePrimaryCtxRelease_v2");
    auto set = load<CUresult (CUDAAPI *)(CUcontext)>(lib, "cuCtxSetCurrent");
    auto current = load<CUresult (CUDAAPI *)(CUcontext *)>(lib, "cuCtxGetCurrent");
    auto create_ctx = load<CUresult (CUDAAPI *)(CUcontext *, unsigned int, CUdevice)>(lib, "cuCtxCreate_v2");
    auto destroy_ctx = load<CUresult (CUDAAPI *)(CUcontext)>(lib, "cuCtxDestroy_v2");
    auto create_stream = load<CUresult (CUDAAPI *)(CUstream *, unsigned int)>(lib, "cuStreamCreate");
    auto destroy_stream = load<CUresult (CUDAAPI *)(CUstream)>(lib, "cuStreamDestroy_v2");
    auto sync_stream = load<CUresult (CUDAAPI *)(CUstream)>(lib, "cuStreamSynchronize");
    auto alloc = load<CUresult (CUDAAPI *)(CUdeviceptr *, size_t)>(lib, "cuMemAlloc_v2");
    auto free_device = load<CUresult (CUDAAPI *)(CUdeviceptr)>(lib, "cuMemFree_v2");
    auto h2d = load<CUresult (CUDAAPI *)(CUdeviceptr, const void *, size_t, CUstream)>(lib, "cuMemcpyHtoDAsync_v2");
    auto event_create = load<CUresult (CUDAAPI *)(CUevent *, unsigned int)>(lib, "cuEventCreate");
    auto event_record = load<CUresult (CUDAAPI *)(CUevent, CUstream)>(lib, "cuEventRecord");
    auto event_sync = load<CUresult (CUDAAPI *)(CUevent)>(lib, "cuEventSynchronize");
    auto event_destroy = load<CUresult (CUDAAPI *)(CUevent)>(lib, "cuEventDestroy_v2");

    OK(init(0));
    CUdevice device;
    OK(get(&device, 0));
    CUcontext primary;
    OK(retain(&primary, device));
    constexpr int threads = 8, operations = 500;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    for (int i = 0; i < threads; ++i) {
        workers.emplace_back([&] {
            OK(set(primary));
            CUcontext observed = nullptr;
            OK(current(&observed)); CHECK(observed == primary);
            CUstream stream;
            OK(create_stream(&stream, CU_STREAM_NON_BLOCKING));
            CUdeviceptr pointer;
            OK(alloc(&pointer, 64));
            CUevent event;
            OK(event_create(&event, CU_EVENT_DISABLE_TIMING));
            char bytes[64] = {};
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int j = 0; j < operations; ++j) {
                OK(current(&observed)); CHECK(observed == primary);
                OK(h2d(pointer, bytes, sizeof(bytes), stream));
                OK(event_record(event, stream));
            }
            OK(sync_stream(stream)); OK(event_sync(event));
            OK(event_destroy(event)); OK(free_device(pointer)); OK(destroy_stream(stream));
            OK(set(nullptr));
        });
    }
    while (ready.load(std::memory_order_acquire) != threads) std::this_thread::yield();
    auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto &worker : workers) worker.join();
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    CHECK(elapsed < 10.0);
    OK(release(device));

    // Another thread's TLS reference must not keep a destroyed context live.
    CUcontext owned;
    OK(create_ctx(&owned, 0, device));
    // Keep teardown draining long enough for a peer to observe retirement.
    CUstream pending_stream;
    OK(create_stream(&pending_stream, CU_STREAM_NON_BLOCKING));
    CUdeviceptr pending;
    constexpr size_t large_copy = 2'000'000;
    OK(alloc(&pending, large_copy));
    std::vector<char> source(large_copy);
    CHECK(setenv("FAKE_CUDA_H2D_BW_GBPS", "0.01", 1) == 0);
    OK(h2d(pending, source.data(), source.size(), pending_stream));
    CHECK(unsetenv("FAKE_CUDA_H2D_BW_GBPS") == 0);
    std::atomic<bool> attached{false}, saw_retirement{false};
    std::thread observer([&] {
        OK(set(owned));
        attached.store(true, std::memory_order_release);
        CUcontext observed = owned;
        while (observed == owned) {
            OK(current(&observed));
            if (observed == owned) std::this_thread::yield();
        }
        CHECK(observed == nullptr);
        CUstream stream = nullptr;
        CHECK(create_stream(&stream, 0) == CUDA_ERROR_INVALID_CONTEXT);
        CHECK(h2d(pending, source.data(), source.size(), pending_stream) == CUDA_ERROR_INVALID_CONTEXT);
        CHECK(set(owned) == CUDA_ERROR_INVALID_CONTEXT);
        saw_retirement.store(true, std::memory_order_release);
    });
    while (!attached.load(std::memory_order_acquire)) std::this_thread::yield();
    OK(destroy_ctx(owned));
    observer.join();
    CHECK(saw_retirement.load(std::memory_order_acquire));
    CHECK(dlclose(lib) == 0);
    std::printf("PASS: %d threads, %d submissions per thread, %.3f s; retired TLS context rejected\n",
                threads, operations, elapsed);
}
