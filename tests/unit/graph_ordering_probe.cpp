// Timing-model regression: virtual addresses are never dereferenced.
#include <cuda.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

namespace {
void *library;
template <typename T> T symbol(const char *name) {
    void *address = dlsym(library, name);
    if (!address) { std::fprintf(stderr, "missing %s\n", name); std::exit(1); }
    return reinterpret_cast<T>(address);
}
#define API(name, ...) symbol<decltype(&::name)>(#name)(__VA_ARGS__)
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); std::exit(1); } } while (false)
#define OK(expr) CHECK((expr) == CUDA_SUCCESS)

float elapsed(CUevent start, CUevent end) {
    float ms = 0;
    OK(API(cuEventSynchronize, end));
    OK(API(cuEventElapsedTime_v2, &ms, start, end));
    return ms;
}
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    setenv("FAKE_CUDA_H2D_BW_GBPS", "0.1", 1);
    setenv("FAKE_CUDA_D2H_BW_GBPS", "0.1", 1);
    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    CHECK(library);
    OK(API(cuInit, 0));
    CUcontext context;
    OK(API(cuDevicePrimaryCtxRetain, &context, 0));
    OK(API(cuCtxSetCurrent, context));
    CUstream a, b;
    OK(API(cuStreamCreate, &a, CU_STREAM_NON_BLOCKING));
    OK(API(cuStreamCreate, &b, CU_STREAM_NON_BLOCKING));
    CUevent first, second;
    OK(API(cuEventCreate, &first, 0));
    OK(API(cuEventCreate, &second, 0));
    constexpr size_t bytes = 8'000'000; // Exactly 80 ms per modeled copy.
    CUdeviceptr device;
    OK(API(cuMemAlloc_v2, &device, bytes));
    char host = 0; // Timing-only Driver does not read host memory.
    CUgraph graph;
    CUgraphExec one, two;
    OK(API(cuStreamBeginCapture_v2, a, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(API(cuMemcpyHtoDAsync_v2, device, &host, bytes, a));
    OK(API(cuMemcpyDtoHAsync_v2, &host, device, bytes, a));
    OK(API(cuStreamEndCapture, a, &graph));
    OK(API(cuStreamQuery, a)); // Capture must not enqueue device work.
    OK(API(cuGraphInstantiateWithFlags, &one, graph, 0));
    OK(API(cuGraphInstantiateWithFlags, &two, graph, 0));
    OK(API(cuGraphDestroy, graph));

    const bool independent = argv[2][0] == 'i';
    OK(API(cuGraphLaunch, one, a));
    OK(API(cuEventRecord, first, a));
    OK(API(cuGraphLaunch, independent ? two : one, b));
    OK(API(cuEventRecord, second, b));
    // Destroying exec handles must preserve submitted work and stream tails.
    OK(API(cuGraphExecDestroy, one));
    OK(API(cuGraphExecDestroy, two));
    const float delta = elapsed(first, second);
    std::printf("%s completion separation: %.3f ms\n", argv[2], delta);
    if (independent) CHECK(delta >= 79 && delta < 140);
    else CHECK(delta >= 159 && delta < 240);

    // A wait keeps its original record even when the same event is re-recorded.
    OK(API(cuEventRecord, first, a));
    OK(API(cuStreamWaitEvent, b, first, 0));
    OK(API(cuEventRecord, second, b));
    OK(API(cuMemcpyHtoDAsync_v2, device, &host, bytes, a));
    OK(API(cuEventRecord, first, a));
    CHECK(elapsed(second, first) >= 79);
    OK(API(cuMemFree_v2, device));
    OK(API(cuEventDestroy_v2, first));
    OK(API(cuEventDestroy_v2, second));
    OK(API(cuStreamDestroy_v2, a));
    OK(API(cuStreamDestroy_v2, b));
    OK(API(cuDevicePrimaryCtxRelease_v2, 0));
    std::puts("PASS: capture isolation, exec ordering/lifetime, event record snapshot");
}
