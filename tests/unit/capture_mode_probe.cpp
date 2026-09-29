// The same ABI/return-code checks run against real CUDA and the fake Driver.
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {
void *library;
template <typename T> T symbol(const char *name) {
    void *p = dlsym(library, name);
    if (!p) { std::fprintf(stderr, "missing %s\n", name); std::exit(1); }
    return reinterpret_cast<T>(p);
}
#define API(name, ...) symbol<decltype(&::name)>(#name)(__VA_ARGS__)
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (false)
#define EXPECT(x, expected) do { CUresult code = (x); if (code != (expected)) { std::fprintf(stderr, "line %d: %s = %d expected %d\n", __LINE__, #x, code, expected); std::exit(1); } } while (false)
#define OK(x) EXPECT(x, CUDA_SUCCESS)
}
int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) return 2;
    const int ordinal = argc == 3 ? std::atoi(argv[2]) : 0;
    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    CHECK(library);
    OK(API(cuInit, 0));
    CUcontext ctx;
    OK(API(cuDevicePrimaryCtxRetain, &ctx, ordinal));
    OK(API(cuCtxSetCurrent, ctx));
    CUstream stream;
    OK(API(cuStreamCreate, &stream, CU_STREAM_NON_BLOCKING));
    CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
    OK(API(cuThreadExchangeStreamCaptureMode, &mode));
    CHECK(mode == CU_STREAM_CAPTURE_MODE_GLOBAL);
    std::thread other([] {
        CUstreamCaptureMode value = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
        OK(API(cuThreadExchangeStreamCaptureMode, &value));
        CHECK(value == CU_STREAM_CAPTURE_MODE_GLOBAL);
    });
    other.join();
    OK(API(cuThreadExchangeStreamCaptureMode, &mode));
    CHECK(mode == CU_STREAM_CAPTURE_MODE_RELAXED);
    std::puts("PASS: thread-local mode exchange");
    std::fflush(stdout);
    if (argc == 2) {
        EXPECT(API(cuThreadExchangeStreamCaptureMode, nullptr), CUDA_ERROR_INVALID_VALUE);
        mode = static_cast<CUstreamCaptureMode>(99);
        EXPECT(API(cuThreadExchangeStreamCaptureMode, &mode), CUDA_ERROR_INVALID_VALUE);
    }

    // PyTorch 2.13 uses a relaxed guard for RNG state allocation during capture.
    OK(API(cuStreamBeginCapture_v2, stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    mode = CU_STREAM_CAPTURE_MODE_RELAXED;
    OK(API(cuThreadExchangeStreamCaptureMode, &mode));
    CUdeviceptr pointer;
    OK(API(cuMemAlloc_v2, &pointer, 1024));
    OK(API(cuThreadExchangeStreamCaptureMode, &mode));
    CUgraph graph = nullptr;
    OK(API(cuStreamEndCapture, stream, &graph));
    OK(API(cuGraphDestroy, graph));
    OK(API(cuMemFree_v2, pointer));
    std::puts("PASS: relaxed allocation");
    std::fflush(stdout);

    OK(API(cuStreamBeginCapture_v2, stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    EXPECT(API(cuStreamSynchronize, stream), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    CUstreamCaptureStatus status;
    OK(API(cuStreamIsCapturing, stream, &status));
    CHECK(status == CU_STREAM_CAPTURE_STATUS_INVALIDATED);
    EXPECT(API(cuStreamEndCapture, stream, &graph), CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
    CHECK(graph == nullptr);
    OK(API(cuStreamIsCapturing, stream, &status));
    CHECK(status == CU_STREAM_CAPTURE_STATUS_NONE);
    std::puts("PASS: invalidation and recovery");
    std::fflush(stdout);

    OK(API(cuStreamBeginCapture_v2, stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    EXPECT(API(cuMemAlloc_v2, &pointer, 1024), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    OK(API(cuStreamIsCapturing, stream, &status));
    CHECK(status == CU_STREAM_CAPTURE_STATUS_INVALIDATED);
    EXPECT(API(cuStreamEndCapture, stream, &graph), CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
    CHECK(graph == nullptr);

    OK(API(cuStreamBeginCapture_v2, stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    std::thread wrong_thread([&] {
        OK(API(cuCtxSetCurrent, ctx));
        CUgraph rejected = nullptr;
        EXPECT(API(cuStreamEndCapture, stream, &rejected), CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD);
    });
    wrong_thread.join();
    OK(API(cuStreamIsCapturing, stream, &status));
    CHECK(status == CU_STREAM_CAPTURE_STATUS_NONE);
    EXPECT(API(cuStreamEndCapture, stream, &graph), CUDA_ERROR_ILLEGAL_STATE);
    OK(API(cuStreamBeginCapture_v2, stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(API(cuStreamEndCapture, stream, &graph));
    OK(API(cuGraphDestroy, graph));
    OK(API(cuStreamDestroy_v2, stream));
    OK(API(cuDevicePrimaryCtxRelease_v2, ordinal));
    std::puts("PASS: capture-mode exchange, thread isolation, relaxed allocation, invalidation/recovery, owner thread");
}
