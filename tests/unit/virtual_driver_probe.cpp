/* Standalone CUDA 13 Driver ABI probe; no GPU or CUDA library at link time.
 * Build: c++ -std=c++23 -Wall -Wextra -Werror -I.venv-cu130/lib/python3.12/site-packages/nvidia/cu13/include tests/unit/virtual_driver_probe.cpp -ldl -o /tmp/virtual_driver_probe
 * Run:   /tmp/virtual_driver_probe build/libcuda.so.1
 * Bandwidth overrides (decimal GB/s): FAKE_CUDA_H2D_BW_GBPS,
 * FAKE_CUDA_D2H_BW_GBPS, FAKE_CUDA_HBM_BW_GBPS.
 */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if CUDA_VERSION < 13000 || CUDA_VERSION >= 14000
#error "Build this probe with the CUDA 13 cuda.h"
#endif

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL: %s (%s:%d)\n", #expr, __FILE__, __LINE__); exit(1); \
} } while (0)
#define OK(expr) do { CUresult r_ = (expr); if (r_ != CUDA_SUCCESS) { \
    fprintf(stderr, "FAIL: %s returned %d (%s:%d)\n", #expr, (int)r_, __FILE__, __LINE__); exit(1); \
} } while (0)
#define LOAD(fn, symbol) do { \
    dlerror(); (fn) = reinterpret_cast<decltype(fn)>(dlsym(lib, symbol)); \
    if (!(fn)) { fprintf(stderr, "missing Driver ABI symbol %s: %s\n", symbol, dlerror()); exit(1); } \
} while (0)

static double seconds(void) {
    struct timespec ts;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* A 0.12 s transfer is long enough to distinguish queueing from immediate completion.
 * Bounds also prevent deliberately slow configurations from making the test unbounded. */
static size_t transfer_size(const char *env, double *duration) {
    const char *text = getenv(env);
    if (!text) { CHECK(setenv(env, "0.01", 1) == 0); text = getenv(env); }
    char *end;
    errno = 0;
    double bw = strtod(text, &end);
    if (errno || end == text || *end || !isfinite(bw) || bw < 0.0001 || bw > 0.04) {
        fprintf(stderr, "%s must be in [0.0001, 0.04] decimal GB/s (got %s)\n", env, text);
        exit(2);
    }
    size_t bytes = (size_t)(bw * 1e9 * 0.12);
    if (bytes > 4u * 1024u * 1024u) bytes = 4u * 1024u * 1024u;
    *duration = (double)bytes / (bw * 1e9);
    return bytes;
}

/* Deliberately broad timing margins: scheduling noise is not an ABI failure. */
static void slept_for(double elapsed, double expected, const char *what) {
    if (elapsed < expected * 0.55 || elapsed > expected + 0.8) {
        fprintf(stderr, "FAIL: %s: %.3f s, expected about %.3f s\n", what, elapsed, expected);
        exit(1);
    }
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    double h2d_time, d2h_time, d2d_time;
    size_t h2d_bytes = transfer_size("FAKE_CUDA_H2D_BW_GBPS", &h2d_time);
    size_t d2h_bytes = transfer_size("FAKE_CUDA_D2H_BW_GBPS", &d2h_time);
    size_t d2d_bytes = transfer_size("FAKE_CUDA_HBM_BW_GBPS", &d2d_time);
    double start = seconds();
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen %s: %s\n", argv[1], dlerror()); return 1; }

    CUresult (CUDAAPI *init)(unsigned int), (CUDAAPI *get)(CUdevice *, int);
    CUresult (CUDAAPI *retain)(CUcontext *, CUdevice), (CUDAAPI *set)(CUcontext);
    CUresult (CUDAAPI *release)(CUdevice);
    CUresult (CUDAAPI *create_stream)(CUstream *, unsigned int);
    CUresult (CUDAAPI *destroy_stream)(CUstream), (CUDAAPI *sync_stream)(CUstream);
    CUresult (CUDAAPI *query_stream)(CUstream);
    CUresult (CUDAAPI *alloc)(CUdeviceptr *, size_t), (CUDAAPI *free_device)(CUdeviceptr);
    CUresult (CUDAAPI *alloc_async)(CUdeviceptr *, size_t, CUstream);
    CUresult (CUDAAPI *free_async)(CUdeviceptr, CUstream);
    CUresult (CUDAAPI *pointer_attribute)(void *, CUpointer_attribute, CUdeviceptr);
    CUresult (CUDAAPI *h2d)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (CUDAAPI *d2h)(void *, CUdeviceptr, size_t, CUstream);
    CUresult (CUDAAPI *d2d)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult (CUDAAPI *create_event)(CUevent *, unsigned int);
    CUresult (CUDAAPI *record_event)(CUevent, CUstream);
    CUresult (CUDAAPI *query_event)(CUevent), (CUDAAPI *sync_event)(CUevent);
    CUresult (CUDAAPI *destroy_event)(CUevent);
    CUresult (CUDAAPI *wait_event)(CUstream, CUevent, unsigned int);
    CUresult (CUDAAPI *begin_capture)(CUstream, CUstreamCaptureMode);
    CUresult (CUDAAPI *end_capture)(CUstream, CUgraph *);
    CUresult (CUDAAPI *instantiate)(CUgraphExec *, CUgraph, CUDA_GRAPH_INSTANTIATE_PARAMS *);
    CUresult (CUDAAPI *launch_graph)(CUgraphExec, CUstream);
    CUresult (CUDAAPI *destroy_graph)(CUgraph), (CUDAAPI *destroy_exec)(CUgraphExec);
    CUresult (CUDAAPI *load_module)(CUmodule *, const void *);
    CUresult (CUDAAPI *get_function)(CUfunction *, CUmodule, const char *);
    CUresult (CUDAAPI *unload_module)(CUmodule);
    CUresult (CUDAAPI *launch_kernel)(CUfunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **);

    LOAD(init, "cuInit"); LOAD(get, "cuDeviceGet");
    LOAD(retain, "cuDevicePrimaryCtxRetain"); LOAD(set, "cuCtxSetCurrent");
    LOAD(release, "cuDevicePrimaryCtxRelease_v2");
    LOAD(create_stream, "cuStreamCreate"); LOAD(destroy_stream, "cuStreamDestroy_v2");
    LOAD(sync_stream, "cuStreamSynchronize"); LOAD(query_stream, "cuStreamQuery");
    LOAD(alloc, "cuMemAlloc_v2"); LOAD(free_device, "cuMemFree_v2");
    LOAD(alloc_async, "cuMemAllocAsync"); LOAD(free_async, "cuMemFreeAsync");
    LOAD(pointer_attribute, "cuPointerGetAttribute");
    LOAD(h2d, "cuMemcpyHtoDAsync_v2"); LOAD(d2h, "cuMemcpyDtoHAsync_v2");
    LOAD(d2d, "cuMemcpyDtoDAsync_v2");
    LOAD(create_event, "cuEventCreate"); LOAD(record_event, "cuEventRecord");
    LOAD(query_event, "cuEventQuery"); LOAD(sync_event, "cuEventSynchronize");
    LOAD(destroy_event, "cuEventDestroy_v2"); LOAD(wait_event, "cuStreamWaitEvent");
    LOAD(begin_capture, "cuStreamBeginCapture_v2"); LOAD(end_capture, "cuStreamEndCapture");
    LOAD(instantiate, "cuGraphInstantiateWithParams"); LOAD(launch_graph, "cuGraphLaunch");
    LOAD(destroy_graph, "cuGraphDestroy"); LOAD(destroy_exec, "cuGraphExecDestroy");
    LOAD(load_module, "cuModuleLoadData"); LOAD(get_function, "cuModuleGetFunction");
    LOAD(launch_kernel, "cuLaunchKernel"); LOAD(unload_module, "cuModuleUnload");

    OK(init(0));
    CUdevice device;
    CUcontext context;
    OK(get(&device, 0)); OK(retain(&context, device)); CHECK(context != NULL);
    OK(set(context));
    CUstream a = NULL, b = NULL;
    CUevent done = NULL, waited = NULL;
    OK(create_stream(&a, 0)); OK(create_stream(&b, 0));
    OK(create_event(&done, CU_EVENT_DEFAULT)); OK(create_event(&waited, CU_EVENT_DEFAULT));
    CUdeviceptr dst = 0, src = 0, temporary = 0;
    OK(alloc(&dst, 4u * 1024u * 1024u)); OK(alloc(&src, 4u * 1024u * 1024u));
    CHECK(dst != 0 && src != 0 && dst != src); /* Opaque virtual addresses, never dereferenced. */
    CUmemorytype memory_type = CU_MEMORYTYPE_HOST;
    CUdeviceptr range_start = 0, device_pointer = 0;
    size_t range_size = 0;
    CUcontext pointer_context = NULL;
    OK(pointer_attribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dst + 1));
    CHECK(memory_type == CU_MEMORYTYPE_DEVICE);
    OK(pointer_attribute(&device_pointer, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, dst + 1));
    CHECK(device_pointer == dst + 1);
    OK(pointer_attribute(&range_start, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, dst + 1));
    OK(pointer_attribute(&range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, dst + 1));
    OK(pointer_attribute(&pointer_context, CU_POINTER_ATTRIBUTE_CONTEXT, dst + 1));
    CHECK(range_start == dst && range_size == 4u * 1024u * 1024u && pointer_context == context);
    CHECK(pointer_attribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, 0) == CUDA_ERROR_INVALID_VALUE);
    auto *host = static_cast<unsigned char *>(calloc(4u * 1024u * 1024u, 1));
    CHECK(host != NULL);

    /* Two long operations on A must queue FIFO. B's tiny transfer must progress
       independently while A is busy; event synchronization must actually wait. */
    double t = seconds();
    OK(h2d(dst, host, h2d_bytes, a));
    OK(h2d(src, host, h2d_bytes, a));
    OK(record_event(done, a));
    CHECK(query_event(done) == CUDA_ERROR_NOT_READY);
    CHECK(query_stream(a) == CUDA_ERROR_NOT_READY);
    OK(d2d(dst, src, 1, b));
    OK(sync_stream(b));
    CHECK(seconds() - t < h2d_time * 1.5 + 0.05);
    CHECK(query_event(done) == CUDA_ERROR_NOT_READY);
    OK(sync_event(done));
    slept_for(seconds() - t, 2.0 * h2d_time, "same-stream FIFO / event sync");
    OK(sync_stream(a));
    OK(query_event(done));

    /* B cannot pass an event recorded after a long transfer on A. */
    t = seconds();
    OK(d2h(host, dst, d2h_bytes, a));
    OK(record_event(done, a));
    OK(wait_event(b, done, 0));
    OK(d2d(dst, src, 1, b));
    OK(record_event(waited, b));
    CHECK(query_event(waited) == CUDA_ERROR_NOT_READY);
    OK(sync_stream(b));
    slept_for(seconds() - t, d2h_time, "cross-stream event wait / stream sync");
    OK(query_event(waited));
    OK(sync_stream(a));

    t = seconds();
    OK(d2d(dst, src, d2d_bytes, a));
    OK(sync_stream(a));
    slept_for(seconds() - t, d2d_time, "DtoD stream synchronization");

    OK(alloc_async(&temporary, 1024, a)); CHECK(temporary != 0);
    OK(free_async(temporary, a)); OK(sync_stream(a));

    static const char ptx[] =
        ".version 8.0\n.target sm_90\n.address_size 64\n"
        ".visible .entry virtual_probe_kernel() { ret; }\n";
    CUmodule module = NULL;
    CUfunction kernel = NULL;
    OK(load_module(&module, ptx)); CHECK(module != NULL);
    OK(get_function(&kernel, module, "virtual_probe_kernel")); CHECK(kernel != NULL);
    OK(launch_kernel(kernel, 1, 1, 1, 1, 1, 1, 0, a, NULL, NULL));
    OK(sync_stream(a));

    CUgraph graph = NULL;
    CUgraphExec exec = NULL;
    CUDA_GRAPH_INSTANTIATE_PARAMS params = {};
    OK(begin_capture(a, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(d2d(dst, src, 1, a));
    OK(launch_kernel(kernel, 1, 1, 1, 1, 1, 1, 0, a, NULL, NULL));
    OK(end_capture(a, &graph)); CHECK(graph != NULL);
    OK(instantiate(&exec, graph, &params)); CHECK(exec != NULL);
    OK(launch_graph(exec, a)); OK(sync_stream(a));
    OK(destroy_exec(exec)); OK(destroy_graph(graph)); OK(unload_module(module));
    OK(destroy_event(waited)); OK(destroy_event(done));
    OK(free_device(src)); OK(free_device(dst));
    OK(destroy_stream(b)); OK(destroy_stream(a));
    OK(set(NULL)); OK(release(device));
    free(host);
    CHECK(dlclose(lib) == 0);
    CHECK(seconds() - start < 2.0);
    puts("PASS: virtual CUDA Driver ABI, FIFO, concurrency, waits, async memory, kernel and graph");
    return 0;
}
