/* Standalone CUDA 13 Driver ABI probe; no GPU or CUDA library at link time.
 * Build: c++ -std=c++23 -Wall -Wextra -Werror -I.venv-cu130/lib/python3.12/site-packages/nvidia/cu13/include tests/unit/virtual_multistream_probe.cpp -ldl -lm -o /tmp/virtual_multistream_probe
 * Run:   /tmp/virtual_multistream_probe build/libcuda.so.1
 */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

/* A 2 MB transfer takes ~0.2 s at 0.01 decimal GB/s. Allow ample timer noise. */
static void waited(double elapsed, const char *what) {
    if (!isfinite(elapsed) || elapsed < 0.08 || elapsed > 0.8) {
        fprintf(stderr, "FAIL: %s took %.3f s (expected an in-flight transfer)\n", what, elapsed);
        exit(1);
    }
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    CHECK(setenv("FAKE_CUDA_H2D_BW_GBPS", "0.01", 1) == 0);
    CHECK(setenv("FAKE_CUDA_D2H_BW_GBPS", "0.01", 1) == 0);
    double start = seconds();
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen %s: %s\n", argv[1], dlerror()); return 1; }

    CUresult (CUDAAPI *init)(unsigned int), (CUDAAPI *get)(CUdevice *, int);
    CUresult (CUDAAPI *retain)(CUcontext *, CUdevice), (CUDAAPI *set)(CUcontext);
    CUresult (CUDAAPI *release)(CUdevice);
    CUresult (CUDAAPI *create_stream)(CUstream *, unsigned int);
    CUresult (CUDAAPI *destroy_stream)(CUstream), (CUDAAPI *sync_stream)(CUstream);
    CUresult (CUDAAPI *alloc)(CUdeviceptr *, size_t), (CUDAAPI *free_device)(CUdeviceptr);
    CUresult (CUDAAPI *h2d)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (CUDAAPI *d2h)(void *, CUdeviceptr, size_t, CUstream);
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
    LOAD(sync_stream, "cuStreamSynchronize");
    LOAD(alloc, "cuMemAlloc_v2"); LOAD(free_device, "cuMemFree_v2");
    LOAD(h2d, "cuMemcpyHtoDAsync_v2"); LOAD(d2h, "cuMemcpyDtoHAsync_v2");
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
    OK(get(&device, 0)); OK(retain(&context, device)); OK(set(context));
    CUstream a = NULL, b = NULL, nb = NULL;
    OK(create_stream(&a, CU_STREAM_DEFAULT));
    OK(create_stream(&b, CU_STREAM_DEFAULT));
    OK(create_stream(&nb, CU_STREAM_NON_BLOCKING));
    enum { BYTES = 2000000 };
    CUdeviceptr ptr = 0;
    auto *host = static_cast<unsigned char *>(calloc(BYTES, 1));
    CHECK(host != NULL);
    OK(alloc(&ptr, BYTES)); CHECK(ptr != 0); /* Opaque device address: never dereference. */
    CUevent slow = NULL, e1 = NULL, e2 = NULL, done = NULL;
    OK(create_event(&slow, CU_EVENT_DEFAULT));
    OK(create_event(&e1, CU_EVENT_DEFAULT));
    OK(create_event(&e2, CU_EVENT_DEFAULT));
    OK(create_event(&done, CU_EVENT_DEFAULT));

    /* Opposite copy engines keep resource FIFO from masquerading as a legacy barrier. */
    double t = seconds();
    OK(h2d(ptr, host, BYTES, a));
    OK(record_event(slow, a));
    OK(d2h(host, ptr, 1, CU_STREAM_LEGACY));
    OK(sync_stream(CU_STREAM_LEGACY));
    waited(seconds() - t, "blocking stream -> legacy");
    OK(query_event(slow));

    t = seconds();
    OK(h2d(ptr, host, BYTES, CU_STREAM_LEGACY));
    OK(record_event(slow, CU_STREAM_LEGACY));
    OK(d2h(host, ptr, 1, a));
    OK(sync_stream(a));
    waited(seconds() - t, "legacy -> blocking stream");
    OK(query_event(slow));

    OK(h2d(ptr, host, BYTES, nb));
    OK(record_event(slow, nb));
    OK(d2h(host, ptr, 1, CU_STREAM_LEGACY));
    OK(sync_stream(CU_STREAM_LEGACY));
    CHECK(query_event(slow) == CUDA_ERROR_NOT_READY); /* No implicit wait. */
    OK(sync_event(slow));
    OK(sync_stream(nb));

    static const char ptx[] =
        ".version 8.0\n.target sm_90\n.address_size 64\n"
        ".visible .entry multistream_probe_kernel() { ret; }\n";
    CUmodule module = NULL;
    CUfunction kernel = NULL;
    OK(load_module(&module, ptx));
    OK(get_function(&kernel, module, "multistream_probe_kernel"));

    /* Capture A -> e1 -> B (wait, 10 ms kernel, e2) -> A (wait). */
    CUgraph graph = NULL;
    CUgraphExec exec = NULL;
    CUDA_GRAPH_INSTANTIATE_PARAMS params = {};
    OK(begin_capture(a, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(record_event(e1, a));
    OK(wait_event(b, e1, 0));
    OK(launch_kernel(kernel, 1, 1, 1, 1, 1, 1, 0, b, NULL, NULL));
    OK(record_event(e2, b));
    OK(wait_event(a, e2, 0));
    OK(end_capture(a, &graph)); CHECK(graph != NULL);
    OK(instantiate(&exec, graph, &params)); CHECK(exec != NULL);
    for (int i = 0; i < 2; ++i) {
        t = seconds();
        OK(launch_graph(exec, a));
        OK(record_event(done, a));
        CHECK(query_event(e2) == CUDA_ERROR_NOT_READY);
        CHECK(query_event(done) == CUDA_ERROR_NOT_READY);
        OK(sync_event(done));
        CHECK(seconds() - t >= 0.005);
        OK(query_event(e2));
        OK(sync_stream(a));
    }
    OK(destroy_exec(exec)); OK(destroy_graph(graph));

    /* The imported B lane must flow back to A before end-capture. */
    graph = NULL;
    OK(begin_capture(a, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(record_event(e1, a));
    OK(wait_event(b, e1, 0));
    OK(launch_kernel(kernel, 1, 1, 1, 1, 1, 1, 0, b, NULL, NULL));
    OK(record_event(e2, b));
    CHECK(end_capture(a, &graph) == CUDA_ERROR_STREAM_CAPTURE_UNJOINED);
    CHECK(graph == NULL);

    OK(unload_module(module));
    OK(destroy_event(done)); OK(destroy_event(e2));
    OK(destroy_event(e1));
    OK(destroy_stream(nb)); OK(destroy_stream(b)); OK(destroy_stream(a));
    OK(free_device(ptr));

    /* Destroying an unsynchronized stream must not discard its in-flight copy.
       A synchronous free of its allocation must still wait for that work. */
    CUstream doomed = NULL;
    CUdeviceptr pending = 0;
    OK(create_stream(&doomed, CU_STREAM_NON_BLOCKING));
    OK(alloc(&pending, BYTES));
    OK(h2d(pending, host, BYTES, doomed));
    OK(record_event(slow, doomed));
    OK(destroy_stream(doomed));
    CHECK(query_event(slow) == CUDA_ERROR_NOT_READY);
    t = seconds();
    OK(free_device(pending));
    waited(seconds() - t, "synchronous free after stream destroy");
    OK(query_event(slow));
    OK(destroy_event(slow));

    OK(set(NULL)); OK(release(device));
    free(host);
    CHECK(dlclose(lib) == 0);
    CHECK(seconds() - start < 2.0);
    puts("PASS: legacy barriers, nonblocking stream, cross-stream graph, unjoined capture, destroy/free");
    return 0;
}
