/* Standalone CUDA 13 Driver ABI probe. Only cuda.h types are used: no CUDA
 * library is linked, and virtual device addresses are never dereferenced. */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if CUDA_VERSION < 13000 || CUDA_VERSION >= 14000
#error "Build this probe with CUDA 13 cuda.h"
#endif

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL: %s (%s:%d)\n", #expr, __FILE__, __LINE__); exit(1); \
} } while (0)
#define OK(expr) do { CUresult result_ = (expr); if (result_ != CUDA_SUCCESS) { \
    fprintf(stderr, "FAIL: %s returned %d (%s:%d)\n", #expr, (int)result_, __FILE__, __LINE__); exit(1); \
} } while (0)
#define ERROR(expr, expected) do { CUresult result_ = (expr); if (result_ != (expected)) { \
    fprintf(stderr, "FAIL: %s returned %d, expected %d (%s:%d)\n", \
            #expr, (int)result_, (int)(expected), __FILE__, __LINE__); exit(1); \
} } while (0)
#define LOAD(field, symbol) do { \
    api.field = reinterpret_cast<decltype(api.field)>(dlsym(lib, symbol)); \
    if (!api.field) { fprintf(stderr, "missing Driver ABI symbol %s: %s\n", symbol, dlerror()); exit(1); } \
} while (0)

/* Private function table: call only the ABI resolved from the supplied library. */
static struct {
    CUresult (CUDAAPI *init)(unsigned int);
    CUresult (CUDAAPI *get_proc)(const char *, void **, int, cuuint64_t,
                                  CUdriverProcAddressQueryResult *);
    CUresult (CUDAAPI *count)(int *), (CUDAAPI *get)(CUdevice *, int);
    CUresult (CUDAAPI *uuid)(CUuuid *, CUdevice);
    CUresult (CUDAAPI *can_peer)(int *, CUdevice, CUdevice);
    CUresult (CUDAAPI *retain)(CUcontext *, CUdevice), (CUDAAPI *release)(CUdevice);
    CUresult (CUDAAPI *create_ctx)(CUcontext *, CUctxCreateParams *, unsigned int, CUdevice);
    CUresult (CUDAAPI *destroy_ctx)(CUcontext), (CUDAAPI *set)(CUcontext);
    CUresult (CUDAAPI *current)(CUcontext *), (CUDAAPI *ctx_device)(CUdevice *);
    CUresult (CUDAAPI *ctx_sync)(void);
    CUresult (CUDAAPI *enable_peer)(CUcontext, unsigned int), (CUDAAPI *disable_peer)(CUcontext);
    CUresult (CUDAAPI *create_stream)(CUstream *, unsigned int);
    CUresult (CUDAAPI *destroy_stream)(CUstream), (CUDAAPI *sync_stream)(CUstream);
    CUresult (CUDAAPI *query_stream)(CUstream);
    CUresult (CUDAAPI *stream_ctx)(CUstream, CUcontext *);
    CUresult (CUDAAPI *stream_device)(CUstream, CUdevice *);
    CUresult (CUDAAPI *alloc)(CUdeviceptr *, size_t), (CUDAAPI *free_device)(CUdeviceptr);
    CUresult (CUDAAPI *mem_info)(size_t *, size_t *);
    CUresult (CUDAAPI *pointer_attribute)(void *, CUpointer_attribute, CUdeviceptr);
    CUresult (CUDAAPI *h2d)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (CUDAAPI *d2h)(void *, CUdeviceptr, size_t, CUstream);
    CUresult (CUDAAPI *d2d)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult (CUDAAPI *peer)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);
    CUresult (CUDAAPI *peer_async)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream);
    CUresult (CUDAAPI *create_event)(CUevent *, unsigned int);
    CUresult (CUDAAPI *record_event)(CUevent, CUstream);
    CUresult (CUDAAPI *record_event_flags)(CUevent, CUstream, unsigned int);
    CUresult (CUDAAPI *query_event)(CUevent), (CUDAAPI *destroy_event)(CUevent);
    CUresult (CUDAAPI *wait_event)(CUstream, CUevent, unsigned int);
} api;

static double now(void) {
    struct timespec ts;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void elapsed_between(double start, double lower, double upper, const char *label) {
    double elapsed = now() - start;
    if (elapsed < lower || elapsed > upper) {
        fprintf(stderr, "FAIL: %s took %.3f s (expected %.3f..%.3f s)\n",
                label, elapsed, lower, upper);
        exit(1);
    }
}

int main(int argc, char **argv) {
    enum { BYTES = 1500000 }; /* 0.15 s at 0.01 decimal GB/s. */
    CHECK(argc == 2);
    CHECK(setenv("FAKE_CUDA_H2D_BW_GBPS", "0.01", 1) == 0);
    CHECK(setenv("FAKE_CUDA_D2H_BW_GBPS", "0.01", 1) == 0);
    CHECK(setenv("FAKE_CUDA_HBM_BW_GBPS", "0.01", 1) == 0);
    CHECK(setenv("FAKE_CUDA_P2P_BW_GBPS", "0.01", 1) == 0);
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen %s: %s\n", argv[1], dlerror()); return 1; }
    LOAD(init, "cuInit"); LOAD(get_proc, "cuGetProcAddress_v2");
    LOAD(count, "cuDeviceGetCount"); LOAD(get, "cuDeviceGet");
    LOAD(uuid, "cuDeviceGetUuid_v2"); LOAD(can_peer, "cuDeviceCanAccessPeer");
    LOAD(retain, "cuDevicePrimaryCtxRetain"); LOAD(release, "cuDevicePrimaryCtxRelease_v2");
    LOAD(create_ctx, "cuCtxCreate_v4"); LOAD(destroy_ctx, "cuCtxDestroy_v2");
    LOAD(set, "cuCtxSetCurrent"); LOAD(current, "cuCtxGetCurrent");
    LOAD(ctx_device, "cuCtxGetDevice"); LOAD(ctx_sync, "cuCtxSynchronize");
    LOAD(enable_peer, "cuCtxEnablePeerAccess"); LOAD(disable_peer, "cuCtxDisablePeerAccess");
    LOAD(create_stream, "cuStreamCreate"); LOAD(destroy_stream, "cuStreamDestroy_v2");
    LOAD(sync_stream, "cuStreamSynchronize"); LOAD(query_stream, "cuStreamQuery");
    LOAD(stream_ctx, "cuStreamGetCtx"); LOAD(stream_device, "cuStreamGetDevice");
    LOAD(alloc, "cuMemAlloc_v2"); LOAD(free_device, "cuMemFree_v2");
    LOAD(mem_info, "cuMemGetInfo_v2"); LOAD(pointer_attribute, "cuPointerGetAttribute");
    LOAD(h2d, "cuMemcpyHtoDAsync_v2"); LOAD(d2h, "cuMemcpyDtoHAsync_v2");
    LOAD(d2d, "cuMemcpyDtoDAsync_v2");
    LOAD(peer, "cuMemcpyPeer"); LOAD(peer_async, "cuMemcpyPeerAsync");
    LOAD(create_event, "cuEventCreate"); LOAD(record_event, "cuEventRecord");
    LOAD(record_event_flags, "cuEventRecordWithFlags");
    LOAD(query_event, "cuEventQuery"); LOAD(destroy_event, "cuEventDestroy_v2");
    LOAD(wait_event, "cuStreamWaitEvent");
    const char *new_symbols[] = {"cuDeviceCanAccessPeer", "cuCtxEnablePeerAccess",
                                 "cuCtxDisablePeerAccess", "cuMemcpyPeer",
                                 "cuMemcpyPeerAsync", "cuEventRecordWithFlags"};
    for (size_t i = 0; i < sizeof(new_symbols) / sizeof(new_symbols[0]); ++i) {
        void *address = NULL;
        CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        OK(api.get_proc(new_symbols[i], &address, 13000, 0, &status));
        CHECK(address != NULL && status == CU_GET_PROC_ADDRESS_SUCCESS);
    }

    OK(api.init(0));
    int count = -1, can = -1;
    CUdevice dev[2] = {-1, -1}, owner = -1;
    CHECK(getenv("FAKE_CUDA_DEVICE_COUNT") && strcmp(getenv("FAKE_CUDA_DEVICE_COUNT"), "2") == 0);
    OK(api.count(&count)); CHECK(count == 2);
    OK(api.get(&dev[0], 0)); OK(api.get(&dev[1], 1)); CHECK(dev[0] != dev[1]);
    ERROR(api.get(&owner, 2), CUDA_ERROR_INVALID_DEVICE);
    CUuuid ids[2];
    OK(api.uuid(&ids[0], dev[0])); OK(api.uuid(&ids[1], dev[1]));
    CHECK(memcmp(ids[0].bytes, ids[1].bytes, sizeof(ids[0].bytes)) != 0);
    OK(api.can_peer(&can, dev[0], dev[1])); CHECK(can == 1);
    OK(api.can_peer(&can, dev[1], dev[0])); CHECK(can == 1);
    OK(api.can_peer(&can, dev[0], dev[0])); CHECK(can == 0);
    ERROR(api.can_peer(&can, dev[0], 2), CUDA_ERROR_INVALID_DEVICE);
    ERROR(api.can_peer(NULL, dev[0], dev[1]), CUDA_ERROR_INVALID_VALUE);

    CUcontext ctx[2] = {NULL, NULL}, extra = NULL, found = NULL;
    CUstream streams[2][3] = {{NULL}};
    CUdeviceptr ptr[2] = {0, 0}, extra_ptr = 0;
    auto *host = static_cast<unsigned char *>(calloc(BYTES, 1));
    CHECK(host != NULL);
    OK(api.retain(&ctx[0], dev[0])); OK(api.retain(&ctx[1], dev[1]));
    CHECK(ctx[0] && ctx[1] && ctx[0] != ctx[1]);
    for (int i = 0; i < 2; ++i) {
        OK(api.set(ctx[i])); OK(api.current(&found)); CHECK(found == ctx[i]);
        OK(api.ctx_device(&owner)); CHECK(owner == dev[i]);
        OK(api.alloc(&ptr[i], BYTES)); CHECK(ptr[i] != 0);
        for (int j = 0; j < 3; ++j) {
            OK(api.create_stream(&streams[i][j], CU_STREAM_NON_BLOCKING));
            OK(api.stream_ctx(streams[i][j], &found)); CHECK(found == ctx[i]);
            OK(api.stream_device(streams[i][j], &owner)); CHECK(owner == dev[i]);
        }
    }
    CHECK(ptr[0] != ptr[1]);
    for (int i = 0; i < 2; ++i) {
        size_t free_bytes = 0, total_bytes = 0;
        int ordinal = -1;
        OK(api.set(ctx[i]));
        OK(api.mem_info(&free_bytes, &total_bytes));
        CHECK(total_bytes - free_bytes == BYTES); /* Capacity is per device. */
        OK(api.pointer_attribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr[i]));
        CHECK(ordinal == i);
        OK(api.pointer_attribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr[1 - i]));
        CHECK(ordinal == 1 - i); /* UVA ownership metadata is not mapping permission. */
        CUdeviceptr accessible = 0;
        ERROR(api.pointer_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr[1 - i]),
              CUDA_ERROR_INVALID_VALUE);
    }
    OK(api.set(ctx[0]));
    ERROR(api.h2d(ptr[0], host, 1, streams[1][0]), CUDA_ERROR_INVALID_HANDLE);
    ERROR(api.d2d(ptr[0], ptr[1], 1, streams[0][0]), CUDA_ERROR_INVALID_VALUE);
    ERROR(api.enable_peer(ctx[1], 1), CUDA_ERROR_INVALID_VALUE);
    OK(api.enable_peer(ctx[1], 0));
    ERROR(api.enable_peer(ctx[1], 0), CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
    OK(api.set(ctx[1])); OK(api.enable_peer(ctx[0], 0));
    CUdeviceptr mapped_pointer = 0;
    OK(api.pointer_attribute(&mapped_pointer, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr[0]));
    CHECK(mapped_pointer == ptr[0]);
    OK(api.d2d(ptr[1], ptr[0], 1, streams[1][0]));
    OK(api.sync_stream(streams[1][0]));

    /* Six concurrent copies: H2D, D2H and compute must be separate on each
     * device, and identical queues on different devices must not serialize. */
    double t = now();
    for (int i = 0; i < 2; ++i) {
        OK(api.set(ctx[i]));
        OK(api.h2d(ptr[i], host, BYTES, streams[i][0]));
        OK(api.d2h(host, ptr[i], BYTES, streams[i][1]));
        OK(api.d2d(ptr[i], ptr[i], BYTES, streams[i][2]));
    }
    OK(api.set(ctx[0])); OK(api.ctx_sync());
    OK(api.set(ctx[1])); OK(api.ctx_sync());
    elapsed_between(t, 0.09, 0.29, "six independent device/resource queues");

    /* Waiting in device 1 must observe an event recorded in device 0. */
    CUevent event = NULL;
    OK(api.set(ctx[0])); OK(api.create_event(&event, CU_EVENT_DISABLE_TIMING));
    t = now();
    OK(api.h2d(ptr[0], host, BYTES, streams[0][0]));
    OK(api.record_event_flags(event, streams[0][0], 0));
    ERROR(api.record_event_flags(event, streams[0][0], CU_EVENT_RECORD_EXTERNAL), CUDA_ERROR_NOT_SUPPORTED);
    OK(api.set(ctx[1]));
    OK(api.wait_event(streams[1][0], event, 0));
    OK(api.d2d(ptr[1], ptr[1], 1, streams[1][0]));
    ERROR(api.query_stream(streams[1][0]), CUDA_ERROR_NOT_READY);
    OK(api.sync_stream(streams[1][0]));
    elapsed_between(t, 0.09, 0.29, "cross-device event wait");
    OK(api.set(ctx[0])); OK(api.query_event(event));

    /* A peer copy belongs to the destination stream, not an unrelated
     * destination stream. Its source context must also drain the copy. */
    OK(api.set(ctx[0]));
    t = now();
    OK(api.h2d(ptr[0], host, BYTES, streams[0][1]));
    OK(api.h2d(ptr[0], host, BYTES, streams[0][1]));
    OK(api.peer_async(ptr[0], ctx[0], ptr[1], ctx[1], BYTES, streams[0][0]));
    OK(api.sync_stream(streams[0][0]));
    elapsed_between(t, 0.09, 0.29, "peer async destination stream sync");
    ERROR(api.query_stream(streams[0][1]), CUDA_ERROR_NOT_READY);
    OK(api.set(ctx[1])); OK(api.ctx_sync());
    OK(api.set(ctx[0])); OK(api.sync_stream(streams[0][1]));

    t = now();
    OK(api.peer_async(ptr[0], ctx[0], ptr[1], ctx[1], BYTES, streams[0][0]));
    OK(api.set(ctx[1])); OK(api.ctx_sync());
    elapsed_between(t, 0.09, 0.29, "peer async source context drain");
    OK(api.set(ctx[0])); OK(api.query_stream(streams[0][0]));

    /* Each ordered direction has its own peer queue; same direction is FIFO
     * even across destination streams. */
    t = now();
    OK(api.peer_async(ptr[0], ctx[0], ptr[1], ctx[1], BYTES, streams[0][0]));
    OK(api.set(ctx[1]));
    OK(api.peer_async(ptr[1], ctx[1], ptr[0], ctx[0], BYTES, streams[1][0]));
    OK(api.sync_stream(streams[1][0]));
    OK(api.set(ctx[0])); OK(api.sync_stream(streams[0][0]));
    elapsed_between(t, 0.09, 0.29, "opposite peer directions overlap");
    t = now();
    OK(api.peer_async(ptr[0], ctx[0], ptr[1], ctx[1], BYTES, streams[0][0]));
    OK(api.peer_async(ptr[0], ctx[0], ptr[1], ctx[1], BYTES, streams[0][2]));
    ERROR(api.query_stream(streams[0][2]), CUDA_ERROR_NOT_READY);
    OK(api.sync_stream(streams[0][2]));
    elapsed_between(t, 0.23, 0.49, "same-direction peer FIFO");
    OK(api.sync_stream(streams[0][0]));
    t = now();
    OK(api.peer(ptr[0], ctx[0], ptr[1], ctx[1], BYTES));
    elapsed_between(t, 0.09, 0.29, "synchronous peer copy");

    ERROR(api.peer_async(ptr[0] + BYTES - 1, ctx[0], ptr[1], ctx[1], 2, streams[0][0]), CUDA_ERROR_INVALID_VALUE);
    ERROR(api.peer_async(ptr[0], ctx[0], ptr[1] + BYTES - 1, ctx[1], 2, streams[0][0]), CUDA_ERROR_INVALID_VALUE);
    ERROR(api.peer_async(ptr[0], ctx[1], ptr[1], ctx[1], 1, streams[0][0]), CUDA_ERROR_INVALID_CONTEXT);
    ERROR(api.peer_async(ptr[0], ctx[0], ptr[1], ctx[1], 1, streams[1][0]), CUDA_ERROR_INVALID_HANDLE);

    /* A second context on device 1 is distinct from its primary and owns its
     * allocation. Retired contexts must never be accepted as peer arguments. */
    OK(api.create_ctx(&extra, NULL, 0, dev[1])); CHECK(extra && extra != ctx[1]);
    OK(api.current(&found)); CHECK(found == extra);
    OK(api.ctx_device(&owner)); CHECK(owner == dev[1]);
    OK(api.alloc(&extra_ptr, 32)); CHECK(extra_ptr && extra_ptr != ptr[1]);
    OK(api.set(ctx[0]));
    ERROR(api.peer_async(ptr[0], ctx[0], extra_ptr, ctx[1], 1, streams[0][0]), CUDA_ERROR_INVALID_VALUE);
    OK(api.set(extra)); OK(api.destroy_ctx(extra));
    OK(api.set(ctx[0]));
    ERROR(api.enable_peer(extra, 0), CUDA_ERROR_INVALID_CONTEXT);
    ERROR(api.peer_async(ptr[0], ctx[0], extra_ptr, extra, 1, streams[0][0]), CUDA_ERROR_INVALID_CONTEXT);
    OK(api.disable_peer(ctx[1]));
    ERROR(api.disable_peer(ctx[1]), CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
    OK(api.set(ctx[1])); OK(api.disable_peer(ctx[0]));

    OK(api.set(ctx[0])); OK(api.destroy_event(event));
    for (int i = 0; i < 2; ++i) {
        OK(api.set(ctx[i]));
        for (int j = 0; j < 3; ++j) OK(api.destroy_stream(streams[i][j]));
        OK(api.free_device(ptr[i]));
    }
    OK(api.set(NULL));
    OK(api.release(dev[0])); OK(api.release(dev[1]));
    free(host);
    CHECK(dlclose(lib) == 0);
    puts("PASS: two devices, independent queues, cross-device events, peer access and copies");
    return 0;
}
