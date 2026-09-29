// Standalone Driver ABI contract: link only dl, never libcuda or CUDA Runtime.
// Run as: api_contract /absolute/path/to/libcuda.so.1
#include <cuda.h>
#include <dlfcn.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#if CUDA_VERSION < 13000 || CUDA_VERSION >= 14000
#error "api_contract requires CUDA 13 Driver headers"
#endif

namespace {
void *library = nullptr;

[[noreturn]] void fail(const char *what, int line, int actual = -1, int expected = -1) {
    std::fprintf(stderr, "api_contract:%d: %s (actual %d, expected %d)\n", line, what, actual, expected);
    std::exit(1);
}
#define CHECK(x) do { if (!(x)) fail(#x, __LINE__); } while (false)
#define EXPECT(x, code) do { CUresult r = (x); if (r != (code)) \
    fail(#x, __LINE__, static_cast<int>(r), static_cast<int>(code)); } while (false)
#define OK(x) EXPECT(x, CUDA_SUCCESS)

template <typename T> T symbol(const char *name) {
    dlerror();
    void *p = dlsym(library, name);
    if (const char *error = dlerror(); error || !p) {
        std::fprintf(stderr, "missing Driver ABI export %s: %s\n", name, error ? error : "null");
        std::exit(1);
    }
    return reinterpret_cast<T>(p);
}
// The stringized name is the actual un-macro-mapped ABI export; the type comes
// from cuda.h (including CUDA 13's v4 context-create and v2 elapsed-time ABI).
#define API(name, ...) symbol<decltype(&::name)>(#name)(__VA_ARGS__)
using Ctx12 = CUresult (CUDAAPI *)(CUcontext *, unsigned int, CUdevice);
using Proc12 = CUresult (CUDAAPI *)(const char *, void **, int, cuuint64_t);
using Elapsed12 = CUresult (CUDAAPI *)(float *, CUevent, CUevent);
using CaptureInfoLegacy = CUresult (CUDAAPI *)(CUstream, CUstreamCaptureStatus *, cuuint64_t *);
using CaptureInfoV2 = CUresult (CUDAAPI *)(CUstream, CUstreamCaptureStatus *, cuuint64_t *,
                                           CUgraph *, const CUgraphNode **, size_t *);
using GetProc13 = CUresult (CUDAAPI *)(const char *, void **, int, cuuint64_t,
                                       CUdriverProcAddressQueryResult *);

// One-for-one with concrete entry points in src/driver_api/{common,cu12,cu13}.c; resolver-only aliases
// (e.g. cuMemAlloc without _v2) are checked separately below.
constexpr std::string_view exports[] = {
    "cuInit", "cuDriverGetVersion", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetName",
    "cuDeviceGetUuid_v2", "cuDeviceGetUuid", "cuDeviceTotalMem_v2", "cuDeviceGetAttribute",
    "cuDeviceCanAccessPeer", "cuDevicePrimaryCtxRetain", "cuDevicePrimaryCtxGetState",
    "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxSetFlags_v2", "cuDevicePrimaryCtxReset_v2",
    "cuCtxCreate_v2", "cuCtxCreate_v4", "cuCtxDestroy_v2", "cuCtxGetCurrent",
    "cuCtxSetCurrent", "cuCtxPushCurrent_v2", "cuCtxPopCurrent_v2", "cuCtxGetDevice",
    "cuCtxGetFlags", "cuCtxGetApiVersion", "cuCtxSynchronize", "cuCtxEnablePeerAccess",
    "cuCtxDisablePeerAccess", "cuCtxGetStreamPriorityRange", "cuModuleGetLoadingMode",
    "cuStreamCreate", "cuStreamCreateWithPriority", "cuStreamDestroy_v2", "cuStreamGetCtx",
    "cuStreamGetDevice", "cuStreamGetFlags", "cuStreamGetPriority", "cuStreamQuery",
    "cuStreamSynchronize", "cuMemGetInfo_v2", "cuMemAlloc_v2", "cuMemFree_v2",
    "cuMemAllocAsync", "cuMemFreeAsync", "cuPointerGetAttribute", "cuMemcpyHtoD_v2",
    "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoH_v2", "cuMemcpyDtoHAsync_v2",
    "cuMemcpyDtoD_v2", "cuMemcpyDtoDAsync_v2", "cuMemcpyPeer", "cuMemcpyPeerAsync",
    "cuMemsetD8_v2", "cuMemsetD8Async", "cuEventCreate", "cuEventDestroy_v2",
    "cuEventRecord", "cuEventRecordWithFlags", "cuEventQuery", "cuEventSynchronize",
    "cuEventElapsedTime", "cuEventElapsedTime_v2", "cuStreamWaitEvent",
    "cuStreamBeginCapture_v2", "cuStreamEndCapture", "cuStreamIsCapturing",
    "cuStreamGetCaptureInfo", "cuStreamGetCaptureInfo_v2", "cuStreamGetCaptureInfo_v3",
    "cuGraphInstantiateWithFlags", "cuGraphInstantiateWithParams", "cuGraphGetNodes",
    "cuGraphLaunch", "cuGraphDestroy", "cuGraphExecDestroy",
    "cuModuleLoadData", "cuModuleGetFunction", "cuModuleUnload", "cuLibraryLoadData",
    "cuLibraryUnload", "cuLibraryGetKernel", "cuLibraryGetModule", "cuKernelGetFunction",
    "cuLaunchKernel",
    "cuGetErrorName", "cuGetErrorString", "cuGetProcAddress", "cuGetProcAddress_v2",
    "cuGetExportTable"
};

void resolver() {
    auto get12 = symbol<Proc12>("cuGetProcAddress");
    auto get13 = symbol<GetProc13>("cuGetProcAddress_v2");
    void *p = nullptr;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    EXPECT(get13("cuInit", nullptr, 13000, 0, &status), CUDA_ERROR_INVALID_VALUE);
    for (auto name : exports) {
        // Non-versioned names with distinct historical ABIs are selected below.
        if (name == "cuGetProcAddress" || name == "cuEventElapsedTime" ||
            name == "cuStreamGetCaptureInfo") continue;
        std::string text(name);
        p = nullptr;
        status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        OK(get13(text.c_str(), &p, 13000, 0, &status));
        CHECK(p == dlsym(library, text.c_str()) && status == CU_GET_PROC_ADDRESS_SUCCESS);
    }
    for (auto name : {"cuDeviceTotalMem", "cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxSetFlags",
                      "cuDevicePrimaryCtxReset", "cuCtxDestroy", "cuCtxPushCurrent", "cuCtxPopCurrent",
                      "cuStreamDestroy", "cuMemGetInfo", "cuMemAlloc", "cuMemFree", "cuMemcpyHtoD",
                      "cuMemcpyHtoDAsync", "cuMemcpyDtoH", "cuMemcpyDtoHAsync", "cuMemcpyDtoD",
                      "cuMemcpyDtoDAsync", "cuMemsetD8", "cuEventDestroy"}) {
        OK(get13(name, &p, 13000, 0, &status));
        CHECK(p != nullptr && status == CU_GET_PROC_ADDRESS_SUCCESS);
        CHECK(dlsym(library, name) == nullptr); // resolver alias, not a physical export
    }
    OK(get13("cuCtxCreate", &p, 12000, 0, &status));
    CHECK(p == dlsym(library, "cuCtxCreate_v2"));
    OK(get13("cuCtxCreate", &p, 13000, 0, &status));
    CHECK(p == dlsym(library, "cuCtxCreate_v4"));
    OK(get13("cuGetProcAddress", &p, 11000, 0, &status));
    CHECK(p == dlsym(library, "cuGetProcAddress"));
    OK(get13("cuGetProcAddress", &p, 12000, 0, &status));
    CHECK(p == dlsym(library, "cuGetProcAddress_v2"));
    OK(get13("cuEventElapsedTime", &p, 12000, 0, &status));
    CHECK(p == dlsym(library, "cuEventElapsedTime"));
    OK(get13("cuEventElapsedTime", &p, 13000, 0, &status));
    CHECK(p == dlsym(library, "cuEventElapsedTime_v2"));
    OK(get13("cuStreamBeginCapture", &p, 9000, 0, &status));
    CHECK(!p && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
    OK(get13("cuStreamBeginCapture", &p, 10000, 0, &status));
    CHECK(!p && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
    OK(get13("cuStreamBeginCapture", &p, 10010, 0, &status));
    CHECK(p == dlsym(library, "cuStreamBeginCapture_v2"));
    OK(get13("cuStreamGetCaptureInfo", &p, 10010, 0, &status));
    CHECK(p == dlsym(library, "cuStreamGetCaptureInfo"));
    OK(get13("cuStreamGetCaptureInfo", &p, 11030, 0, &status));
    CHECK(p == dlsym(library, "cuStreamGetCaptureInfo_v2"));
    OK(get13("cuStreamGetCaptureInfo", &p, 12080, 0, &status));
    CHECK(p == dlsym(library, "cuStreamGetCaptureInfo_v2"));
    OK(get13("cuStreamGetCaptureInfo", &p, 13000, 0, &status));
    CHECK(p == dlsym(library, "cuStreamGetCaptureInfo_v3"));
    OK(get13("cuGraphInstantiateWithFlags", &p, 11040, 0, &status));
    CHECK(p == dlsym(library, "cuGraphInstantiateWithFlags"));
    OK(get13("cuGraphInstantiate", &p, 11040, 0, &status));
    CHECK(p == dlsym(library, "cuGraphInstantiateWithFlags"));
    OK(get12("cuInit", &p, 12000, 0));
    CHECK(p == dlsym(library, "cuInit"));
    p = reinterpret_cast<void *>(1);
    OK(get13("cuNotImplemented", &p, 13000, 0, &status));
    CHECK(!p && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
    OK(get13(nullptr, &p, 13000, 0, &status));
    CHECK(!p && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
    CHECK(dlsym(library, "cudaMalloc") == nullptr); // Driver only, never Runtime
}

void initialization_and_devices(CUdevice &dev) {
    EXPECT(API(cuInit, 1), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuInit, 0));
    int version = 0, count = 0, attr = 0, peer = -1;
    EXPECT(API(cuDriverGetVersion, nullptr), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuDriverGetVersion, &version)); CHECK(version >= 13000);
    EXPECT(API(cuDeviceGetCount, nullptr), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuDeviceGetCount, &count)); CHECK(count >= 1 && count <= 8);
    EXPECT(API(cuDeviceGet, &dev, count), CUDA_ERROR_INVALID_DEVICE);
    OK(API(cuDeviceGet, &dev, 0)); CHECK(dev == 0);
    char name[128]{};
    OK(API(cuDeviceGetName, name, sizeof(name), dev)); CHECK(name[0]);
    EXPECT(API(cuDeviceGetName, name, 0, dev), CUDA_ERROR_INVALID_VALUE);
    CUuuid a{}, b{};
    OK(API(cuDeviceGetUuid_v2, &a, dev));
    OK(symbol<decltype(&::cuDeviceGetUuid_v2)>("cuDeviceGetUuid")(&b, dev));
    CHECK(std::memcmp(&a, &b, sizeof(a)) == 0);
    size_t total = 0;
    OK(API(cuDeviceTotalMem_v2, &total, dev)); CHECK(total > 0);
    OK(API(cuDeviceGetAttribute, &attr, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)); CHECK(attr > 0);
    EXPECT(API(cuDeviceGetAttribute, &attr, static_cast<CUdevice_attribute>(0), dev), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuDeviceCanAccessPeer, &peer, dev, dev)); CHECK(peer == 0);
    EXPECT(API(cuDeviceCanAccessPeer, &peer, dev, count), CUDA_ERROR_INVALID_DEVICE);
    if (count > 1) { OK(API(cuDeviceCanAccessPeer, &peer, dev, 1)); CHECK(peer == 1); }
    CUmoduleLoadingMode mode{};
    EXPECT(API(cuModuleGetLoadingMode, nullptr), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuModuleGetLoadingMode, &mode)); CHECK(mode == CU_MODULE_LAZY_LOADING);
    const char *name_result = nullptr, *description = nullptr;
    EXPECT(API(cuGetErrorName, CUDA_SUCCESS, nullptr), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuGetErrorName, CUDA_ERROR_INVALID_VALUE, &name_result));
    CHECK(std::strcmp(name_result, "CUDA_ERROR_INVALID_VALUE") == 0);
    EXPECT(API(cuGetErrorString, CUDA_SUCCESS, nullptr), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuGetErrorString, CUDA_SUCCESS, &description)); CHECK(description && *description);
    CUuuid unknown{};
    const void *table = nullptr;
    EXPECT(API(cuGetExportTable, nullptr, &unknown), CUDA_ERROR_INVALID_VALUE);
    // Private Runtime UUIDs/tables are undocumented: exercise only the unknown-UUID rejection.
    EXPECT(API(cuGetExportTable, &table, &unknown), CUDA_ERROR_NOT_SUPPORTED);
    CHECK(table == nullptr);
}

void contexts(CUdevice dev, CUcontext &ctx) {
    unsigned flags = 99, api_version = 0;
    int active = -1;
    OK(API(cuDevicePrimaryCtxGetState, dev, &flags, &active)); CHECK(active == 0);
    OK(API(cuDevicePrimaryCtxSetFlags_v2, dev, 0));
    CUcontext primary = nullptr;
    OK(API(cuDevicePrimaryCtxRetain, &primary, dev)); CHECK(primary);
    OK(API(cuDevicePrimaryCtxGetState, dev, &flags, &active)); CHECK(active == 1);
    EXPECT(API(cuDevicePrimaryCtxSetFlags_v2, dev, 0), CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE);
    OK(API(cuCtxSetCurrent, primary));
    CUcontext current = nullptr;
    OK(API(cuCtxGetCurrent, &current)); CHECK(current == primary);
    OK(API(cuCtxGetApiVersion, primary, &api_version)); CHECK(api_version >= 13000);
    int least = 0, greatest = 0;
    OK(API(cuCtxGetStreamPriorityRange, &least, &greatest)); CHECK(least >= greatest);
    EXPECT(API(cuCtxGetStreamPriorityRange, nullptr, &greatest), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuCtxSetCurrent, nullptr));
    EXPECT(API(cuCtxSynchronize), CUDA_ERROR_INVALID_CONTEXT);
    EXPECT(API(cuCtxGetDevice, &dev), CUDA_ERROR_INVALID_CONTEXT);
    OK(API(cuDevicePrimaryCtxRelease_v2, dev));
    EXPECT(API(cuCtxGetApiVersion, primary, &api_version), CUDA_ERROR_INVALID_CONTEXT);
    OK(API(cuDevicePrimaryCtxReset_v2, dev));

    auto create12 = symbol<Ctx12>("cuCtxCreate_v2");
    OK(create12(&ctx, 0, dev)); CHECK(ctx);
    OK(API(cuCtxGetCurrent, &current)); CHECK(current == ctx);
    CUdevice owner = -1;
    OK(API(cuCtxGetDevice, &owner)); CHECK(owner == dev);
    OK(API(cuCtxGetFlags, &flags)); CHECK(flags == 0);
    OK(API(cuCtxGetApiVersion, ctx, &api_version));
    OK(API(cuCtxPushCurrent_v2, ctx));
    OK(API(cuCtxPopCurrent_v2, &current)); CHECK(current == ctx);
    OK(API(cuCtxGetCurrent, &current)); CHECK(current == ctx);
    EXPECT(API(cuCtxEnablePeerAccess, ctx, 0), CUDA_ERROR_INVALID_CONTEXT);
    EXPECT(API(cuCtxDisablePeerAccess, ctx), CUDA_ERROR_INVALID_CONTEXT);
    CUcontext second = nullptr;
    EXPECT(API(cuCtxCreate_v4, &second, nullptr, 0, -1), CUDA_ERROR_INVALID_DEVICE);
    CUctxCreateParams params{};
    EXPECT(API(cuCtxCreate_v4, &second, &params, 0, dev), CUDA_ERROR_NOT_SUPPORTED);
    OK(API(cuCtxCreate_v4, &second, nullptr, 0, dev)); CHECK(second != ctx);
    // These two contexts share one device; independent-device peer behavior is
    // exercised by multidevice_probe.
    EXPECT(API(cuCtxEnablePeerAccess, ctx, 0), CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
    EXPECT(API(cuCtxDisablePeerAccess, ctx), CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
    OK(API(cuCtxDestroy_v2, second));
    OK(API(cuCtxGetCurrent, &current)); CHECK(current == ctx);
    EXPECT(API(cuCtxSetCurrent, second), CUDA_ERROR_INVALID_CONTEXT);
}

void streams(CUcontext ctx, CUdevice dev, CUstream &s) {
    EXPECT(API(cuStreamCreate, nullptr, 0), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuStreamCreate, &s, CU_STREAM_NON_BLOCKING)); CHECK(s);
    CUstream priority = nullptr;
    OK(API(cuStreamCreateWithPriority, &priority, 0, -1));
    int value = 0;
    unsigned flags = 0;
    CUcontext owner = nullptr;
    CUdevice device = -1;
    OK(API(cuStreamGetCtx, s, &owner)); CHECK(owner == ctx);
    OK(API(cuStreamGetDevice, s, &device)); CHECK(device == dev);
    OK(API(cuStreamGetFlags, s, &flags)); CHECK(flags == CU_STREAM_NON_BLOCKING);
    OK(API(cuStreamGetPriority, priority, &value)); CHECK(value == -1);
    OK(API(cuStreamGetPriority, s, &value)); CHECK(value == 0);
    CUstream clamped = nullptr;
    OK(API(cuStreamCreateWithPriority, &clamped, 0, -99));
    OK(API(cuStreamGetPriority, clamped, &value)); CHECK(value == -1);
    OK(API(cuStreamDestroy_v2, clamped));
    OK(API(cuStreamCreateWithPriority, &clamped, 0, 99));
    OK(API(cuStreamGetPriority, clamped, &value)); CHECK(value == 0);
    OK(API(cuStreamDestroy_v2, clamped));
    OK(API(cuStreamQuery, s)); OK(API(cuStreamSynchronize, s));
    OK(API(cuStreamDestroy_v2, priority));
    EXPECT(API(cuStreamQuery, priority), CUDA_ERROR_INVALID_HANDLE);
    EXPECT(API(cuStreamDestroy_v2, priority), CUDA_ERROR_INVALID_HANDLE);
}

void memory(CUcontext ctx, CUstream s, CUdeviceptr &dst, CUdeviceptr &src) {
    size_t free_bytes = 0, total = 0;
    OK(API(cuMemGetInfo_v2, &free_bytes, &total)); CHECK(total > 0 && free_bytes == total);
    EXPECT(API(cuMemAlloc_v2, &dst, 0), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuMemAlloc_v2, &dst, 64)); OK(API(cuMemAlloc_v2, &src, 64));
    CHECK(dst && src && dst != src); // Virtual, non-dereferenceable device addresses.
    size_t after = 0;
    OK(API(cuMemGetInfo_v2, &after, &total)); CHECK(free_bytes - after == 128);
    CUmemorytype type{};
    CUcontext owner = nullptr;
    CUdeviceptr start = 0, pointer = 0;
    size_t range = 0;
    OK(API(cuPointerGetAttribute, &type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dst));
    CHECK(type == CU_MEMORYTYPE_DEVICE);
    OK(API(cuPointerGetAttribute, &owner, CU_POINTER_ATTRIBUTE_CONTEXT, dst)); CHECK(owner == ctx);
    OK(API(cuPointerGetAttribute, &start, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, dst + 1)); CHECK(start == dst);
    OK(API(cuPointerGetAttribute, &range, CU_POINTER_ATTRIBUTE_RANGE_SIZE, dst + 1)); CHECK(range == 64);
    OK(API(cuPointerGetAttribute, &pointer, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, dst + 1)); CHECK(pointer == dst + 1);
    EXPECT(API(cuPointerGetAttribute, &type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, 0), CUDA_ERROR_INVALID_VALUE);
    EXPECT(API(cuPointerGetAttribute, nullptr, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dst), CUDA_ERROR_INVALID_VALUE);
    EXPECT(API(cuPointerGetAttribute, &type, static_cast<CUpointer_attribute>(999), dst), CUDA_ERROR_NOT_SUPPORTED);
    std::array<unsigned char, 64> host{};
    OK(API(cuMemcpyHtoD_v2, dst, host.data(), 1));
    OK(API(cuMemcpyDtoH_v2, host.data(), dst, 1));
    OK(API(cuMemcpyDtoD_v2, dst, src, 1));
    OK(API(cuMemcpyHtoDAsync_v2, dst, host.data(), 1, s));
    OK(API(cuMemcpyDtoHAsync_v2, host.data(), dst, 1, s));
    OK(API(cuMemcpyDtoDAsync_v2, dst, src, 1, s));
    EXPECT(API(cuMemcpyHtoD_v2, dst, nullptr, 1), CUDA_ERROR_INVALID_VALUE);
    EXPECT(API(cuMemcpyDtoH_v2, nullptr, dst, 1), CUDA_ERROR_INVALID_VALUE);
    EXPECT(API(cuMemcpyDtoD_v2, dst, src, 65), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuMemsetD8_v2, dst, 0x5a, 1));
    OK(API(cuMemsetD8Async, dst, 0xa5, 1, s));
    EXPECT(API(cuMemsetD8_v2, dst, 0, 65), CUDA_ERROR_INVALID_VALUE);
    // These entry points also accept same-context pointers; cross-device P2P
    // queue/order validation belongs to multidevice_probe.
    OK(API(cuMemcpyPeer, dst, ctx, src, ctx, 1));
    OK(API(cuMemcpyPeerAsync, dst, ctx, src, ctx, 1, s));
    EXPECT(API(cuMemcpyPeer, dst, nullptr, src, ctx, 1), CUDA_ERROR_INVALID_CONTEXT);
    CUdeviceptr temporary = 0;
    OK(API(cuMemAllocAsync, &temporary, 32, s)); CHECK(temporary && temporary != dst);
    OK(API(cuMemFreeAsync, temporary, s));
    EXPECT(API(cuMemFreeAsync, temporary, s), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuStreamSynchronize, s));
    OK(API(cuMemFree_v2, src));
    EXPECT(API(cuMemFree_v2, src), CUDA_ERROR_INVALID_VALUE);
    EXPECT(API(cuPointerGetAttribute, &type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, src), CUDA_ERROR_INVALID_VALUE);
}

void events_graphs_modules(CUstream s, CUdeviceptr dst) {
    CUevent first = nullptr, second = nullptr;
    EXPECT(API(cuEventCreate, nullptr, 0), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuEventCreate, &first, CU_EVENT_DEFAULT));
    OK(API(cuEventCreate, &second, CU_EVENT_DEFAULT));
    float ms = 0;
    EXPECT(symbol<Elapsed12>("cuEventElapsedTime")(&ms, first, second), CUDA_ERROR_NOT_READY);
    OK(API(cuEventRecord, first, s));
    OK(API(cuEventRecordWithFlags, second, s, 0));
    EXPECT(API(cuEventRecordWithFlags, second, s, CU_EVENT_RECORD_EXTERNAL), CUDA_ERROR_NOT_SUPPORTED);
    OK(API(cuStreamWaitEvent, s, first, 0));
    EXPECT(API(cuStreamWaitEvent, s, first, 1), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuEventSynchronize, second)); OK(API(cuEventQuery, first));
    OK(symbol<Elapsed12>("cuEventElapsedTime")(&ms, first, second));
    OK(API(cuEventElapsedTime_v2, &ms, first, second)); CHECK(ms >= 0);
    CUstreamCaptureStatus capture{};
    OK(API(cuStreamIsCapturing, s, &capture)); CHECK(capture == CU_STREAM_CAPTURE_STATUS_NONE);
    cuuint64_t capture_id = ~cuuint64_t{0};
    OK(symbol<CaptureInfoLegacy>("cuStreamGetCaptureInfo")(s, &capture, &capture_id));
    CHECK(capture == CU_STREAM_CAPTURE_STATUS_NONE && capture_id == 0);
    CUgraph graph = nullptr;
    EXPECT(API(cuStreamEndCapture, s, &graph), CUDA_ERROR_ILLEGAL_STATE);
    OK(API(cuStreamBeginCapture_v2, s, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(API(cuStreamIsCapturing, s, &capture)); CHECK(capture == CU_STREAM_CAPTURE_STATUS_ACTIVE);
    CUgraph capture_graph = nullptr;
    const CUgraphNode *dependencies = nullptr;
    const CUgraphEdgeData *edges = nullptr;
    size_t dependency_count = 0;
    OK(symbol<CaptureInfoV2>("cuStreamGetCaptureInfo_v2")(
        s, &capture, &capture_id, &capture_graph, &dependencies, &dependency_count));
    CHECK(capture_id && capture_graph && capture == CU_STREAM_CAPTURE_STATUS_ACTIVE);
    CHECK(!dependencies && dependency_count == 0);
    size_t active_nodes = 99;
    OK(API(cuGraphGetNodes, capture_graph, nullptr, &active_nodes)); CHECK(active_nodes == 0);
    EXPECT(API(cuStreamSynchronize, s), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    OK(API(cuStreamIsCapturing, s, &capture)); CHECK(capture == CU_STREAM_CAPTURE_STATUS_INVALIDATED);
    EXPECT(API(cuStreamEndCapture, s, &graph), CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
    CHECK(graph == nullptr);
    OK(API(cuStreamBeginCapture_v2, s, CU_STREAM_CAPTURE_MODE_GLOBAL));
    EXPECT(API(cuCtxSynchronize), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    EXPECT(API(cuStreamEndCapture, s, &graph), CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
    OK(API(cuStreamBeginCapture_v2, s, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(API(cuStreamGetCaptureInfo_v3, s, &capture, &capture_id, &capture_graph,
           &dependencies, &edges, &dependency_count));
    CHECK(!edges && dependency_count == 0);
    EXPECT(API(cuMemAllocAsync, &dst, 4, s), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    OK(API(cuMemsetD8Async, dst, 0, 1, s));
    OK(API(cuStreamGetCaptureInfo_v3, s, &capture, &capture_id, &capture_graph,
           &dependencies, &edges, &dependency_count));
    CHECK(dependencies && edges && dependency_count == 1 && dependencies[0]);
    active_nodes = 0;
    OK(API(cuGraphGetNodes, capture_graph, nullptr, &active_nodes)); CHECK(active_nodes == 1);
    CUgraphNode captured_node = dependencies[0];
    OK(API(cuStreamEndCapture, s, &graph)); CHECK(graph == capture_graph);
    size_t node_count = 0;
    EXPECT(API(cuGraphGetNodes, graph, nullptr, nullptr), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuGraphGetNodes, graph, nullptr, &node_count)); CHECK(node_count == 1);
    CUgraphNode node = nullptr;
    OK(API(cuGraphGetNodes, graph, &node, &node_count));
    CHECK(node_count == 1 && node == captured_node);
    CUgraphExec exec = nullptr;
    CUDA_GRAPH_INSTANTIATE_PARAMS params{};
    EXPECT(API(cuGraphInstantiateWithParams, &exec, graph, nullptr), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuGraphInstantiateWithParams, &exec, graph, &params));
    CHECK(exec && params.result_out == CUDA_GRAPH_INSTANTIATE_SUCCESS);
    OK(API(cuGraphLaunch, exec, s)); OK(API(cuStreamSynchronize, s));
    CUgraphExec flagged = nullptr;
    constexpr unsigned long long graph_flags = CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH |
                                               CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
    EXPECT(API(cuGraphInstantiateWithFlags, &flagged, graph, 4), CUDA_ERROR_NOT_SUPPORTED);
    OK(API(cuGraphInstantiateWithFlags, &flagged, graph, graph_flags));
    OK(API(cuGraphLaunch, flagged, s)); OK(API(cuStreamSynchronize, s));
    OK(API(cuGraphExecDestroy, flagged));
    OK(API(cuGraphExecDestroy, exec));
    EXPECT(API(cuGraphLaunch, exec, s), CUDA_ERROR_INVALID_HANDLE);
    OK(API(cuGraphDestroy, graph));
    EXPECT(API(cuGraphGetNodes, graph, nullptr, &node_count), CUDA_ERROR_INVALID_HANDLE);
    EXPECT(API(cuGraphDestroy, graph), CUDA_ERROR_INVALID_HANDLE);
    CUlibrary library = nullptr;
    EXPECT(API(cuLibraryLoadData, nullptr, "ptx", nullptr, nullptr, 0, nullptr, nullptr, 0), CUDA_ERROR_INVALID_VALUE);
    EXPECT(API(cuLibraryLoadData, &library, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0), CUDA_ERROR_INVALID_VALUE);
    CUjit_option option = CU_JIT_MAX_REGISTERS;
    void *value = nullptr;
    EXPECT(API(cuLibraryLoadData, &library, "ptx", &option, &value, 1, nullptr, nullptr, 0), CUDA_ERROR_NOT_SUPPORTED);
    static constexpr char ptx[] = ".version 8.0\n.target sm_90\n.address_size 64\n"
                                  ".visible .entry contract_kernel() { ret; }\n";
    CUlibraryOption preserve = CU_LIBRARY_BINARY_IS_PRESERVED;
    OK(API(cuLibraryLoadData, &library, ptx, nullptr, nullptr, 0, &preserve, &value, 1));
    CHECK(library);
    CUlibraryOption unsupported = CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE;
    CUlibrary rejected = nullptr;
    EXPECT(API(cuLibraryLoadData, &rejected, ptx, nullptr, nullptr, 0, &unsupported, &value, 1), CUDA_ERROR_NOT_SUPPORTED);
    CUkernel kernel = nullptr, same_kernel = nullptr;
    CUmodule library_module = nullptr;
    CUfunction library_function = nullptr, same_function = nullptr;
    EXPECT(API(cuLibraryGetKernel, nullptr, library, "contract_kernel"), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuLibraryGetKernel, &kernel, library, "contract_kernel"));
    OK(API(cuLibraryGetKernel, &same_kernel, library, "contract_kernel"));
    CHECK(kernel && kernel == same_kernel);
    OK(API(cuLibraryGetModule, &library_module, library)); CHECK(library_module);
    OK(API(cuKernelGetFunction, &library_function, kernel));
    OK(API(cuKernelGetFunction, &same_function, kernel));
    CHECK(library_function && library_function == same_function);
    OK(API(cuLaunchKernel, reinterpret_cast<CUfunction>(kernel), 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr));
    OK(API(cuLaunchKernel, library_function, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr));
    OK(API(cuLibraryUnload, library));
    EXPECT(API(cuKernelGetFunction, &library_function, kernel), CUDA_ERROR_INVALID_HANDLE);
    EXPECT(API(cuLibraryGetModule, &library_module, library), CUDA_ERROR_INVALID_HANDLE);
    EXPECT(API(cuLaunchKernel, same_function, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr), CUDA_ERROR_INVALID_HANDLE);
    EXPECT(API(cuLibraryUnload, library), CUDA_ERROR_INVALID_HANDLE);
    CUmodule mod = nullptr;
    CUfunction fn = nullptr;
    EXPECT(API(cuModuleLoadData, &mod, nullptr), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuModuleLoadData, &mod, ptx));
    OK(API(cuModuleGetFunction, &fn, mod, "contract_kernel"));
    EXPECT(API(cuLaunchKernel, fn, 0, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr), CUDA_ERROR_INVALID_VALUE);
    OK(API(cuLaunchKernel, fn, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr));
    OK(API(cuStreamSynchronize, s));
    OK(API(cuModuleUnload, mod));
    EXPECT(API(cuLaunchKernel, fn, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr), CUDA_ERROR_INVALID_HANDLE);
    OK(API(cuEventDestroy_v2, first)); OK(API(cuEventDestroy_v2, second));
    EXPECT(API(cuEventQuery, first), CUDA_ERROR_INVALID_HANDLE);
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: api_contract /path/to/libcuda.so.1\n"); return 2; }
    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    for (auto name : exports) { std::string text(name); CHECK(symbol<void *>(text.c_str())); }
    resolver();
    CUdevice dev = -1;
    initialization_and_devices(dev);
    CUcontext ctx = nullptr;
    contexts(dev, ctx);
    CUstream s = nullptr;
    streams(ctx, dev, s);
    CUdeviceptr dst = 0, src = 0;
    memory(ctx, s, dst, src);
    events_graphs_modules(s, dst);
    OK(API(cuMemFree_v2, dst));
    OK(API(cuStreamDestroy_v2, s));
    OK(API(cuCtxSynchronize));
    OK(API(cuCtxDestroy_v2, ctx));
    EXPECT(API(cuCtxGetApiVersion, ctx, nullptr), CUDA_ERROR_INVALID_VALUE);
    CHECK(dlclose(library) == 0);
    std::puts("PASS: Driver ABI exports, resolver, devices, contexts, streams, virtual memory, events, graphs and modules");
}
