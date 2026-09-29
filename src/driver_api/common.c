#include "bridge_internal.h"
FORWARD(cuInit, core_init, (unsigned int flags), (flags))
FORWARD(cuDriverGetVersion, core_version, (int *version), (version))
FORWARD(cuDeviceGetCount, core_count, (int *count), (count))
FORWARD(cuDeviceGet, core_device, (CUdevice *device, int ordinal), (device, ordinal))
FORWARD(cuDeviceGetName, core_name, (char *name, int size, CUdevice device), (name, size, device))
FORWARD(cuDeviceGetUuid_v2, core_uuid, (CUuuid *uuid, CUdevice device), (uuid, device))
FORWARD(cuDeviceGetUuid, core_uuid, (CUuuid *uuid, CUdevice device), (uuid, device))
FORWARD(cuDeviceTotalMem_v2, core_memory, (size_t *bytes, CUdevice device), (bytes, device))
FORWARD(cuDeviceGetAttribute, core_attribute, (int *value, CUdevice_attribute attribute, CUdevice device), (value, attribute, device))
FORWARD(cuDeviceCanAccessPeer, core_can_access_peer,
        (int *can_access, CUdevice device, CUdevice peer), (can_access, device, peer))
FORWARD(cuDevicePrimaryCtxRetain, core_primary_retain, (CUcontext *context, CUdevice device), (context, device))
FORWARD(cuDevicePrimaryCtxGetState, core_primary_state, (CUdevice device, unsigned int *flags, int *active), (device, flags, active))
FORWARD(cuDevicePrimaryCtxRelease_v2, core_primary_release, (CUdevice device), (device))
FORWARD(cuDevicePrimaryCtxSetFlags_v2, core_primary_flags, (CUdevice device, unsigned int flags), (device, flags))
FORWARD(cuDevicePrimaryCtxReset_v2, core_primary_reset, (CUdevice device), (device))

FORWARD(cuCtxDestroy_v2, core_context_destroy, (CUcontext context), (context))
FORWARD(cuCtxGetCurrent, core_context_current, (CUcontext *context), (context))
FORWARD(cuCtxSetCurrent, core_context_set, (CUcontext context), (context))
FORWARD(cuCtxPushCurrent_v2, core_context_push, (CUcontext context), (context))
FORWARD(cuCtxPopCurrent_v2, core_context_pop, (CUcontext *context), (context))
FORWARD(cuCtxGetDevice, core_context_device, (CUdevice *device), (device))
FORWARD(cuCtxGetFlags, core_context_flags, (unsigned int *flags), (flags))
FORWARD(cuCtxGetApiVersion, core_context_version, (CUcontext context, unsigned int *version), (context, version))
FORWARD(cuCtxSynchronize, core_context_sync, (void), ())
FORWARD(cuCtxEnablePeerAccess, core_context_enable_peer,
        (CUcontext peer, unsigned int flags), (peer, flags))
FORWARD(cuCtxDisablePeerAccess, core_context_disable_peer, (CUcontext peer), (peer))
CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *least, int *greatest) {
    if (!least || !greatest) return CUDA_ERROR_INVALID_VALUE;
    CUcontext current = NULL;
    CUresult result = core_context_current(&current);
    if (result != CUDA_SUCCESS) return result;
    if (!current) return CUDA_ERROR_INVALID_CONTEXT;
    *least = 0;
    *greatest = -1;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
    if (!mode) return CUDA_ERROR_INVALID_VALUE;
    *mode = CU_MODULE_LAZY_LOADING;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamCreate(CUstream *stream, unsigned int flags) {
    return core_stream_create(stream, flags, 0);
}
FORWARD(cuStreamCreateWithPriority, core_stream_create, (CUstream *stream, unsigned int flags, int priority), (stream, flags, priority))
FORWARD(cuStreamDestroy_v2, core_stream_destroy, (CUstream stream), (stream))
FORWARD(cuStreamGetCtx, core_stream_context, (CUstream stream, CUcontext *context), (stream, context))
FORWARD(cuStreamGetDevice, core_stream_device, (CUstream stream, CUdevice *device), (stream, device))
FORWARD(cuStreamGetFlags, core_stream_flags, (CUstream stream, unsigned int *flags), (stream, flags))
FORWARD(cuStreamGetPriority, core_stream_priority, (CUstream stream, int *priority), (stream, priority))
FORWARD(cuStreamQuery, core_stream_query, (CUstream stream), (stream))
FORWARD(cuStreamSynchronize, core_stream_sync, (CUstream stream), (stream))
FORWARD(cuMemGetInfo_v2, virtual_mem_get_info, (size_t *free_bytes, size_t *total_bytes), (free_bytes, total_bytes))
FORWARD(cuMemAlloc_v2, virtual_mem_alloc, (CUdeviceptr *ptr, size_t bytes), (ptr, bytes))
FORWARD(cuMemFree_v2, virtual_mem_free, (CUdeviceptr ptr), (ptr))
FORWARD(cuMemAllocAsync, virtual_mem_alloc_async, (CUdeviceptr *ptr, size_t bytes, CUstream stream), (ptr, bytes, stream))
FORWARD(cuMemFreeAsync, virtual_mem_free_async, (CUdeviceptr ptr, CUstream stream), (ptr, stream))
FORWARD(cuPointerGetAttribute, virtual_pointer_get_attribute,
        (void *data, CUpointer_attribute attribute, CUdeviceptr ptr), (data, attribute, ptr))
FORWARD(cuMemcpyHtoD_v2, virtual_memcpy_h2d, (CUdeviceptr dst, const void *src, size_t bytes), (dst, src, bytes, NULL, 0))
FORWARD(cuMemcpyHtoDAsync_v2, virtual_memcpy_h2d, (CUdeviceptr dst, const void *src, size_t bytes, CUstream stream), (dst, src, bytes, stream, 1))
FORWARD(cuMemcpyDtoH_v2, virtual_memcpy_d2h, (void *dst, CUdeviceptr src, size_t bytes), (dst, src, bytes, NULL, 0))
FORWARD(cuMemcpyDtoHAsync_v2, virtual_memcpy_d2h, (void *dst, CUdeviceptr src, size_t bytes, CUstream stream), (dst, src, bytes, stream, 1))
FORWARD(cuMemcpyDtoD_v2, virtual_memcpy_d2d, (CUdeviceptr dst, CUdeviceptr src, size_t bytes), (dst, src, bytes, NULL, 0))
FORWARD(cuMemcpyDtoDAsync_v2, virtual_memcpy_d2d, (CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream), (dst, src, bytes, stream, 1))
FORWARD(cuMemcpyPeer, virtual_memcpy_peer,
        (CUdeviceptr dst, CUcontext dst_context, CUdeviceptr src, CUcontext src_context, size_t bytes),
        (dst, dst_context, src, src_context, bytes, NULL, 0))
FORWARD(cuMemcpyPeerAsync, virtual_memcpy_peer,
        (CUdeviceptr dst, CUcontext dst_context, CUdeviceptr src, CUcontext src_context,
         size_t bytes, CUstream stream),
        (dst, dst_context, src, src_context, bytes, stream, 1))
FORWARD(cuMemsetD8_v2, virtual_memset_d8, (CUdeviceptr dst, unsigned char value, size_t count), (dst, value, count, NULL, 0))
FORWARD(cuMemsetD8Async, virtual_memset_d8, (CUdeviceptr dst, unsigned char value, size_t count, CUstream stream), (dst, value, count, stream, 1))
FORWARD(cuEventCreate, virtual_event_create, (CUevent *event, unsigned int flags), (event, flags))
FORWARD(cuEventDestroy_v2, virtual_event_destroy, (CUevent event), (event))
FORWARD(cuEventRecord, virtual_event_record, (CUevent event, CUstream stream), (event, stream))
CUresult CUDAAPI cuEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags) {
    if (flags) return CUDA_ERROR_NOT_SUPPORTED; /* external graph event nodes are not modeled */
    return virtual_event_record(event, stream);
}
FORWARD(cuEventQuery, virtual_event_query, (CUevent event), (event))
FORWARD(cuEventSynchronize, virtual_event_sync, (CUevent event), (event))

FORWARD(cuStreamWaitEvent, virtual_stream_wait_event, (CUstream stream, CUevent event, unsigned int flags), (stream, event, flags))
FORWARD(cuStreamBeginCapture_v2, virtual_stream_begin_capture, (CUstream stream, CUstreamCaptureMode mode), (stream, mode))
FORWARD(cuStreamEndCapture, virtual_stream_end_capture, (CUstream stream, CUgraph *graph), (stream, graph))
FORWARD(cuStreamIsCapturing, virtual_stream_is_capturing, (CUstream stream, CUstreamCaptureStatus *status), (stream, status))

FORWARD(cuGraphInstantiateWithFlags, virtual_graph_instantiate_flags,
        (CUgraphExec *exec, CUgraph graph, unsigned long long flags), (exec, graph, flags))
CUresult CUDAAPI cuGraphInstantiateWithParams(CUgraphExec *exec, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS *params) {
    if (!params) return CUDA_ERROR_INVALID_VALUE;
    params->hErrNode_out = NULL;
    params->result_out = CUDA_GRAPH_INSTANTIATE_ERROR;
    if (params->flags || params->hUploadStream) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult result = virtual_graph_instantiate(exec, graph);
    if (result == CUDA_SUCCESS) params->result_out = CUDA_GRAPH_INSTANTIATE_SUCCESS;
    return result;
}
FORWARD(cuGraphGetNodes, virtual_graph_get_nodes,
        (CUgraph graph, CUgraphNode *nodes, size_t *count), (graph, nodes, count))
FORWARD(cuGraphLaunch, virtual_graph_launch, (CUgraphExec exec, CUstream stream), (exec, stream))
FORWARD(cuGraphDestroy, virtual_graph_destroy, (CUgraph graph), (graph))
FORWARD(cuGraphExecDestroy, virtual_graph_exec_destroy, (CUgraphExec exec), (exec))
FORWARD(cuModuleLoadData, virtual_module_load_data, (CUmodule *module, const void *image), (module, image))
FORWARD(cuModuleGetFunction, virtual_module_get_function, (CUfunction *function, CUmodule module, const char *name), (function, module, name))
FORWARD(cuModuleUnload, virtual_module_unload, (CUmodule module), (module))
FORWARD(cuLibraryLoadData, virtual_library_load_data,
        (CUlibrary *library, const void *code, CUjit_option *jit_options, void **jit_values,
         unsigned int jit_count, CUlibraryOption *library_options, void **library_values,
         unsigned int library_count),
        (library, code, jit_options, jit_values, jit_count, library_options, library_values, library_count))
FORWARD(cuLibraryUnload, virtual_library_unload, (CUlibrary library), (library))
FORWARD(cuLibraryGetKernel, virtual_library_get_kernel,
        (CUkernel *kernel, CUlibrary library, const char *name), (kernel, library, name))
FORWARD(cuLibraryGetModule, virtual_library_get_module,
        (CUmodule *module, CUlibrary library), (module, library))
FORWARD(cuKernelGetFunction, virtual_kernel_get_function,
        (CUfunction *function, CUkernel kernel), (function, kernel))
FORWARD(cuLaunchKernel, virtual_launch_kernel,
        (CUfunction function, unsigned int gridX, unsigned int gridY, unsigned int gridZ,
         unsigned int blockX, unsigned int blockY, unsigned int blockZ, unsigned int sharedMem,
         CUstream stream, void **params, void **extra),
        (function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, params, extra))
CUresult CUDAAPI cuGetErrorName(CUresult error, const char **name) {
    if (!name) return CUDA_ERROR_INVALID_VALUE;
    switch (error) {
    case CUDA_SUCCESS: *name = "CUDA_SUCCESS"; break;
    case CUDA_ERROR_INVALID_VALUE: *name = "CUDA_ERROR_INVALID_VALUE"; break;
    case CUDA_ERROR_INVALID_DEVICE: *name = "CUDA_ERROR_INVALID_DEVICE"; break;
    case CUDA_ERROR_INVALID_CONTEXT: *name = "CUDA_ERROR_INVALID_CONTEXT"; break;
    case CUDA_ERROR_INVALID_HANDLE: *name = "CUDA_ERROR_INVALID_HANDLE"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *name = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default: *name = "CUDA_ERROR_UNKNOWN"; break;
    }
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGetErrorString(CUresult error, const char **description) {
    if (!description) return CUDA_ERROR_INVALID_VALUE;
    const char *name;
    cuGetErrorName(error, &name);
    *description = name;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **function, int version, cuuint64_t flags);
CUresult CUDAAPI cuGetProcAddress_v2(const char *symbol, void **function, int version, cuuint64_t flags, CUdriverProcAddressQueryResult *status);
CUresult CUDAAPI cuGetExportTable(const void **table, const CUuuid *uuid) {
    return private_export_table(table, uuid);
}

typedef struct { const char *name; void *function; } Entry;
static void *resolve(const char *symbol, int version) {
    if (!symbol) return NULL;
    if (!strcmp(symbol, "cuCtxCreate"))
        return version >= 13000 ? (void *)cuCtxCreate_v4 : (void *)cuCtxCreate_v2;
    if (!strcmp(symbol, "cuGetProcAddress"))
        return version >= 12000 ? (void *)cuGetProcAddress_v2 : (void *)cuGetProcAddress;
    if (!strcmp(symbol, "cuEventElapsedTime"))
        return version >= 13000 ? (void *)cuEventElapsedTime_v2 : (void *)cuEventElapsedTime;
    if (!strcmp(symbol, "cuStreamBeginCapture"))
        return version >= 10010 ? (void *)cuStreamBeginCapture_v2 : NULL;
    if (!strcmp(symbol, "cuStreamGetCaptureInfo"))
        return version >= 13000 ? (void *)cuStreamGetCaptureInfo_v3 :
               version >= 11030 ? (void *)cuStreamGetCaptureInfo_v2 :
               version >= 10010 ? (void *)cuStreamGetCaptureInfo : NULL;
    if (!strcmp(symbol, "cuGraphInstantiate"))
        return version >= 11040 ? (void *)cuGraphInstantiateWithFlags : NULL;
#define ITEM(name) { #name, (void *)name }
    const Entry entries[] = {
        ITEM(cuInit), ITEM(cuDriverGetVersion), ITEM(cuDeviceGetCount), ITEM(cuDeviceGet),
        ITEM(cuDeviceGetName), ITEM(cuDeviceGetUuid), ITEM(cuDeviceGetUuid_v2),
        ITEM(cuDeviceTotalMem_v2), {"cuDeviceTotalMem", (void *)cuDeviceTotalMem_v2},
        ITEM(cuDeviceGetAttribute), ITEM(cuDeviceCanAccessPeer), ITEM(cuDevicePrimaryCtxRetain),
        ITEM(cuDevicePrimaryCtxGetState), ITEM(cuDevicePrimaryCtxRelease_v2),
        {"cuDevicePrimaryCtxRelease", (void *)cuDevicePrimaryCtxRelease_v2},
        ITEM(cuDevicePrimaryCtxSetFlags_v2),
        {"cuDevicePrimaryCtxSetFlags", (void *)cuDevicePrimaryCtxSetFlags_v2},
        ITEM(cuDevicePrimaryCtxReset_v2),
        {"cuDevicePrimaryCtxReset", (void *)cuDevicePrimaryCtxReset_v2},
        ITEM(cuCtxCreate_v2), ITEM(cuCtxCreate_v4), ITEM(cuCtxDestroy_v2),
        {"cuCtxDestroy", (void *)cuCtxDestroy_v2}, ITEM(cuCtxGetCurrent),
        ITEM(cuCtxSetCurrent), ITEM(cuCtxPushCurrent_v2),
        {"cuCtxPushCurrent", (void *)cuCtxPushCurrent_v2}, ITEM(cuCtxPopCurrent_v2),
        {"cuCtxPopCurrent", (void *)cuCtxPopCurrent_v2}, ITEM(cuCtxGetDevice),
        ITEM(cuCtxGetFlags), ITEM(cuCtxGetApiVersion), ITEM(cuCtxSynchronize),
                ITEM(cuCtxEnablePeerAccess), ITEM(cuCtxDisablePeerAccess),
        ITEM(cuCtxGetStreamPriorityRange), ITEM(cuModuleGetLoadingMode),
        ITEM(cuStreamCreate), ITEM(cuStreamCreateWithPriority), ITEM(cuStreamDestroy_v2),
        {"cuStreamDestroy", (void *)cuStreamDestroy_v2}, ITEM(cuStreamGetCtx),
        ITEM(cuStreamGetDevice), ITEM(cuStreamGetFlags), ITEM(cuStreamGetPriority),
        ITEM(cuStreamQuery), ITEM(cuStreamSynchronize), ITEM(cuStreamWaitEvent),
        ITEM(cuStreamBeginCapture_v2), ITEM(cuStreamEndCapture), ITEM(cuStreamIsCapturing),
                ITEM(cuStreamGetCaptureInfo_v2), ITEM(cuStreamGetCaptureInfo_v3),
        ITEM(cuMemGetInfo_v2), {"cuMemGetInfo", (void *)cuMemGetInfo_v2},
        ITEM(cuMemAlloc_v2), {"cuMemAlloc", (void *)cuMemAlloc_v2},
        ITEM(cuMemFree_v2), {"cuMemFree", (void *)cuMemFree_v2},
        ITEM(cuMemAllocAsync), ITEM(cuMemFreeAsync), ITEM(cuPointerGetAttribute),
        ITEM(cuMemcpyHtoD_v2), {"cuMemcpyHtoD", (void *)cuMemcpyHtoD_v2},
        ITEM(cuMemcpyHtoDAsync_v2), {"cuMemcpyHtoDAsync", (void *)cuMemcpyHtoDAsync_v2},
        ITEM(cuMemcpyDtoH_v2), {"cuMemcpyDtoH", (void *)cuMemcpyDtoH_v2},
        ITEM(cuMemcpyDtoHAsync_v2), {"cuMemcpyDtoHAsync", (void *)cuMemcpyDtoHAsync_v2},
        ITEM(cuMemcpyDtoD_v2), {"cuMemcpyDtoD", (void *)cuMemcpyDtoD_v2},
        ITEM(cuMemcpyDtoDAsync_v2), {"cuMemcpyDtoDAsync", (void *)cuMemcpyDtoDAsync_v2},
                ITEM(cuMemcpyPeer), ITEM(cuMemcpyPeerAsync),
        ITEM(cuMemsetD8_v2), {"cuMemsetD8", (void *)cuMemsetD8_v2}, ITEM(cuMemsetD8Async),
        ITEM(cuEventCreate), ITEM(cuEventDestroy_v2),
        {"cuEventDestroy", (void *)cuEventDestroy_v2}, ITEM(cuEventRecord),
                ITEM(cuEventRecordWithFlags),
        ITEM(cuEventQuery), ITEM(cuEventSynchronize), ITEM(cuEventElapsedTime_v2),
        ITEM(cuGraphInstantiateWithFlags), ITEM(cuGraphInstantiateWithParams),
                ITEM(cuGraphGetNodes), ITEM(cuGraphLaunch), ITEM(cuGraphDestroy),
        ITEM(cuGraphExecDestroy), ITEM(cuModuleLoadData), ITEM(cuModuleGetFunction),
        ITEM(cuModuleUnload), ITEM(cuLibraryLoadData), ITEM(cuLibraryUnload),
                ITEM(cuLibraryGetKernel), ITEM(cuLibraryGetModule), ITEM(cuKernelGetFunction),
                ITEM(cuLaunchKernel), ITEM(cuGetErrorName),
        ITEM(cuGetErrorString), ITEM(cuGetExportTable), ITEM(cuGetProcAddress_v2)
    };
#undef ITEM
    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); ++i)
        if (!strcmp(symbol, entries[i].name)) return entries[i].function;
    return NULL;
}
CUresult CUDAAPI cuGetProcAddress_v2(const char *symbol, void **function, int version,
                                      cuuint64_t flags, CUdriverProcAddressQueryResult *status) {
    (void)flags;
    if (!function) return CUDA_ERROR_INVALID_VALUE;
    *function = resolve(symbol, version);
    if (status) *status = *function ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    if (getenv("FAKE_CUDA_TRACE")) fprintf(stderr, "fake-cuda: resolve %s: %s\n",
        symbol ? symbol : "(null)", *function ? "yes" : "no");
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **function, int version, cuuint64_t flags) {
    return cuGetProcAddress_v2(symbol, function, version, flags, NULL);
}
