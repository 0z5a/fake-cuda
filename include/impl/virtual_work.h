#ifndef FAKE_CUDA_IMPL_VIRTUAL_WORK_H
#define FAKE_CUDA_IMPL_VIRTUAL_WORK_H

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
/* cuStreamQuery must forward here rather than to the blocking core_stream_sync. */
CUresult core_stream_query(CUstream stream);
CUresult virtual_mem_get_info(size_t *free_bytes, size_t *total_bytes);
CUresult virtual_mem_alloc(CUdeviceptr *ptr, size_t bytes);
CUresult virtual_mem_free(CUdeviceptr ptr);
CUresult virtual_mem_alloc_async(CUdeviceptr *ptr, size_t bytes, CUstream stream);
CUresult virtual_mem_free_async(CUdeviceptr ptr, CUstream stream);
CUresult virtual_pointer_get_attribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult virtual_memcpy_h2d(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream, int async);
CUresult virtual_memcpy_d2h(void *dst, CUdeviceptr src, size_t bytes, CUstream stream, int async);
CUresult virtual_memcpy_d2d(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream, int async);
CUresult virtual_memcpy_peer(CUdeviceptr dst, CUcontext dst_context, CUdeviceptr src,
                             CUcontext src_context, size_t bytes, CUstream stream, int async);
CUresult virtual_memset_d8(CUdeviceptr dst, unsigned char value, size_t count, CUstream stream, int async);
CUresult virtual_event_create(CUevent *event, unsigned int flags);
CUresult virtual_event_destroy(CUevent event);
CUresult virtual_event_record(CUevent event, CUstream stream);
CUresult virtual_event_query(CUevent event);
CUresult virtual_event_sync(CUevent event);
CUresult virtual_event_elapsed(float *milliseconds, CUevent start, CUevent end);
CUresult virtual_stream_wait_event(CUstream stream, CUevent event, unsigned int flags);
CUresult virtual_stream_begin_capture(CUstream stream, CUstreamCaptureMode mode);
CUresult virtual_stream_end_capture(CUstream stream, CUgraph *graph);
CUresult virtual_stream_is_capturing(CUstream stream, CUstreamCaptureStatus *status);
CUresult virtual_thread_exchange_capture_mode(CUstreamCaptureMode *mode);
CUresult virtual_stream_get_capture_info(CUstream stream, CUstreamCaptureStatus *status,
                                         cuuint64_t *id, CUgraph *graph,
                                         const CUgraphNode **dependencies,
                                         const CUgraphEdgeData **edges, size_t *count);
CUresult virtual_graph_get_nodes(CUgraph graph, CUgraphNode *nodes, size_t *count);
CUresult virtual_graph_instantiate_flags(CUgraphExec *exec, CUgraph graph, unsigned long long flags);
CUresult virtual_graph_instantiate(CUgraphExec *exec, CUgraph graph);
CUresult virtual_graph_launch(CUgraphExec exec, CUstream stream);
CUresult virtual_graph_destroy(CUgraph graph);
CUresult virtual_graph_exec_destroy(CUgraphExec exec);
CUresult virtual_module_load_data(CUmodule *module, const void *image);
CUresult virtual_module_get_function(CUfunction *function, CUmodule module, const char *name);
CUresult virtual_module_unload(CUmodule module);
CUresult virtual_library_load_data(CUlibrary *library, const void *code,
                                   CUjit_option *jit_options, void **jit_values,
                                   unsigned int jit_count, CUlibraryOption *library_options,
                                   void **library_values, unsigned int library_count);
CUresult virtual_library_unload(CUlibrary library);
CUresult virtual_library_get_kernel(CUkernel *kernel, CUlibrary library, const char *name);
CUresult virtual_library_get_module(CUmodule *module, CUlibrary library);
CUresult virtual_kernel_get_function(CUfunction *function, CUkernel kernel);
CUresult virtual_launch_kernel(CUfunction function, unsigned int gridX, unsigned int gridY,
                               unsigned int gridZ, unsigned int blockX, unsigned int blockY,
                               unsigned int blockZ, unsigned int sharedMem, CUstream stream,
                               void **params, void **extra);
#ifdef __cplusplus
}
#endif
#endif
