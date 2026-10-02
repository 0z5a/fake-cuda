#include "impl/virtual_work.h"
#include "impl/core.h"
#include "impl/scheduler.h"
#include "impl/device.h"

#include <chrono>

#include <mutex>
#include <new>
#include <thread>


using fake_cuda::VirtualClock;
using namespace fake_cuda::detail;

namespace {

CUresult copy_work(CUstream stream, Kind kind, size_t bytes, int async,
                   CUdeviceptr first, CUdeviceptr second, const void *host) {
    Time finish{};
    CUresult r = scheduler().in_stream(stream, [&](Scheduler &s, CUcontext ctx, Key key) {
        s.reap();
        if (bytes && !host && (kind == Kind::h2d || kind == Kind::d2h)) return CUDA_ERROR_INVALID_VALUE;
        bool is_capture = s.graph.capturing(key);
        Node node{kind, bytes, nullptr, first, second};
        const bool d2d = kind == Kind::compute && second;
        if (d2d && s.memory.copy_node(ctx, first, second, bytes, node) != CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        const Time earliest = d2d
            ? s.queue.earliest_peer(key, s.queue.device_for(node.second_context), s.queue.device_for(node.first_context))
            : s.queue.earliest(key, kind);
        auto valid = [&](CUdeviceptr ptr) {
            return is_capture ? s.memory.check_bounds(ctx, ptr, bytes, d2d) :
                                s.memory.check_pointer(ctx, ptr, bytes, earliest, d2d);
        };
        if ((first && valid(first) != CUDA_SUCCESS) || (second && valid(second) != CUDA_SUCCESS))
            return CUDA_ERROR_INVALID_VALUE;
        if (!async && is_capture) return s.graph.invalidate(key);
        OpPtr op;
        CUresult result = s.enqueue(key, node, &op);
        if (op) finish = op->end;
        return result;
    });
    if (r == CUDA_SUCCESS && !async) std::this_thread::sleep_until(finish);
    return r;
}
} // namespace

namespace fake_cuda {
CUresult virtual_synchronize_stream(CUcontext ctx, CUstream stream, bool query) {
    Time end{};
    {
        Scheduler &s = scheduler();
        std::scoped_lock lock(s.mutex);
        if (s.retired.count(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
        if (s.graph.capturing(key_for(ctx, stream))) return s.graph.invalidate(key_for(ctx, stream));
        s.reap();
        end = s.queue.stream_end(key_for(ctx, stream));
    }
    if (query) return VirtualClock::now() >= end ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    std::this_thread::sleep_until(end);
    return CUDA_SUCCESS;
}
CUresult virtual_synchronize_context(CUcontext ctx) {
    Time end{};
    {
        Scheduler &s = scheduler();
        std::scoped_lock lock(s.mutex);
        if (s.retired.count(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
        if (s.graph.active(ctx)) return s.graph.invalidate_context(ctx);
        end = s.queue.context_end(ctx);
        s.reap();
    }
    std::this_thread::sleep_until(end);
    return CUDA_SUCCESS;
}
CUresult virtual_begin_retire_context(CUcontext ctx) {
    return scheduler().begin_retire_context(ctx);
}
CUresult virtual_enable_peer(CUcontext ctx, CUcontext peer) {
    try {
        Scheduler &s = scheduler();
        std::scoped_lock lock(s.mutex);
        if (s.retired.contains(ctx) || s.retired.contains(peer)) return CUDA_ERROR_INVALID_CONTEXT;
        return s.memory.enable_peer(ctx, peer);
    } catch (const std::bad_alloc &) { return CUDA_ERROR_OUT_OF_MEMORY; }
}
CUresult virtual_disable_peer(CUcontext ctx, CUcontext peer) {
    Scheduler &s = scheduler();
    std::scoped_lock lock(s.mutex);
    if (s.retired.contains(ctx) || s.retired.contains(peer)) return CUDA_ERROR_INVALID_CONTEXT;
    return s.memory.disable_peer(ctx, peer);
}
void virtual_drain_context(CUcontext ctx) {
    Time end{};
    {
        Scheduler &s = scheduler();
        std::scoped_lock lock(s.mutex);
        end = s.queue.context_end(ctx);
    }
    std::this_thread::sleep_until(end);
}
void virtual_retire_stream(CUcontext ctx, CUstream stream) {
    Scheduler &s = scheduler();
    std::scoped_lock lock(s.mutex);
    s.retire_stream(ctx, stream);
}
void virtual_retire_context(CUcontext ctx) {
    Scheduler &s = scheduler();
    std::scoped_lock lock(s.mutex);
    s.retire_context(ctx);
}
} // namespace fake_cuda

extern "C" {
CUresult virtual_register_context(CUcontext context, CUdevice device) {
    try {
        Scheduler &s = scheduler();
        std::scoped_lock lock(s.mutex);
        s.queue.register_context(context, device);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc &) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}
CUresult virtual_mem_get_info(size_t *free_bytes, size_t *total_bytes) {
    if (!free_bytes || !total_bytes) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        *total_bytes = fake_cuda::virtual_core_device(s.queue.device_for(ctx))->profile().memory_bytes;
        *free_bytes = s.memory.free_bytes(s.queue.device_for(ctx));
        return CUDA_SUCCESS;
    });
}
CUresult virtual_mem_alloc(CUdeviceptr *ptr, size_t bytes) {
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        if (CUresult status = s.graph.check_unsafe_call(); status != CUDA_SUCCESS) return status;
        return s.memory.allocate(ctx, s.queue.device_for(ctx), ptr, bytes, VirtualClock::now());
    });
}
CUresult virtual_mem_free(CUdeviceptr ptr) {
    Time finish{};
    CUresult r = scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        if (CUresult status = s.graph.check_unsafe_call(); status != CUDA_SUCCESS) return status;
        finish = s.queue.context_end(ctx);
        return s.memory.free(ctx, ptr, finish);
    });
    if (r != CUDA_SUCCESS) return r;
    std::this_thread::sleep_until(finish);
    std::scoped_lock lock(scheduler().mutex);
    scheduler().reap();
    return CUDA_SUCCESS;
}
CUresult virtual_mem_alloc_async(CUdeviceptr *ptr, size_t bytes, CUstream stream) {
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext ctx, Key key) {
        if (s.graph.capturing(key)) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        if (!ptr || !bytes) return CUDA_ERROR_INVALID_VALUE;
        CUdevice device = s.queue.device_for(ctx);
        if (!s.memory.can_allocate(device, bytes)) return CUDA_ERROR_OUT_OF_MEMORY;
        OpPtr op = s.queue.schedule(key, Kind::marker);
        return s.memory.allocate(ctx, device, ptr, bytes, op->end);
    });
}
CUresult virtual_mem_free_async(CUdeviceptr ptr, CUstream stream) {
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext ctx, Key key) {
        if (s.graph.capturing(key)) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        if (s.memory.can_free_async(ctx, ptr, s.queue.earliest(key, Kind::marker)) != CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        Time finish = s.queue.schedule(key, Kind::marker)->end;
        s.memory.free_async(ptr, finish);
        return CUDA_SUCCESS;
    });
}
CUresult virtual_pointer_get_attribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    if (!data) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        return s.memory.attribute(ctx, data, attribute, ptr);
    });
}
CUresult virtual_memcpy_h2d(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream, int async) {
    if (!dst) return CUDA_ERROR_INVALID_VALUE;
    return copy_work(stream, Kind::h2d, bytes, async, dst, 0, src);
}
CUresult virtual_memcpy_d2h(void *dst, CUdeviceptr src, size_t bytes, CUstream stream, int async) {
    if (!src) return CUDA_ERROR_INVALID_VALUE;
    return copy_work(stream, Kind::d2h, bytes, async, src, 0, dst);
}
CUresult virtual_memcpy_d2d(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream, int async) {
    if (!dst || !src) return CUDA_ERROR_INVALID_VALUE;
    return copy_work(stream, Kind::compute, bytes, async, dst, src, nullptr);
}
CUresult virtual_memcpy_peer(CUdeviceptr dst, CUcontext dst_context, CUdeviceptr src,
                             CUcontext src_context, size_t bytes, CUstream stream, int async) {
    if (!dst || !src) return CUDA_ERROR_INVALID_VALUE;
    // Resolve opaque handles before taking the scheduler lock: core resolution
    // acquires the Registry lock, whereas context retirement locks in reverse.
    CUdevice dst_device = -1, src_device = -1;
    CUresult result = virtual_core_context_device(dst_context, &dst_device);
    if (result != CUDA_SUCCESS) return result;
    result = virtual_core_context_device(src_context, &src_device);
    if (result != CUDA_SUCCESS) return result;

    Time finish{};
    result = scheduler().in_stream(stream, [&](Scheduler &s, CUcontext ctx, Key key) {
        if (ctx != dst_context || s.retired.contains(src_context)) return CUDA_ERROR_INVALID_CONTEXT;
        if (s.graph.capturing(key)) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        Time at = s.queue.earliest_peer(key, src_device, dst_device);
        if (s.memory.check_pointer(dst_context, dst, bytes, at) != CUDA_SUCCESS ||
            s.memory.check_pointer(src_context, src, bytes, at) != CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        OpPtr op = s.queue.schedule_peer(key, src_device, dst_device, bytes);
        s.memory.include_use(src, op->end);
        s.memory.include_use(dst, op->end);
        if (src_context != dst_context) s.queue.include_context(src_context, op->end);
        finish = op->end;
        return CUDA_SUCCESS;
    });
    if (result == CUDA_SUCCESS && !async) std::this_thread::sleep_until(finish);
    return result;
}
CUresult virtual_memset_d8(CUdeviceptr dst, unsigned char, size_t count, CUstream stream, int async) {
    if (!dst) return CUDA_ERROR_INVALID_VALUE;
    return copy_work(stream, Kind::compute, count, async, dst, 0, nullptr);
}
CUresult virtual_event_create(CUevent *event, unsigned int flags) {
    if (!event || (flags & ~(CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS)))
        return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        CUevent id = s.opaque<CUevent>();
        s.events.emplace(id, Event{ctx, flags, {}});
        *event = id;
        return CUDA_SUCCESS;
    });
}
CUresult virtual_event_destroy(CUevent event) {
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        auto it = s.events.find(event);
        if (it == s.events.end() || it->second.context != ctx) return CUDA_ERROR_INVALID_HANDLE;
        s.events.erase(it);
        return CUDA_SUCCESS;
    });
}
CUresult virtual_event_record(CUevent event, CUstream stream) {
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext ctx, Key key) {
        auto it = s.events.find(event);
        if (it == s.events.end() || it->second.context != ctx) return CUDA_ERROR_INVALID_HANDLE;
        return s.enqueue(key, {Kind::record, 0, event});
    });
}
CUresult virtual_event_query(CUevent event) {
    Time finish{};
    CUresult r = scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        auto it = s.events.find(event);
        if (it == s.events.end() || it->second.context != ctx) return CUDA_ERROR_INVALID_HANDLE;
        if (it->second.record) finish = it->second.record->end;
        return CUDA_SUCCESS;
    });
    return r == CUDA_SUCCESS && VirtualClock::now() < finish ? CUDA_ERROR_NOT_READY : r;
}
CUresult virtual_event_sync(CUevent event) {
    Time finish{};
    CUresult r = scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        auto it = s.events.find(event);
        if (it == s.events.end() || it->second.context != ctx) return CUDA_ERROR_INVALID_HANDLE;
        if (it->second.record) finish = it->second.record->end;
        return CUDA_SUCCESS;
    });
    if (r == CUDA_SUCCESS) std::this_thread::sleep_until(finish);
    return r;
}
CUresult virtual_event_elapsed(float *milliseconds, CUevent start, CUevent end) {
    if (!milliseconds) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        auto a = s.events.find(start), b = s.events.find(end);
        if (a == s.events.end() || b == s.events.end() || a->second.context != ctx || b->second.context != ctx)
            return CUDA_ERROR_INVALID_HANDLE;
        if ((a->second.flags | b->second.flags) & CU_EVENT_DISABLE_TIMING) return CUDA_ERROR_INVALID_HANDLE;
        if (!a->second.record || !b->second.record ||
            VirtualClock::now() < a->second.record->end || VirtualClock::now() < b->second.record->end)
            return CUDA_ERROR_NOT_READY;
        *milliseconds = std::chrono::duration<float, std::milli>(b->second.record->end - a->second.record->end).count();
        return CUDA_SUCCESS;
    });
}
CUresult virtual_stream_wait_event(CUstream stream, CUevent event, unsigned int flags) {
    if (flags) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext, Key key) {
        auto it = s.events.find(event);
        if (it == s.events.end()) return CUDA_ERROR_INVALID_HANDLE;
        return s.enqueue(key, {Kind::wait, 0, event});
    });
}
CUresult virtual_library_load_data(CUlibrary *library, const void *code,
                                   CUjit_option *jit_options, void **jit_values,
                                   unsigned int jit_count, CUlibraryOption *library_options,
                                   void **library_values, unsigned int library_count) {
    if (!library || !code || (jit_count && (!jit_options || !jit_values)) ||
        (library_count && (!library_options || !library_values))) return CUDA_ERROR_INVALID_VALUE;
    // Preserving the caller-owned code is an optional lifetime hint. No code
    // image is read by this simulator; JIT and other library options need real
    // compilation or a host function/data table and cannot be honored.
    if (jit_count) return CUDA_ERROR_NOT_SUPPORTED;
    for (unsigned i = 0; i < library_count; ++i)
        if (library_options[i] != CU_LIBRARY_BINARY_IS_PRESERVED) return CUDA_ERROR_NOT_SUPPORTED;
    try {
        Scheduler &s = scheduler();
        std::scoped_lock lock(s.mutex);
        CUlibrary id = s.opaque<CUlibrary>();
        s.libraries.emplace(id, Scheduler::Library{});
        *library = id;
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc &) { return CUDA_ERROR_OUT_OF_MEMORY; }
}
CUresult virtual_library_unload(CUlibrary library) {
    Scheduler &s = scheduler();
    std::scoped_lock lock(s.mutex);
    auto lib = s.libraries.find(library);
    if (lib == s.libraries.end()) return CUDA_ERROR_INVALID_HANDLE;
    for (const auto &[name, kernel] : lib->second.kernels) {
        (void)name;
        s.kernels.erase(kernel);
        for (auto it = s.library_functions.begin(); it != s.library_functions.end();)
            if (it->first.first == kernel) it = s.library_functions.erase(it); else ++it;
    }
    for (const auto &[ctx, module] : lib->second.modules) {
        (void)ctx;
        for (auto it = s.functions.begin(); it != s.functions.end();)
            if (it->second.module == module) it = s.functions.erase(it); else ++it;
        s.modules.erase(module);
    }
    s.libraries.erase(lib);
    return CUDA_SUCCESS;
}
CUresult virtual_library_get_kernel(CUkernel *kernel, CUlibrary library, const char *name) {
    if (!kernel || !name || !*name) return CUDA_ERROR_INVALID_VALUE;
    try {
        Scheduler &s = scheduler();
        std::scoped_lock lock(s.mutex);
        auto lib = s.libraries.find(library);
        if (lib == s.libraries.end()) return CUDA_ERROR_INVALID_HANDLE;
        auto found = lib->second.kernels.find(name);
        if (found != lib->second.kernels.end()) { *kernel = found->second; return CUDA_SUCCESS; }
        CUkernel id = s.opaque<CUkernel>();
        lib->second.kernels.emplace(name, id);
        s.kernels.emplace(id, fake_cuda::KernelRecord{library,
            std::make_shared<const fake_cuda::KernelIdentity>(
                fake_cuda::KernelIdentity{reinterpret_cast<std::uintptr_t>(library), name})});
        *kernel = id;
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc &) { return CUDA_ERROR_OUT_OF_MEMORY; }
}
CUresult virtual_library_get_module(CUmodule *module, CUlibrary library) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        auto lib = s.libraries.find(library);
        if (lib == s.libraries.end()) return CUDA_ERROR_INVALID_HANDLE;
        if (auto found = lib->second.modules.find(ctx); found != lib->second.modules.end()) {
            *module = found->second;
            return CUDA_SUCCESS;
        }
        CUmodule id = s.opaque<CUmodule>();
        s.modules.emplace(id, fake_cuda::ModuleRecord{ctx, reinterpret_cast<std::uintptr_t>(library)});
        lib->second.modules.emplace(ctx, id);
        *module = id;
        return CUDA_SUCCESS;
    });
}
CUresult virtual_kernel_get_function(CUfunction *function, CUkernel kernel) {
    if (!function) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        auto lib_id = s.kernels.find(kernel);
        if (lib_id == s.kernels.end()) return CUDA_ERROR_INVALID_HANDLE;
        auto lib = s.libraries.find(lib_id->second.library);
        if (lib == s.libraries.end()) return CUDA_ERROR_INVALID_HANDLE;
        auto key = std::make_pair(kernel, ctx);
        if (auto found = s.library_functions.find(key); found != s.library_functions.end()) {
            *function = found->second;
            return CUDA_SUCCESS;
        }
        CUmodule module;
        if (auto found = lib->second.modules.find(ctx); found != lib->second.modules.end()) module = found->second;
        else {
            module = s.opaque<CUmodule>();
            s.modules.emplace(module, fake_cuda::ModuleRecord{ctx, reinterpret_cast<std::uintptr_t>(lib_id->second.library)});
            lib->second.modules.emplace(ctx, module);
        }
        CUfunction id = s.opaque<CUfunction>();
        s.functions.emplace(id, fake_cuda::FunctionRecord{module, lib_id->second.identity});
        s.library_functions.emplace(key, id);
        *function = id;
        return CUDA_SUCCESS;
    });
}
CUresult virtual_module_load_data(CUmodule *module, const void *image) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        CUmodule id = s.opaque<CUmodule>();
        s.modules.emplace(id, fake_cuda::ModuleRecord{ctx, reinterpret_cast<std::uintptr_t>(id)});
        *module = id;
        return CUDA_SUCCESS;
    });
}
CUresult virtual_module_get_function(CUfunction *function, CUmodule module, const char *name) {
    if (!function || !name || !*name) return CUDA_ERROR_INVALID_VALUE;
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        auto it = s.modules.find(module);
        if (it == s.modules.end() || it->second.context != ctx) return CUDA_ERROR_INVALID_HANDLE;
        for (const auto &[handle, record] : s.functions)
            if (record.module == module && record.kernel->symbol == name) {
                *function = handle;
                return CUDA_SUCCESS;
            }
        CUfunction id = s.opaque<CUfunction>();
        s.functions.emplace(id, fake_cuda::FunctionRecord{module,
            std::make_shared<const fake_cuda::KernelIdentity>(
                fake_cuda::KernelIdentity{it->second.load_id, name})});
        *function = id;
        return CUDA_SUCCESS;
    });
}
CUresult virtual_module_unload(CUmodule module) {
    return scheduler().in_context([&](Scheduler &s, CUcontext ctx) {
        auto it = s.modules.find(module);
        if (it == s.modules.end() || it->second.context != ctx) return CUDA_ERROR_INVALID_HANDLE;
        for (auto f = s.functions.begin(); f != s.functions.end();)
            if (f->second.module == module) f = s.functions.erase(f); else ++f;
        for (auto &[handle, library] : s.libraries) {
            (void)handle;
            if (auto found = library.modules.find(ctx);
                found != library.modules.end() && found->second == module) library.modules.erase(found);
        }
        for (auto f = s.library_functions.begin(); f != s.library_functions.end();)
            if (f->first.second == ctx && !s.functions.contains(f->second))
                f = s.library_functions.erase(f);
            else ++f;
        s.modules.erase(it);
        return CUDA_SUCCESS;
    });
}
CUresult virtual_launch_kernel(CUfunction function, unsigned int gridX, unsigned int gridY,
                               unsigned int gridZ, unsigned int blockX, unsigned int blockY,
                               unsigned int blockZ, unsigned int shared_bytes, CUstream stream,
                               void **params, void **extra) {
    if (params && extra) return CUDA_ERROR_INVALID_VALUE;
    int priority = 0;
    CUresult status = core_stream_priority(stream, &priority);
    if (status != CUDA_SUCCESS) return status;
    return scheduler().in_stream(stream, [&](Scheduler &s, CUcontext ctx, Key key) {
        std::shared_ptr<const fake_cuda::KernelIdentity> identity;
        auto it = s.functions.find(function);
        if (it == s.functions.end()) {
            // CUDA 13 also accepts context-independent library kernel handles.
            auto kernel = s.kernels.find(reinterpret_cast<CUkernel>(function));
            if (kernel == s.kernels.end() || !s.libraries.contains(kernel->second.library))
                return CUDA_ERROR_INVALID_HANDLE;
            identity = kernel->second.identity;
        } else {
            auto module = s.modules.find(it->second.module);
            if (module == s.modules.end() || module->second.context != ctx) return CUDA_ERROR_INVALID_HANDLE;
            identity = it->second.kernel;
        }
        auto launch = std::make_shared<fake_cuda::KernelLaunch>(fake_cuda::KernelLaunch{
            identity, {gridX, gridY, gridZ}, {blockX, blockY, blockZ}, shared_bytes,
            fake_cuda::ParameterEncoding::unknown_layout, {}});
        const auto &profile = fake_cuda::virtual_core_device(s.queue.device_for(ctx))->profile();
        CUresult result = launch->validate(profile);
        if (result != CUDA_SUCCESS) return result;
        if (extra && (result = launch->snapshot(extra)) != CUDA_SUCCESS) return result;
        // kernelParams needs the image's parameter layout. Never retain void** or guess sizes.
        return s.enqueue(key, {Kind::kernel, 0, nullptr, 0, 0, priority, std::move(launch)});
    });
}
} // extern "C"
