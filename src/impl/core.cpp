#include "impl/core.h"
#include "impl/private.h"
#include "impl/virtual_work_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>


namespace fake_cuda {
int Device::count() noexcept { return device_configuration().count; }
bool Device::valid(CUdevice ordinal) noexcept { return ordinal >= 0 && ordinal < count(); }

CUresult Device::name(char *out, int size) const {
    if (!out || size <= 0) return CUDA_ERROR_INVALID_VALUE;
    std::snprintf(out, static_cast<size_t>(size), "%s", profile_.name.c_str());
    return CUDA_SUCCESS;
}
CUresult Device::uuid(CUuuid *out) const {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    const unsigned char id[16] = {'F','A','K','E','-','C','U','D','A',0,0,0,0,0,0,1};
    std::memcpy(out->bytes, id, sizeof(id));
    const auto id_ordinal = static_cast<unsigned int>(ordinal_ + 1);
    for (unsigned int i = 0; i < 4; ++i)
        out->bytes[15 - i] = static_cast<char>(id_ordinal >> (i * 8));
    return CUDA_SUCCESS;
}
CUresult Device::memory(size_t *out) const {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    *out = profile_.memory_bytes;
    return CUDA_SUCCESS;
}
CUresult Device::attribute(int *out, CUdevice_attribute attribute) const {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (attribute < 1) return CUDA_ERROR_INVALID_VALUE;
    const auto found = profile_.attributes.find(attribute);
    if (found == profile_.attributes.end())
        return attribute >= CU_DEVICE_ATTRIBUTE_MAX ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_NOT_SUPPORTED;
    *out = found->second;
    return CUDA_SUCCESS;
}

Context::Context(std::shared_ptr<Device> device, bool primary, unsigned int flags,
                 std::uintptr_t handle_id) noexcept
    : handle_id_(handle_id), device_(device), primary_(primary), flags_(flags) {}
CUdevice Context::device_ordinal() const noexcept { return device_.lock()->ordinal(); }

Stream::Stream(std::shared_ptr<Context> context, unsigned int flags, int priority,
               std::uintptr_t handle_id) noexcept
    : handle_id_(handle_id), context_(std::move(context)), flags_(flags), priority_(priority) {}

thread_local Registry::ThreadState Registry::thread_;
Registry::Registry() : devices_(Device::count()) {
    for (int ordinal = 0; ordinal < Device::count(); ++ordinal)
        devices_[ordinal] = std::make_shared<Device>(ordinal);
}

std::shared_ptr<Context> Registry::live_locked(CUcontext context) const {
    auto it = contexts_.find(context);
    return it == contexts_.end() || it->second->retiring() ? nullptr : it->second;
}
std::shared_ptr<Context> Registry::selected_locked() const {
    return live_locked(thread_.current ? thread_.current->handle() : nullptr);
}
std::shared_ptr<Context> Registry::stream_owner_locked(CUstream stream) const {
    if (!stream || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD)
        return selected_locked();
    auto it = streams_.find(stream);
    return it == streams_.end() ? nullptr : live_locked(it->second->context()->handle());
}
CUresult Registry::ensure_primary_locked(CUdevice device, std::shared_ptr<Context> *out) {
    auto& primary = devices_[device]->primary_;
    if (!primary) {
        auto ctx = std::make_shared<Context>(devices_[device], true, 0, ++next_handle_);
        contexts_.emplace(ctx->handle(), ctx);
        CUresult result;
        try { result = virtual_register_context(ctx->handle(), device); }
        catch (...) {
            contexts_.erase(ctx->handle());
            throw;
        }
        if (result != CUDA_SUCCESS) {
            contexts_.erase(ctx->handle());
            return result;
        }
        primary = std::move(ctx);
    }
    *out = primary;
    return CUDA_SUCCESS;
}
void Registry::retire_locked(const std::shared_ptr<Context>& context) {
    context->begin_retirement();
    virtual_retire_context(context->handle());
    for (auto it = streams_.begin(); it != streams_.end();) {
        if (it->second->context() == context) it = streams_.erase(it);
        else ++it;
    }
    contexts_.erase(context->handle());
    peers_.erase(context->handle());
    for (auto it = peers_.begin(); it != peers_.end();) {
        it->second.erase(context->handle());
        if (it->second.empty()) it = peers_.erase(it);
        else ++it;
    }
    if (thread_.current == context) thread_.current.reset();
    for (auto& item : thread_.stack) if (item == context) item.reset();
}

CUresult Registry::primary_get(CUcontext *out, CUdevice device, bool retain) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    try {
        std::lock_guard lock(mutex_);
        std::shared_ptr<Context> ctx;
        CUresult result = ensure_primary_locked(device, &ctx);
        if (result != CUDA_SUCCESS) return result;
        if (ctx->retiring()) return CUDA_ERROR_INVALID_CONTEXT;
        if (retain) ctx->retain();
        *out = ctx->handle();
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) { return CUDA_ERROR_OUT_OF_MEMORY; }
}
CUresult Registry::primary_state(CUdevice device, unsigned int *flags, int *active) {
    if (!flags || !active) return CUDA_ERROR_INVALID_VALUE;
    if (!Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    std::lock_guard lock(mutex_);
    auto ctx = devices_[device]->primary_;
    *flags = ctx ? ctx->flags() : 0;
    *active = ctx && ctx->retains() ? 1 : 0;
    return CUDA_SUCCESS;
}
CUresult Registry::primary_release(CUdevice device) {
    if (!Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    CUcontext destroyed = nullptr;
    {
        std::lock_guard lock(mutex_);
        auto ctx = devices_[device]->primary_;
        if (!ctx || !ctx->retains() || ctx->retiring()) return CUDA_ERROR_INVALID_CONTEXT;
        if (ctx->retains() > 1) {
            ctx->release();
            return CUDA_SUCCESS;
        }
        CUresult result = virtual_begin_retire_context(ctx->handle());
        if (result != CUDA_SUCCESS) return result;
        ctx->begin_retirement();
        ctx->release();
        destroyed = ctx->handle();
    }
    virtual_drain_context(destroyed);
    {
        std::lock_guard lock(mutex_);
        retire_locked(devices_[device]->primary_);
        devices_[device]->primary_.reset();
    }
    private_context_destroyed(destroyed);
    return CUDA_SUCCESS;
}
CUresult Registry::primary_flags(CUdevice device, unsigned int flags) {
    if (!Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    try {
        std::lock_guard lock(mutex_);
        std::shared_ptr<Context> ctx;
        CUresult result = ensure_primary_locked(device, &ctx);
        if (result != CUDA_SUCCESS) return result;
        if (ctx->retiring()) return CUDA_ERROR_INVALID_CONTEXT;
        if (ctx->retains()) return CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE;
        ctx->set_flags(flags);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) { return CUDA_ERROR_OUT_OF_MEMORY; }
}
CUresult Registry::primary_reset(CUdevice device) {
    if (!Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    CUcontext destroyed = nullptr;
    {
        std::lock_guard lock(mutex_);
        auto ctx = devices_[device]->primary_;
        if (!ctx) return CUDA_SUCCESS;
        if (ctx->retiring()) return CUDA_ERROR_INVALID_CONTEXT;
        CUresult result = virtual_begin_retire_context(ctx->handle());
        if (result != CUDA_SUCCESS) return result;
        ctx->begin_retirement();
        destroyed = ctx->handle();
    }
    virtual_drain_context(destroyed);
    {
        std::lock_guard lock(mutex_);
        retire_locked(devices_[device]->primary_);
        devices_[device]->primary_.reset();
    }
    private_context_destroyed(destroyed);
    return CUDA_SUCCESS;
}
CUresult Registry::context_create(CUcontext *out, unsigned int flags, CUdevice device) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    try {
        std::lock_guard lock(mutex_);
        thread_.stack.reserve(thread_.stack.size() + 1);
        auto ctx = std::make_shared<Context>(devices_[device], false, flags, ++next_handle_);
        contexts_.emplace(ctx->handle(), ctx);
        CUresult result;
        try { result = virtual_register_context(ctx->handle(), device); }
        catch (...) {
            contexts_.erase(ctx->handle());
            throw;
        }
        if (result != CUDA_SUCCESS) {
            contexts_.erase(ctx->handle());
            return result;
        }
        thread_.stack.push_back(thread_.current);
        thread_.current = ctx;
        *out = ctx->handle();
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) { return CUDA_ERROR_OUT_OF_MEMORY; }
}
CUresult Registry::context_destroy(CUcontext context) {
    std::shared_ptr<Context> ctx;
    {
        std::lock_guard lock(mutex_);
        ctx = live_locked(context);
        if (!ctx || ctx->is_primary() || ctx->retiring()) return CUDA_ERROR_INVALID_CONTEXT;
        CUresult result = virtual_begin_retire_context(context);
        if (result != CUDA_SUCCESS) return result;
        ctx->begin_retirement();
    }
    virtual_drain_context(context);
    {
        std::lock_guard lock(mutex_);
        bool was_current = thread_.current == ctx;
        retire_locked(ctx);
        if (was_current && !thread_.stack.empty()) {
            thread_.current = thread_.stack.back();
            thread_.stack.pop_back();
            if (!selected_locked()) thread_.current.reset();
        }
    }
    private_context_destroyed(context);
    return CUDA_SUCCESS;
}
CUresult Registry::context_current(CUcontext *out) const {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    // Only this thread mutates its stack; cross-thread retirement is atomic.
    *out = thread_.current && !thread_.current->retiring() ? thread_.current->handle() : nullptr;
    return CUDA_SUCCESS;
}
CUresult Registry::current_for_work(CUcontext *out) const {
    *out = thread_.current && !thread_.current->retiring() ? thread_.current->handle() : nullptr;
    return *out ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
}
CUresult Registry::context_set(CUcontext context) {
    std::lock_guard lock(mutex_);
    auto ctx = context ? live_locked(context) : nullptr;
    if (context && !ctx) return CUDA_ERROR_INVALID_CONTEXT;
    thread_.current = std::move(ctx);
    return CUDA_SUCCESS;
}
CUresult Registry::context_push(CUcontext context) {
    std::lock_guard lock(mutex_);
    auto ctx = live_locked(context);
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    try { thread_.stack.push_back(thread_.current); }
    catch (const std::bad_alloc&) { return CUDA_ERROR_OUT_OF_MEMORY; }
    thread_.current = std::move(ctx);
    return CUDA_SUCCESS;
}
CUresult Registry::context_pop(CUcontext *out) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    if (!selected_locked() || thread_.stack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    *out = thread_.current->handle();
    thread_.current = thread_.stack.back();
    thread_.stack.pop_back();
    return CUDA_SUCCESS;
}
CUresult Registry::context_device(CUdevice *out) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    auto ctx = selected_locked();
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    *out = ctx->device_ordinal();
    return CUDA_SUCCESS;
}
CUresult Registry::context_flags(unsigned int *out) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    auto ctx = selected_locked();
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    *out = ctx->flags();
    return CUDA_SUCCESS;
}
CUresult Registry::context_version(CUcontext context, unsigned int *out) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    if (!live_locked(context)) return CUDA_ERROR_INVALID_CONTEXT;
    *out = 13020;
    return CUDA_SUCCESS;
}
CUresult Registry::context_device(CUcontext context, CUdevice *out) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    auto ctx = live_locked(context);
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    *out = ctx->device_ordinal();
    return CUDA_SUCCESS;
}
CUresult Registry::peer_enable(CUcontext peer, unsigned int flags) {
    if (flags) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    auto current = selected_locked();
    auto other = live_locked(peer);
    if (!current || !other || current == other) return CUDA_ERROR_INVALID_CONTEXT;
    if (!device_configuration().can_access(current->device_ordinal(), other->device_ordinal()))
        return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
    try {
        auto& enabled = peers_[current->handle()];
        if (!enabled.insert(peer).second) return CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
    } catch (const std::bad_alloc&) {
        auto it = peers_.find(current->handle());
        if (it != peers_.end() && it->second.empty()) peers_.erase(it);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}
CUresult Registry::peer_disable(CUcontext peer) {
    std::lock_guard lock(mutex_);
    auto current = selected_locked();
    auto other = live_locked(peer);
    if (!current || !other || current == other) return CUDA_ERROR_INVALID_CONTEXT;
    if (current->device_ordinal() == other->device_ordinal())
        return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
    auto it = peers_.find(current->handle());
    if (it == peers_.end() || !it->second.erase(peer))
        return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
    if (it->second.empty()) peers_.erase(it);
    return CUDA_SUCCESS;
}
CUresult Registry::stream_create(CUstream *out, unsigned int flags, int priority) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    try {
        std::lock_guard lock(mutex_);
        auto ctx = selected_locked();
        if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
        // Match the -1..0 range advertised by cuCtxGetStreamPriorityRange.
                auto stream = std::make_shared<Stream>(ctx, flags, std::clamp(priority, -1, 0), ++next_handle_);
        streams_.emplace(stream->handle(), stream);
        *out = stream->handle();
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) { return CUDA_ERROR_OUT_OF_MEMORY; }
}
CUresult Registry::stream_destroy(CUstream stream) {
    std::lock_guard lock(mutex_);
    auto it = streams_.find(stream);
    if (it == streams_.end()) return CUDA_ERROR_INVALID_HANDLE;
    virtual_retire_stream(it->second->context()->handle(), stream);
    streams_.erase(it);
    return CUDA_SUCCESS;
}
CUresult Registry::stream_context(CUstream stream, CUcontext *out) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    auto ctx = stream_owner_locked(stream);
    if (!ctx) return CUDA_ERROR_INVALID_HANDLE;
    *out = ctx->handle();
    return CUDA_SUCCESS;
}
CUresult Registry::stream_device(CUstream stream, CUdevice *out) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    auto ctx = stream_owner_locked(stream);
    if (!ctx) return CUDA_ERROR_INVALID_HANDLE;
    *out = ctx->device_ordinal();
    return CUDA_SUCCESS;
}
CUresult Registry::stream_flags(CUstream stream, unsigned int *out) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    if (!stream_owner_locked(stream)) return CUDA_ERROR_INVALID_HANDLE;
    auto it = streams_.find(stream);
    *out = it == streams_.end() ? 0 : it->second->flags();
    return CUDA_SUCCESS;
}
CUresult Registry::stream_priority(CUstream stream, int *out) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(mutex_);
    if (!stream_owner_locked(stream)) return CUDA_ERROR_INVALID_HANDLE;
    auto it = streams_.find(stream);
    *out = it == streams_.end() ? 0 : it->second->priority();
    return CUDA_SUCCESS;
}
CUresult Registry::resolve_stream(CUstream stream, CUcontext *out, unsigned int *flags) {
    std::lock_guard lock(mutex_);
    auto ctx = selected_locked();
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    if (stream_owner_locked(stream) != ctx) return CUDA_ERROR_INVALID_HANDLE;
    auto it = streams_.find(stream);
    *flags = it == streams_.end() ? 0 : it->second->flags();
    *out = ctx->handle();
    return CUDA_SUCCESS;
}
} // namespace fake_cuda

namespace {
fake_cuda::Registry registry;
}
namespace fake_cuda {
Device *virtual_core_device(CUdevice ordinal) { return registry.device(ordinal); }
}

extern "C" {
CUresult core_init(unsigned int flags) { return flags ? CUDA_ERROR_INVALID_VALUE : fake_cuda::device_configuration().status; }
CUresult core_version(int *version) {
    if (!version) return CUDA_ERROR_INVALID_VALUE;
    *version = 13020;
    return CUDA_SUCCESS;
}
CUresult core_count(int *count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = fake_cuda::Device::count();
    return fake_cuda::device_configuration().status;
}
CUresult core_device(CUdevice *device, int ordinal) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    if (!fake_cuda::Device::valid(ordinal)) return CUDA_ERROR_INVALID_DEVICE;
    *device = ordinal;
    return CUDA_SUCCESS;
}
CUresult core_name(char *name, int size, CUdevice device) {
    if (!name || size <= 0) return CUDA_ERROR_INVALID_VALUE;
    if (!fake_cuda::Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    return registry.device(device)->name(name, size);
}
CUresult core_uuid(CUuuid *uuid, CUdevice device) {
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    if (!fake_cuda::Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    return registry.device(device)->uuid(uuid);
}
CUresult core_memory(size_t *bytes, CUdevice device) {
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    if (!fake_cuda::Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    return registry.device(device)->memory(bytes);
}
CUresult core_attribute(int *value, CUdevice_attribute attribute, CUdevice device) {
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    if (!fake_cuda::Device::valid(device)) return CUDA_ERROR_INVALID_DEVICE;
    return registry.device(device)->attribute(value, attribute);
}
CUresult core_can_access_peer(int *out, CUdevice device, CUdevice peer) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!fake_cuda::Device::valid(device) || !fake_cuda::Device::valid(peer))
        return CUDA_ERROR_INVALID_DEVICE;
    *out = fake_cuda::device_configuration().can_access(device, peer);
    return CUDA_SUCCESS;
}
CUresult core_primary_get(CUcontext *out, CUdevice device) { return registry.primary_get(out, device, false); }
CUresult core_primary_retain(CUcontext *out, CUdevice device) { return registry.primary_get(out, device, true); }
CUresult core_primary_state(CUdevice device, unsigned int *flags, int *active) {
    return registry.primary_state(device, flags, active);
}
CUresult core_primary_release(CUdevice device) { return registry.primary_release(device); }
CUresult core_primary_flags(CUdevice device, unsigned int flags) { return registry.primary_flags(device, flags); }
CUresult core_primary_reset(CUdevice device) { return registry.primary_reset(device); }
CUresult core_context_create(CUcontext *out, unsigned int flags, CUdevice device) {
    return registry.context_create(out, flags, device);
}
CUresult core_context_destroy(CUcontext context) { return registry.context_destroy(context); }
CUresult virtual_core_context_device(CUcontext context, CUdevice *device) {
    return registry.context_device(context, device);
}
CUresult core_context_enable_peer(CUcontext peer, unsigned int flags) {
    return registry.peer_enable(peer, flags);
}
CUresult core_context_disable_peer(CUcontext peer) { return registry.peer_disable(peer); }
CUresult core_context_current(CUcontext *out) { return registry.context_current(out); }
CUresult core_context_set(CUcontext context) { return registry.context_set(context); }
CUresult core_context_push(CUcontext context) { return registry.context_push(context); }
CUresult core_context_pop(CUcontext *out) { return registry.context_pop(out); }
CUresult core_context_device(CUdevice *out) { return registry.context_device(out); }
CUresult core_context_flags(unsigned int *out) { return registry.context_flags(out); }
CUresult core_context_version(CUcontext context, unsigned int *out) {
    return registry.context_version(context, out);
}
CUresult core_context_sync(void) {
    CUcontext context = nullptr;
    CUresult result = virtual_core_current(&context);
    return result == CUDA_SUCCESS ? fake_cuda::virtual_synchronize_context(context) : result;
}
CUresult virtual_core_current(CUcontext *out) { return registry.current_for_work(out); }
CUresult virtual_core_resolve_flags(CUstream stream, CUcontext *out, unsigned int *flags) {
    return registry.resolve_stream(stream, out, flags);
}
CUresult virtual_core_resolve(CUstream stream, CUcontext *out) {
    unsigned int flags = 0;
    return virtual_core_resolve_flags(stream, out, &flags);
}
CUresult core_stream_create(CUstream *out, unsigned int flags, int priority) {
    return registry.stream_create(out, flags, priority);
}
CUresult core_stream_destroy(CUstream stream) { return registry.stream_destroy(stream); }
CUresult core_stream_context(CUstream stream, CUcontext *out) { return registry.stream_context(stream, out); }
CUresult core_stream_device(CUstream stream, CUdevice *out) { return registry.stream_device(stream, out); }
CUresult core_stream_flags(CUstream stream, unsigned int *out) { return registry.stream_flags(stream, out); }
CUresult core_stream_priority(CUstream stream, int *out) { return registry.stream_priority(stream, out); }
CUresult core_stream_sync(CUstream stream) {
    CUcontext context = nullptr;
    CUresult result = virtual_core_resolve(stream, &context);
    return result == CUDA_SUCCESS ? fake_cuda::virtual_synchronize_stream(context, stream, false) : result;
}
CUresult core_stream_query(CUstream stream) {
    CUcontext context = nullptr;
    CUresult result = virtual_core_resolve(stream, &context);
    return result == CUDA_SUCCESS ? fake_cuda::virtual_synchronize_stream(context, stream, true) : result;
}
}
