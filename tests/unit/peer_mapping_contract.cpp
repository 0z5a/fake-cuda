#include "impl/core.h"
#include "impl/scheduler.h"
#include "impl/virtual_work.h"

#include <array>
#include <cstdio>
#include <cstdlib>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); std::exit(1); \
} } while (0)
#define OK(expr) CHECK((expr) == CUDA_SUCCESS)
#define INVALID(expr) CHECK((expr) == CUDA_ERROR_INVALID_VALUE)

using namespace fake_cuda;
using namespace fake_cuda::detail;
using namespace std::chrono_literals;

class CountingModel final : public PerformanceModel {
public:
    mutable size_t queries = 0;
    size_t committed = 0;
    PredictionBatch predict(std::span<const KernelQuery> input) const override {
        queries += input.size();
        PredictionBatch batch;
        for (const auto &query : input) {
            batch.results.push_back(synthetic_kernel_prediction());
            batch.results.back().covered_invocation = query.invocation_id;
        }
        return batch;
    }
    void commit(size_t count) noexcept override { committed += count; }
};

int main() {
    constexpr size_t bytes = 1024;
    CHECK(setenv("FAKE_CUDA_P2P_BW_GBPS", "0.00001", 1) == 0);
    OK(core_init(0));
    int count = 0;
    OK(core_count(&count)); CHECK(count == 3);
    std::array<CUcontext, 3> ctx{};
    std::array<CUdeviceptr, 3> ptr{};
    std::array<std::array<CUstream, 2>, 3> streams{};
    for (int device = 0; device < 3; ++device) {
        OK(core_context_create(&ctx[device], 0, device));
        OK(virtual_mem_alloc(&ptr[device], bytes));
        for (auto &stream : streams[device])
            OK(core_stream_create(&stream, CU_STREAM_NON_BLOCKING, 0));
    }
    auto &s = scheduler();
    auto *d0 = virtual_core_device(0), *d1 = virtual_core_device(1);
    auto snapshot = [&] {
        return std::array{d0->compute_queue.available_at(), d1->compute_queue.available_at(),
            d0->h2d_queue.available_at(), d0->d2h_queue.available_at(),
            d0->p2p_queues[1].available_at(), d1->p2p_queues[0].available_at()};
    };

    // UVA allocation identity survives mapping changes; device access does not.
    OK(core_context_set(ctx[0]));
    CUcontext owner = nullptr;
    int ordinal = -1; unsigned int mapped = 0;
    CUdeviceptr accessible = 0, base = 0; size_t size = 0;
    OK(virtual_pointer_get_attribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, ptr[1] + 4)); CHECK(owner == ctx[1]);
    OK(virtual_pointer_get_attribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr[1])); CHECK(ordinal == 1);
    OK(virtual_pointer_get_attribute(&base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, ptr[1] + 4)); CHECK(base == ptr[1]);
    OK(virtual_pointer_get_attribute(&size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, ptr[1])); CHECK(size == bytes);
    OK(virtual_pointer_get_attribute(&mapped, CU_POINTER_ATTRIBUTE_MAPPED, ptr[1])); CHECK(mapped == 1);
    INVALID(virtual_pointer_get_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr[1]));
    INVALID(virtual_pointer_get_attribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, ptr[1] + bytes));
    const auto before_enable = snapshot();
    INVALID(virtual_memcpy_d2d(ptr[0], ptr[1], bytes, streams[0][0], 1));
    CHECK(snapshot() == before_enable);
    OK(core_context_enable_peer(ctx[1], 0));
    CHECK(core_context_enable_peer(ctx[1], 0) == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
    OK(virtual_pointer_get_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr[1] + 4));
    CHECK(accessible == ptr[1] + 4);
    INVALID(virtual_mem_free(ptr[1]));
    INVALID(virtual_mem_free_async(ptr[1], streams[0][0]));
    INVALID(virtual_memcpy_d2d(ptr[0], ptr[1] + bytes - 4, 8, streams[0][0], 1));
    INVALID(virtual_memcpy_d2d(ptr[0], ptr[2], bytes, streams[0][0], 1));
    OK(core_context_set(ctx[1]));
    CHECK(core_context_enable_peer(ctx[0], 0) == CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
    INVALID(virtual_memcpy_d2d(ptr[1], ptr[0], bytes, streams[1][0], 1));

    // A directed context mapping includes future allocations, but not another
    // context on the same peer GPU. Allocation ownership stays with its creator.
    CUdeviceptr later = 0, extra_ptr = 0;
    OK(virtual_mem_alloc(&later, bytes));
    CUcontext extra = nullptr;
    OK(core_context_create(&extra, 0, 1));
    OK(virtual_mem_alloc(&extra_ptr, bytes));
    OK(core_context_set(ctx[0]));
    OK(virtual_pointer_get_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, later));
    INVALID(virtual_pointer_get_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, extra_ptr));
    OK(core_context_enable_peer(extra, 0));
    OK(virtual_pointer_get_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, extra_ptr));
    OK(core_context_disable_peer(extra));
    INVALID(virtual_pointer_get_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, extra_ptr));
    OK(core_context_destroy(extra));

    // Mapping permission is observer -> owner. The charged transfer resource is
    // source -> destination, including the reverse of that permission direction.
    const auto compute0 = d0->compute_queue.available_at();
    const auto compute1 = d1->compute_queue.available_at();
    OK(virtual_memcpy_d2d(ptr[0], later, bytes, streams[0][0], 1));
    auto read = d1->p2p_queues[0].last_operation(); CHECK(read);
    CHECK(read->end - read->start == QueueScheduler::duration(Kind::peer, bytes));
    OK(virtual_memcpy_d2d(later, ptr[0], bytes, streams[0][1], 1));
    auto write = d0->p2p_queues[1].last_operation(); CHECK(write);
    OK(virtual_memcpy_d2d(ptr[0], later, bytes, streams[0][1], 1));
    auto second_read = d1->p2p_queues[0].last_operation(); CHECK(second_read);
    CHECK(second_read->start >= read->end && second_read->start >= write->end);
    CHECK(d0->compute_queue.available_at() == compute0 && d1->compute_queue.available_at() == compute1);
    OK(virtual_memcpy_d2d(ptr[1], later, bytes, streams[0][0], 1));
    auto remote_local = d1->compute_queue.last_operation(); CHECK(remote_local);
    CHECK(remote_local->end - remote_local->start == QueueScheduler::duration(Kind::compute, bytes));
    CHECK(d0->compute_queue.available_at() == compute0);
    CHECK(s.queue.context_end(ctx[1]) >= remote_local->end);
    CHECK(s.queue.context_end(ctx[1]) >= second_read->end);
    CHECK(s.memory.can_free_async(ctx[1], later, second_read->start) == CUDA_ERROR_INVALID_VALUE);
    OK(core_context_set(ctx[1]));
    OK(core_context_sync()); CHECK(VirtualClock::now() >= second_read->end);

    OK(core_context_set(ctx[0]));
    auto model = std::make_shared<CountingModel>(); s.performance_model = model;
    CUmodule module; CUfunction function;
    OK(virtual_module_load_data(&module, "opaque contract image"));
    OK(virtual_module_get_function(&function, module, "peer_contract"));
    auto capture = [&](CUdeviceptr source) {
        CUgraph graph; CUgraphExec exec;
        const auto resources = snapshot(); const auto charged = s.timing.service_time;
        const auto queried = model->queries;
        OK(virtual_stream_begin_capture(streams[0][0], CU_STREAM_CAPTURE_MODE_GLOBAL));
        OK(virtual_launch_kernel(function, 1, 1, 1, 32, 1, 1, 0, streams[0][0], nullptr, nullptr));
        OK(virtual_memcpy_d2d(ptr[0], source, bytes, streams[0][0], 1));
        OK(virtual_memcpy_d2d(source, ptr[0], bytes, streams[0][0], 1));
        OK(virtual_memcpy_d2d(ptr[1], source, bytes, streams[0][0], 1));
        OK(virtual_stream_end_capture(streams[0][0], &graph));
        CHECK(snapshot() == resources && s.timing.service_time == charged && model->queries == queried);
        OK(virtual_graph_instantiate(&exec, graph)); OK(virtual_graph_destroy(graph));
        return exec;
    };
    CUgraphExec exec = capture(later);
    auto rejected = [&](CUgraphExec executable, bool query_allowed = false) {
        const auto resources = snapshot(); const auto charged = s.timing.service_time;
        const auto committed = model->committed, queried = model->queries;
        INVALID(virtual_graph_launch(executable, streams[0][1]));
        CHECK(snapshot() == resources && s.timing.service_time == charged && model->committed == committed);
        CHECK(query_allowed || model->queries == queried);
    };
    OK(core_context_disable_peer(ctx[1]));
    OK(virtual_pointer_get_attribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, later)); CHECK(owner == ctx[1]);
    INVALID(virtual_pointer_get_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, later));
    rejected(exec);
    OK(core_context_enable_peer(ctx[1], 0));
    for (auto stream : streams[0]) {
        OK(virtual_graph_launch(exec, stream));
        auto peer = d0->p2p_queues[1].last_operation(); CHECK(peer);
        CHECK(s.queue.context_end(ctx[1]) >= peer->end);
    }
    CHECK(model->committed == 2);
    // Synchronous owner free must include use submitted from the observing
    // context, even when the source graph handle was destroyed.
    auto peer_finish = s.queue.context_end(ctx[1]);
    OK(core_context_set(ctx[1])); OK(virtual_mem_free(later));
    CHECK(VirtualClock::now() >= peer_finish);
    OK(core_context_set(ctx[0])); rejected(exec);
    OK(virtual_graph_exec_destroy(exec));

    // A later temporal failure also leaves predictions/resources uncommitted.
    CUdeviceptr future;
    OK(s.memory.allocate(ctx[1], 1, &future, bytes, VirtualClock::now() + 1h));
    CUgraphExec pending = capture(future);
    rejected(pending, true);
    OK(core_context_destroy(ctx[1])); // No operation using future was admitted.
    rejected(pending);
    INVALID(virtual_pointer_get_attribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, ptr[1]));
    OK(virtual_graph_exec_destroy(pending));
    CUcontext replacement;
    OK(core_context_create(&replacement, 0, 1));
    CUdeviceptr replacement_ptr;
    OK(virtual_mem_alloc(&replacement_ptr, bytes)); CHECK(replacement_ptr != ptr[1]);
    OK(core_context_set(ctx[0]));
    INVALID(virtual_pointer_get_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, replacement_ptr));
    OK(core_context_enable_peer(replacement, 0));
    OK(virtual_pointer_get_attribute(&accessible, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, replacement_ptr));
    OK(core_context_destroy(replacement));
    OK(core_context_destroy(ctx[0])); OK(core_context_destroy(ctx[2]));
    std::puts("PASS: directed peer pointers, allocation ownership/lifetimes, D2D routing and atomic Graph replay");
}
