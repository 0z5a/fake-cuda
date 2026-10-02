#include "impl/core.h"
#include "impl/scheduler.h"
#include "impl/virtual_work.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); std::exit(1); \
} } while (0)
#define OK(expr) CHECK((expr) == CUDA_SUCCESS)

using namespace fake_cuda;
using namespace fake_cuda::detail;
using namespace std::chrono_literals;

// Deliberately synthetic oracle values; no claim of hardware calibration.
class DelayedModel final : public PerformanceModel {
public:
    Nanoseconds delay{};
    int coverage_fault = 0;
    PredictorResult result{7ms, TimingScope::kernel, IncludedCosts::device_service,
                           ConfidenceKind::synthetic, "contract-only"};
    PredictionBatch predict(std::span<const KernelQuery> queries) const override {
        std::this_thread::sleep_for(delay);
        PredictionBatch batch;
        for (const auto &query : queries) {
            batch.results.push_back(result);
            batch.results.back().covered_invocation = query.invocation_id;
            if (coverage_fault == 1) batch.results.back().covered_invocation = 0;
            if (coverage_fault == 2) batch.results.back().covered_invocation = queries.front().invocation_id;
        }
        return batch;
    }
};

int main() {
    OK(core_init(0));
    CUcontext context; CUstream stream;
    OK(core_context_create(&context, 0, 0));
    OK(core_stream_create(&stream, CU_STREAM_NON_BLOCKING, 0));
    CUmodule module; CUfunction function;
    OK(virtual_module_load_data(&module, "opaque contract image"));
    OK(virtual_module_get_function(&function, module, "timing_contract"));
    auto &s = scheduler();
    auto *device = virtual_core_device(0);
    auto last = [&] {
        auto op = device->compute_queue.last_operation();
        CHECK(op && op->prediction && op->launch);
        return op;
    };
    unsigned int scalar = 17;
    size_t bytes = sizeof(scalar);
    void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &scalar,
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes, CU_LAUNCH_PARAM_END};
    auto launch = [&] { return virtual_launch_kernel(function, 1, 1, 1, 32, 1, 1, 0, stream, nullptr, extra); };
    OK(launch());
    auto original = last();
    CHECK(original->end - original->start == 10ms);
    CHECK(original->prediction->source == "synthetic_constant_v1");
    CHECK(original->prediction->covered_invocation != 0);
    CHECK(!s.timing.host_time.has_value());
    auto sample = [&](LaunchMode mode, Nanoseconds duration) {
        return ReplaySample{context, 0, &device->profile(), *original->launch, mode,
                            {duration, TimingScope::kernel, IncludedCosts::device_service,
                             ConfidenceKind::measured_replay, "synthetic test values, explicit invocation binding"}};
    };
    auto replay = std::make_shared<MeasuredReplay>(std::vector{
        sample(LaunchMode::eager, 7ms), sample(LaunchMode::graph_replay, 11ms),
        sample(LaunchMode::graph_replay, 13ms), sample(LaunchMode::graph_replay, 17ms),
        sample(LaunchMode::graph_replay, 19ms)});
    s.performance_model = replay;
    const auto before = device->compute_queue.available_at();
    scalar = 18;
    CHECK(launch() == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(replay->consumed() == 0 && device->compute_queue.available_at() == before);
    CHECK(s.prediction_error == "measured replay invocation mismatch");
    scalar = 17;
    OK(launch());
    CHECK(last()->end - last()->start == 7ms && replay->consumed() == 1);
    CHECK(last()->prediction->covered_invocation != original->prediction->covered_invocation);

    CUgraph graph; CUgraphExec exec;
    OK(virtual_stream_begin_capture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(launch()); OK(launch());
    OK(virtual_stream_end_capture(stream, &graph));
    CHECK(replay->consumed() == 1); // Capture neither predicts nor charges.
    OK(virtual_graph_instantiate(&exec, graph));
    OK(virtual_graph_destroy(graph));
    auto wrong = sample(LaunchMode::graph_replay, 13ms);
    wrong.launch.grid[0] = 2;
    auto mismatch = std::make_shared<MeasuredReplay>(std::vector{sample(LaunchMode::graph_replay, 11ms), wrong});
    s.performance_model = mismatch;
    const auto before_graph = device->compute_queue.available_at();
    const auto charged = s.timing.service_time;
    CHECK(virtual_graph_launch(exec, stream) == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(mismatch->consumed() == 0 && s.timing.service_time == charged);
    CHECK(device->compute_queue.available_at() == before_graph);
    s.performance_model = replay;
    OK(virtual_graph_launch(exec, stream));
    auto first_replay = last();
    CHECK(first_replay->end - first_replay->start == 13ms);
    CHECK(s.timing.service_time - charged == 24ms && replay->consumed() == 3);
    OK(virtual_graph_launch(exec, stream));
    CHECK(last()->end - last()->start == 19ms);
    CHECK(last()->start >= first_replay->end && replay->consumed() == 5);
    CHECK(last()->prediction->covered_invocation != first_replay->prediction->covered_invocation);
    CHECK(s.timing.service_time - charged == 60ms);
    const auto completed = device->compute_queue.available_at();
    CHECK(virtual_graph_launch(exec, stream) == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(device->compute_queue.available_at() == completed && replay->consumed() == 5);

    // A later memory preflight failure must not consume an otherwise valid prediction.
    CUdeviceptr future;
    OK(s.memory.allocate(context, 0, &future, 4, VirtualClock::now() + 1h));
    Graph pending_memory(context, {
        {{Kind::kernel, 0, nullptr, 0, 0, 0, original->launch}, 0},
        {{Kind::h2d, 4, nullptr, future}, 0}}, 1);
    auto pending = std::make_shared<MeasuredReplay>(std::vector{sample(LaunchMode::graph_replay, 11ms)});
    s.performance_model = pending;
    const auto before_failure = s.timing.service_time;
    OpPtr completion;
    CHECK(pending_memory.launch(s, key_for(context, stream), completion) == CUDA_ERROR_INVALID_VALUE);
    CHECK(!completion && pending->consumed() == 0 && s.timing.service_time == before_failure);
    CHECK(device->compute_queue.available_at() == completed);

    // Exact matching rejects every key component and missing parameter provenance.
    const KernelQuery exact{context, 0, &device->profile(), original->launch.get(), LaunchMode::eager};
    for (int field = 0; field < 10; ++field) {
        auto changed = sample(LaunchMode::eager, 7ms);
        switch (field) {
        case 0: changed.context = nullptr; break;
        case 1: changed.device = 1; break;
        case 2: changed.profile = nullptr; break;
        case 3: changed.mode = LaunchMode::graph_replay; break;
        case 4: changed.launch.kernel = std::make_shared<KernelIdentity>(KernelIdentity{999, "timing_contract"}); break;
        case 5: changed.launch.kernel = std::make_shared<KernelIdentity>(KernelIdentity{original->launch->kernel->load_id, "other"}); break;
        case 6: changed.launch.grid[0] = 2; break;
        case 7: changed.launch.block[0] = 64; break;
        case 8: changed.launch.dynamic_shared_bytes = 16; break;
        case 9: changed.launch.parameters = ParameterEncoding::unknown_layout; break;
        }
        MeasuredReplay rejected({changed});
        CHECK(!rejected.predict(std::span(&exact, 1)).unsupported_reason.empty());
    }
    auto delayed = std::make_shared<DelayedModel>();
    s.performance_model = delayed;
    for (int invalid = 0; invalid < 5; ++invalid) {
        delayed->result = synthetic_kernel_prediction();
        switch (invalid) {
        case 0: delayed->result.scope = TimingScope::whole_forward; break;
        case 1: delayed->result.included_costs = IncludedCosts::end_to_end; break;
        case 2: delayed->result.service_time = -1ns; break;
        case 3: delayed->result.service_time = Nanoseconds::max(); break;
        case 4: delayed->result.source.clear(); break;
        }
        CHECK(launch() == CUDA_ERROR_NOT_SUPPORTED);
        CHECK(device->compute_queue.available_at() == completed);
    }
    delayed->result = {7ms, TimingScope::kernel, IncludedCosts::device_service,
                       ConfidenceKind::synthetic, "contract-only"};
    OK(launch());
    CHECK(last()->end - last()->start == 7ms);
    const auto query_time = s.timing.simulator_query_time;
    const auto service_time = s.timing.service_time;
    delayed->delay = 20ms;
    OK(launch());
    CHECK(last()->end - last()->start == 7ms);
    CHECK(s.timing.service_time - service_time == 7ms);
    CHECK(s.timing.simulator_query_time - query_time >= 20ms);
    CHECK(!s.timing.host_time.has_value());
    const auto accounted = s.timing.service_time;
    const auto scheduled = device->compute_queue.available_at();
    delayed->coverage_fault = 1;
    CHECK(launch() == CUDA_ERROR_NOT_SUPPORTED);
    delayed->coverage_fault = 2;
    CHECK(virtual_graph_launch(exec, stream) == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(s.timing.service_time == accounted && device->compute_queue.available_at() == scheduled);
    CHECK(s.prediction_error == "kernel invocation coverage mismatch");
    OK(virtual_graph_exec_destroy(exec));
    OK(virtual_module_unload(module));
    OK(core_context_destroy(context));
    std::puts("PASS: exact ordered replay, atomic graph rejection, accounting scope and query-time separation");
}
