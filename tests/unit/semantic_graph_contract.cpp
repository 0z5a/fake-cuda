#include "impl/core.h"
#include "impl/scheduler.h"
#include "impl/virtual_work.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL: %s\n", #expr); std::exit(1); } } while (0)
using namespace fake_cuda;
using namespace fake_cuda::detail;

class BindingModel final : public PerformanceModel {
public:
    std::string payload;
    bool stale = false;
    PredictionBatch predict(std::span<const KernelQuery> queries) const override {
        PredictionBatch batch;
        for (const auto &query : queries) {
            auto result = synthetic_kernel_prediction();
            result.service_time = Nanoseconds(1);
            result.covered_invocation = query.invocation_id;
            result.semantic = std::make_shared<SemanticBinding>(SemanticBinding{
                query.launch->semantic, stale ? 1 : query.invocation_id, payload});
            batch.results.push_back(std::move(result));
        }
        return batch;
    }
};

int main() {
    CHECK(core_init(0) == CUDA_SUCCESS);
    CUcontext context; CUstream stream;
    CHECK(core_context_create(&context, 0, 0) == CUDA_SUCCESS);
    CHECK(core_stream_create(&stream, CU_STREAM_NON_BLOCKING, 0) == CUDA_SUCCESS);
    CUmodule module; CUfunction function;
    CHECK(virtual_module_load_data(&module, "opaque") == CUDA_SUCCESS);
    CHECK(virtual_module_get_function(&function, module, "unclassified_kernel") == CUDA_SUCCESS);
    CHECK(virtual_launch_kernel(function, 1, 1, 1, 32, 1, 1, 0, stream, nullptr, nullptr) == CUDA_SUCCESS);
    auto &s = scheduler();
    auto *device = virtual_core_device(0);
    auto launch = std::make_shared<KernelLaunch>(*device->compute_queue.last_operation()->launch);
    auto descriptor = std::make_shared<SemanticTemplate>(SemanticTemplate{1, "layer0", "bounded route/state template"});
    std::weak_ptr<const SemanticTemplate> lifetime = descriptor;
    launch->semantic = descriptor;
    auto graph = std::make_shared<Graph>(context, std::vector<GraphNode>{
        {{Kind::kernel, 0, nullptr, 0, 0, 0, launch}, 0}}, 1);
    auto exec = graph;
    graph.reset(); launch.reset(); descriptor.reset();
    CHECK(!lifetime.expired()); // Executable retains the graph and immutable template.
    auto model = std::make_shared<BindingModel>();
    s.performance_model = model;
    OpPtr completion;
    auto replay = [&] {
        std::lock_guard lock(s.mutex);
        return exec->launch(s, key_for(context, stream), completion);
    };
    auto binding = [&] {
        std::lock_guard lock(s.mutex);
        return device->compute_queue.last_operation()->prediction->semantic;
    };
    model->payload = "counts=1,2,1,0;state_slots=0,1";
    CHECK(replay() == CUDA_SUCCESS);
    auto first = binding();
    model->payload = "counts=2,2,0,0;state_slots=2,3";
    CHECK(replay() == CUDA_SUCCESS);
    auto second = binding();
    CHECK(first->origin == second->origin && first->invocation_id != second->invocation_id);
    CHECK(first->payload != second->payload);
    const auto charged = s.timing.service_time;
    const auto tail = device->compute_queue.available_at();
    model->stale = true;
    CHECK(replay() == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(s.timing.service_time == charged && device->compute_queue.available_at() == tail);
    model->stale = false; model->payload.clear();
    CHECK(replay() == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(virtual_module_unload(module) == CUDA_SUCCESS);
    CHECK(core_context_destroy(context) == CUDA_SUCCESS);
    exec.reset(); completion.reset();
    CHECK(!lifetime.expired()); // The last consumers still retain their bindings.
    first.reset(); second.reset();
    CHECK(lifetime.expired());
    std::puts("PASS: owned graph templates, fresh dynamic bindings, atomic stale-binding rejection");
}
