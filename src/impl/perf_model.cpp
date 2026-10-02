#include "impl/perf_model.h"

#include <limits>
#include <set>

namespace fake_cuda {
bool PredictorResult::supported() const {
    return scope == TimingScope::kernel && included_costs == IncludedCosts::device_service &&
           service_time.count() >= 0 &&
           service_time.count() <= std::numeric_limits<Nanoseconds::rep>::max() / 4 &&
           !source.empty();
}
PredictorResult synthetic_kernel_prediction() {
    return {std::chrono::milliseconds(10), TimingScope::kernel, IncludedCosts::device_service,
            ConfidenceKind::synthetic, "synthetic_constant_v1"};
}
PredictionBatch SyntheticConstant::predict(std::span<const KernelQuery> queries) const {
    PredictionBatch batch;
    for (const auto &query : queries) {
        auto prediction = synthetic_kernel_prediction();
        prediction.covered_invocation = query.invocation_id;
        if (query.launch && query.launch->semantic)
            prediction.semantic = std::make_shared<SemanticBinding>(SemanticBinding{
                query.launch->semantic, query.invocation_id, query.launch->semantic->payload});
        batch.results.push_back(std::move(prediction));
    }
    return batch;
}
std::string validate_predictions(std::span<const KernelQuery> queries, const PredictionBatch &batch) {
    if (!batch.unsupported_reason.empty()) return batch.unsupported_reason;
    if (batch.results.size() != queries.size()) return "predictor result count mismatch";
    std::set<std::uint64_t> covered;
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto &prediction = batch.results[i];
        if (!prediction.supported()) return "unsupported timing scope, costs or duration";
        if (!queries[i].invocation_id || prediction.covered_invocation != queries[i].invocation_id ||
            !covered.insert(prediction.covered_invocation).second)
            return "kernel invocation coverage mismatch";
        const auto origin = queries[i].launch ? queries[i].launch->semantic : nullptr;
        if (origin || prediction.semantic) {
            if (!origin || origin->schema != 1 || origin->op_id.empty() || origin->payload.empty() ||
                !prediction.semantic || prediction.semantic->origin != origin ||
                prediction.semantic->invocation_id != queries[i].invocation_id ||
                prediction.semantic->payload.empty())
                return "semantic invocation binding mismatch";
        }
    }
    return {};
}
PredictionBatch MeasuredReplay::predict(std::span<const KernelQuery> queries) const {
    if (cursor_ > samples_.size() || queries.size() > samples_.size() - cursor_)
        return {{}, "measured replay exhausted"};
    PredictionBatch batch;
    batch.results.reserve(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto &query = queries[i];
        const auto &sample = samples_[cursor_ + i];
        if (!query.launch || !query.launch->kernel || !sample.launch.kernel || !query.profile)
            return {{}, "missing launch, load identity or profile"};
        const auto &launch = *query.launch;
        const auto &expected = sample.launch;
        if (launch.semantic || expected.semantic)
            return {{}, "measured kernel replay requires an explicit semantic binding provider"};
        if (launch.parameters != ParameterEncoding::packed_buffer ||
            expected.parameters != ParameterEncoding::packed_buffer)
            return {{}, "unknown parameter layout"};
        if (query.context != sample.context || query.device != sample.device ||
            query.profile != sample.profile || query.mode != sample.mode ||
            launch.kernel->load_id != expected.kernel->load_id ||
            launch.kernel->symbol != expected.kernel->symbol ||
            launch.grid != expected.grid || launch.block != expected.block ||
            launch.dynamic_shared_bytes != expected.dynamic_shared_bytes ||
            launch.packed_parameters != expected.packed_parameters)
            return {{}, "measured replay invocation mismatch"};
        if (!sample.timing.supported() || sample.timing.confidence != ConfidenceKind::measured_replay)
            return {{}, "invalid measured service-time accounting"};
        batch.results.push_back(sample.timing);
        batch.results.back().covered_invocation = query.invocation_id;
    }
    return batch;
}
} // namespace fake_cuda
