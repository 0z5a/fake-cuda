#include "impl/perf_model.h"

#include <limits>

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
    return {std::vector<PredictorResult>(queries.size(), synthetic_kernel_prediction()), {}};
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
    }
    return batch;
}
} // namespace fake_cuda
