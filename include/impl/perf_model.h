#ifndef FAKE_CUDA_IMPL_PERF_MODEL_H
#define FAKE_CUDA_IMPL_PERF_MODEL_H

#include "kernel_launch.h"

#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace fake_cuda {
using Nanoseconds = std::chrono::nanoseconds;
enum class LaunchMode { eager, graph_replay };
enum class TimingScope { kernel, whole_forward };
enum class IncludedCosts { device_service, end_to_end };
enum class ConfidenceKind { synthetic, measured_replay };

struct MeasurementIdentity {
    std::string binding, code, hardware, conditions;
    size_t sample_index;
};

struct PredictorResult {
    Nanoseconds service_time;
    TimingScope scope = TimingScope::kernel;
    IncludedCosts included_costs = IncludedCosts::device_service;
    ConfidenceKind confidence = ConfidenceKind::synthetic;
    std::string source;
    std::optional<MeasurementIdentity> measurement{};
    std::uint64_t covered_invocation = 0;
    std::shared_ptr<const SemanticBinding> semantic{};
    bool supported() const;
};
PredictorResult synthetic_kernel_prediction();

struct KernelQuery {
    CUcontext context;
    CUdevice device;
    const DeviceProfile *profile;
    const KernelLaunch *launch;
    LaunchMode mode;
    std::uint64_t invocation_id = 0;
};
struct PredictionBatch {
    std::vector<PredictorResult> results;
    std::string unsupported_reason;
};
std::string validate_predictions(std::span<const KernelQuery> queries, const PredictionBatch &batch);
class PerformanceModel {
public:
    virtual ~PerformanceModel() = default;
    virtual PredictionBatch predict(std::span<const KernelQuery> queries) const = 0;
    // Called only after the entire submission passes preflight and is scheduled.
    virtual void commit(size_t) noexcept {}
};
class SyntheticConstant final : public PerformanceModel {
public:
    PredictionBatch predict(std::span<const KernelQuery> queries) const override;
};

// Explicit bindings to this process's loads, contexts and immutable profiles.
// This is a finite ordered oracle, not a symbol/shape cache or cross-run code ID.
struct ReplaySample {
    CUcontext context;
    CUdevice device;
    const DeviceProfile *profile;
    KernelLaunch launch;
    LaunchMode mode;
    PredictorResult timing;
};
class MeasuredReplay final : public PerformanceModel {
public:
    explicit MeasuredReplay(std::vector<ReplaySample> samples) : samples_(std::move(samples)) {}
    PredictionBatch predict(std::span<const KernelQuery> queries) const override;
    void commit(size_t count) noexcept override { cursor_ += count; }
    size_t consumed() const noexcept { return cursor_; }
private:
    std::vector<ReplaySample> samples_;
    size_t cursor_ = 0;
};

struct TimingLedger {
    // Sum of committed kernel service costs, not makespan or transfer time.
    Nanoseconds service_time{};
    // Explicit host-service sum for trace replay; unknown in the paced Driver.
    std::optional<Nanoseconds> host_time;
    Nanoseconds simulator_query_time{};
    // Populated by a bounded replay run, not by the paced Driver path.
    std::optional<Nanoseconds> simulator_run_time;
};
} // namespace fake_cuda
#endif
