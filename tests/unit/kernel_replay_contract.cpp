#include "impl/kernel_replay.h"
#include "impl/replay_file.h"
#include "profile/device_profile.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); std::exit(1); \
} } while (0)
using namespace fake_cuda;
using namespace std::chrono_literals;

class Oracle final : public PerformanceModel {
public:
    explicit Oracle(std::vector<ReplaySample> samples) : replay(std::move(samples)) {}
    MeasuredReplay replay;
    Nanoseconds delay{};
    int fault = 0;
    mutable size_t calls = 0;
    PredictionBatch predict(std::span<const KernelQuery> queries) const override {
        ++calls;
        std::this_thread::sleep_for(delay);
        auto batch = replay.predict(queries);
        if (batch.results.size() < 2) return batch;
        switch (fault) {
        case 1: batch.results[1].covered_invocation = batch.results[0].covered_invocation; break;
        case 2: batch.results.pop_back(); break;
        case 3: batch.results.front().covered_invocation = 999; break;
        case 4: batch.results.front().scope = TimingScope::whole_forward; break;
        case 5: std::swap(batch.results[0], batch.results[1]); break;
        }
        return batch;
    }
    void commit(size_t count) noexcept override { replay.commit(count); }
};

int main(int argc, char **argv) {
    CHECK(argc == 2);
    DeviceProfile profile;
    auto context0 = reinterpret_cast<CUcontext>(1), context1 = reinterpret_cast<CUcontext>(2);
    KernelLaunch launch{std::make_shared<KernelIdentity>(KernelIdentity{1, "timing_contract"}),
                        {1, 1, 1}, {32, 1, 1}, 0, ParameterEncoding::packed_buffer, {}};
    std::vector<ReplayInvocation> trace{
        {{context0, 0, &profile, &launch, LaunchMode::eager, 11}, 1, 0ns, 2ms, {}},
        {{context1, 1, &profile, &launch, LaunchMode::eager, 12}, 1, 0ns, 1ms, {}},
        {{context0, 0, &profile, &launch, LaunchMode::eager, 13}, 2, 0ns, 3ms, {12}},
        {{context1, 1, &profile, &launch, LaunchMode::eager, 14}, 1, 12ms, 2ms, {11}}};
    const std::array durations{3ms, 6ms, 2ms, 1ms};
    std::vector<ReplaySample> samples;
    for (size_t i = 0; i < trace.size(); ++i) {
        const auto &q = trace[i].query;
        samples.push_back({q.context, q.device, q.profile, *q.launch, q.mode,
                          {durations[i], TimingScope::kernel, IncludedCosts::device_service,
                           ConfidenceKind::measured_replay, "synthetic accounting oracle"}});
    }
    Oracle fast(samples), slow(samples);
    fast.delay = 100us;
    slow.delay = 10ms;
    const auto warm = replay_kernel_trace(fast, trace);
    const auto cold = replay_kernel_trace(slow, trace);
    CHECK(warm.error.empty() && cold.error.empty());
    CHECK(fast.replay.consumed() == 4 && slow.replay.consumed() == 4);
    CHECK(warm.makespan == 15ms && cold.makespan == warm.makespan);
    CHECK(warm.timing.service_time == 12ms && cold.timing.service_time == 12ms);
    CHECK(warm.timing.host_time == 8ms && cold.timing.host_time == 8ms);
    CHECK(cold.timing.simulator_query_time >= 10ms);
    CHECK(*cold.timing.simulator_run_time >= cold.timing.simulator_query_time);
    CHECK(warm.intervals.size() == 4 && cold.intervals.size() == 4);
    const std::array starts{2ms, 3ms, 9ms, 14ms}, ends{5ms, 9ms, 11ms, 15ms};
    for (size_t i = 0; i < trace.size(); ++i) {
        const auto &a = warm.intervals[i], &b = cold.intervals[i];
        CHECK(a.start == starts[i] && a.completion == ends[i]);
        CHECK(a.host_start == b.host_start && a.submitted == b.submitted && a.ready == b.ready);
        CHECK(a.start == b.start && a.completion == b.completion);
        CHECK(a.prediction.covered_invocation == trace[i].query.invocation_id);
    }
    for (int fault = 1; fault <= 5; ++fault) {
        Oracle broken(samples); broken.fault = fault;
        auto failed = replay_kernel_trace(broken, trace);
        CHECK(!failed.error.empty() && failed.intervals.empty());
        CHECK(broken.replay.consumed() == 0 && failed.timing.service_time == 0ns);
        CHECK(!failed.timing.host_time.has_value());
    }
    DeviceProfile other_profile;
    for (int fault = 0; fault < 8; ++fault) {
        auto broken_trace = trace;
        switch (fault) {
        case 0: broken_trace[1].query.invocation_id = 11; break;
        case 1: broken_trace[0].query.invocation_id = 0; break;
        case 2: broken_trace[0].dependencies = {14}; break;
        case 3: broken_trace[0].dependencies = {11}; break;
        case 4: broken_trace[0].host_service = -1ns; break;
        case 5: broken_trace[0].arrival = -1ns; break;
        case 6: broken_trace[2].query.device = 1; break;
        case 7: broken_trace[2].query.profile = &other_profile; break;
        }
        Oracle unused(samples);
        CHECK(!replay_kernel_trace(unused, broken_trace).error.empty());
        CHECK(unused.calls == 0 && unused.replay.consumed() == 0);
    }
    auto overflow = trace;
    overflow[0].arrival = Nanoseconds::max();
    Oracle uncommitted(samples);
    auto failed = replay_kernel_trace(uncommitted, overflow);
    CHECK(failed.error == "target timeline overflow" && uncommitted.replay.consumed() == 0);
    CHECK(failed.intervals.empty() && failed.timing.service_time == 0ns);
    Oracle empty(samples);
    auto nothing = replay_kernel_trace(empty, {});
    CHECK(nothing.error.empty() && nothing.makespan == 0ns && nothing.timing.host_time == 0ns);
    CHECK(empty.calls == 0 && empty.replay.consumed() == 0);

    // File -> explicit bindings -> predictor -> offline time ledger.
    ReplayBinding binding{"case0", "contract-code-v1", "contract-hardware-v1", "fixed-empty-arguments",
                          context0, 0, &profile, launch};
    std::ifstream file(argv[1]);
    auto loaded = load_measured_replay(file, std::span(&binding, 1));
    CHECK(loaded.model);
    const std::array file_trace{
        ReplayInvocation{{context0, 0, &profile, &launch, LaunchMode::eager, 21}, 1, 0ns, 2ms, {}},
        ReplayInvocation{{context0, 0, &profile, &launch, LaunchMode::graph_replay, 22}, 1, 0ns, 1ms, {}},
        ReplayInvocation{{context0, 0, &profile, &launch, LaunchMode::graph_replay, 23}, 2, 0ns, 0ns, {22}}};
    const auto imported = replay_kernel_trace(*loaded.model, file_trace);
    CHECK(imported.error.empty() && imported.makespan == 33ms);
    CHECK(imported.timing.service_time == 31ms && imported.timing.host_time == 3ms);
    CHECK(imported.intervals.back().prediction.measurement->sample_index == 2);
    CHECK(imported.intervals.back().prediction.covered_invocation == 23);
    std::puts("PASS: exclusive kernel coverage, separate host/service/wall time, fixed-trace query-delay invariance");
    std::puts("| Query delay | Target makespan | Service sum | Host sum | Query wall | Replay call wall |\n"
              "| ---: | ---: | ---: | ---: | ---: | ---: |");
    auto row = [](int delay_us, const KernelReplayResult &run) {
        std::printf("| %d us | %.3f ms | %.3f ms | %.3f ms | %.3f ms | %.3f ms |\n", delay_us,
                    run.makespan.count()/1e6, run.timing.service_time.count()/1e6,
                    run.timing.host_time->count()/1e6, run.timing.simulator_query_time.count()/1e6,
                    run.timing.simulator_run_time->count()/1e6);
    };
    row(100, warm); row(10000, cold);
    // A pending cross-device event must not create an artificial compute edge.
    std::vector<ReplayInvocation> ready_trace{
        {{context1, 1, &profile, &launch, LaunchMode::eager, 31}, 1, 0ns, 0ns, {}},
        {{context0, 0, &profile, &launch, LaunchMode::eager, 32}, 1, 0ns, 0ns, {31}},
        {{context0, 0, &profile, &launch, LaunchMode::eager, 33}, 2, 0ns, 0ns, {}}};
    const std::array ready_costs{100ms, 10ms, 20ms};
    samples.clear();
    for (size_t i = 0; i < ready_trace.size(); ++i) {
        const auto &q = ready_trace[i].query;
        samples.push_back({q.context, q.device, q.profile, *q.launch, q.mode,
                          {ready_costs[i], TimingScope::kernel, IncludedCosts::device_service,
                           ConfidenceKind::measured_replay, "synthetic resource oracle"}});
    }
    Oracle reserved(samples), ready_model(samples), slow_ready(samples);
    slow_ready.delay = 10ms;
    auto before = replay_kernel_trace(reserved, ready_trace);
    auto after = replay_resource_trace(ready_model, ready_trace);
    auto delayed = replay_resource_trace(slow_ready, ready_trace);
    CHECK(before.makespan == 130ms && after.makespan == 110ms && delayed.makespan == after.makespan);
    CHECK(after.intervals[2].start == 0ns && after.intervals[2].completion == 20ms);
    CHECK(after.intervals[1].start == 100ms && after.intervals[1].completion == 110ms);
    CHECK(after.timing.service_time == before.timing.service_time && ready_model.replay.consumed() == 3);
    Oracle bad_ready(samples); bad_ready.fault = 1;
    CHECK(!replay_resource_trace(bad_ready, ready_trace).error.empty() && bad_ready.replay.consumed() == 0);
    std::puts("PASS: core predictor -> ready ResourceEngine, blocked-operation overtaking and query-delay invariance");
}
