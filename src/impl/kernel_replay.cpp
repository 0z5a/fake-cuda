#include "impl/kernel_replay.h"
#include "impl/resource_engine.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

namespace fake_cuda {
namespace {
bool add(Nanoseconds left, Nanoseconds right, Nanoseconds &result) {
    if (right.count() > std::numeric_limits<Nanoseconds::rep>::max() - left.count()) return false;
    result = left + right;
    return true;
}
}
static KernelReplayResult replay(PerformanceModel &model, std::span<const ReplayInvocation> trace, bool ready_driven) {
    const auto began = std::chrono::steady_clock::now();
    KernelReplayResult run;
    auto finish = [&](std::string error = {}) {
        run.error = std::move(error);
        if (!run.error.empty()) {
            run.intervals.clear();
            run.timing.service_time = Nanoseconds{};
            run.timing.host_time.reset();
            run.makespan = Nanoseconds{};
        }
        run.timing.simulator_run_time = std::chrono::duration_cast<Nanoseconds>(std::chrono::steady_clock::now() - began);
        return std::move(run);
    };
    std::set<std::uint64_t> seen;
    std::map<CUcontext, CUdevice> contexts;
    std::map<CUdevice, const DeviceProfile *> profiles;
    std::vector<KernelQuery> queries;
    for (const auto &entry : trace) {
        const auto &query = entry.query;
        if (!query.invocation_id || seen.contains(query.invocation_id)) return finish("duplicate or zero invocation ID");
        if (entry.arrival.count() < 0 || entry.host_service.count() < 0) return finish("negative target-host time");
        if (!query.context || query.device < 0 || !query.profile || !query.launch || !query.launch->kernel ||
            query.launch->validate(*query.profile) != CUDA_SUCCESS)
            return finish("invalid kernel invocation");
        const auto [context, added_context] = contexts.emplace(query.context, query.device);
        const auto [profile, added_profile] = profiles.emplace(query.device, query.profile);
        if ((!added_context && context->second != query.device) || (!added_profile && profile->second != query.profile))
            return finish("inconsistent context/device/profile mapping");
        for (auto dependency : entry.dependencies)
            if (!seen.contains(dependency)) return finish("dependency must reference an earlier invocation");
        seen.insert(query.invocation_id);
        queries.push_back(query);
    }
    run.timing.host_time = Nanoseconds{};
    if (trace.empty()) return finish();
    const auto query_start = std::chrono::steady_clock::now();
    auto predictions = model.predict(queries);
    run.timing.simulator_query_time = std::chrono::duration_cast<Nanoseconds>(std::chrono::steady_clock::now() - query_start);
    auto error = validate_predictions(queries, predictions);
    if (!error.empty()) return finish(std::move(error));
    Nanoseconds host_cursor{};
    std::map<CUdevice, Nanoseconds> devices;
    std::map<std::pair<CUcontext, std::uint64_t>, Nanoseconds> streams;
    std::map<std::uint64_t, Nanoseconds> completions;
    std::map<CUdevice, size_t> resources;
    std::map<std::pair<CUcontext, std::uint64_t>, std::uint64_t> predecessors;
    for (const auto &[device, profile] : profiles) resources.emplace(device, resources.size());
    ResourceEngine engine(std::vector<ResourceCapacity>(resources.size(), {0, 1}));
    for (size_t i = 0; i < trace.size(); ++i) {
        const auto &entry = trace[i];
        const auto &query = entry.query;
        const auto &prediction = predictions.results[i];
        const auto host_start = std::max(host_cursor, entry.arrival);
        if (!add(host_start, entry.host_service, host_cursor) ||
            !add(*run.timing.host_time, entry.host_service, *run.timing.host_time) ||
            !add(run.timing.service_time, prediction.service_time, run.timing.service_time))
            return finish("target timeline overflow");
        if (ready_driven) {
            auto dependencies = entry.dependencies;
            auto &prior = predecessors[{query.context, entry.stream}];
            if (prior) dependencies.push_back(prior);
            prior = query.invocation_id;
            std::vector<std::uint64_t> resident(resources.size());
            resident[resources.at(query.device)] = 1;
            engine.submit({query.invocation_id, host_cursor, std::move(dependencies),
                           {{prediction.service_time, std::vector<long double>(resources.size()), std::move(resident)}}});
            run.intervals.push_back({query.invocation_id, host_start, host_cursor, {}, {}, {}, prediction});
            continue;
        }
        auto &stream = streams[{query.context, entry.stream}];
        auto &device = devices[query.device];
        auto ready = std::max(host_cursor, stream);
        for (auto dependency : entry.dependencies) ready = std::max(ready, completions.at(dependency));
        const auto start = std::max(ready, device);
        Nanoseconds completion;
        if (!add(start, prediction.service_time, completion)) return finish("target timeline overflow");
        run.intervals.push_back({query.invocation_id, host_start, host_cursor, ready, start, completion, prediction});
        stream = device = completions[query.invocation_id] = completion;
        run.makespan = std::max(run.makespan, completion);
    }
    if (ready_driven) {
        Nanoseconds limit;
        if (!add(host_cursor, run.timing.service_time, limit)) return finish("target timeline overflow");
        while (auto next = engine.next_event()) engine.advance_to(*next);
        for (auto &interval : run.intervals) {
            const auto &progress = engine.progress(interval.invocation_id);
            if (!progress.completion) return finish("resource dependency blocked");
            interval.ready = *progress.ready;
            interval.start = *progress.start;
            interval.completion = *progress.completion;
            run.makespan = std::max(run.makespan, interval.completion);
        }
    }
    model.commit(predictions.results.size());
    return finish();
}
KernelReplayResult replay_kernel_trace(PerformanceModel &model, std::span<const ReplayInvocation> trace) {
    return replay(model, trace, false);
}
KernelReplayResult replay_resource_trace(PerformanceModel &model, std::span<const ReplayInvocation> trace) {
    return replay(model, trace, true);
}
} // namespace fake_cuda
