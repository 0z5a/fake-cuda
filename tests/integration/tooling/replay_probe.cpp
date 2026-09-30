#include "impl/kernel_replay.h"
#include "profile/device_profile.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

using namespace fake_cuda;
namespace {
std::vector<std::string> split(const std::string &text, char delimiter) {
    std::vector<std::string> fields;
    std::istringstream input(text);
    for (std::string field; std::getline(input, field, delimiter);) fields.push_back(std::move(field));
    return fields;
}
template<class T> bool number(const std::string &text, T &value) {
    auto [end, error] = std::from_chars(text.data(), text.data()+text.size(), value);
    return error == std::errc{} && end == text.data()+text.size();
}
bool dimensions(const std::string &text, std::array<unsigned int, 3> &values) {
    const auto fields = split(text, ',');
    if (fields.size() != 3) return false;
    for (size_t i = 0; i < 3; ++i) if (!number(fields[i], values[i]) || !values[i]) return false;
    return true;
}
struct Observation {
    std::uint64_t id;
    CUdevice device;
    KernelLaunch launch;
    unsigned int static_shared, registers;
    PredictorResult prediction;
};
// This provider is confined to one already-profiled observation. Unknown ABI
// stays unknown; the observation is never offered as a general launch cache.
class ObservedDuration final : public PerformanceModel {
public:
    ObservedDuration(KernelQuery query, PredictorResult result) : query_(query), result_(std::move(result)) {}
    PredictionBatch predict(std::span<const KernelQuery> queries) const override {
        if (queries.size() != 1) return {{}, "expected one isolated observation"};
        const auto &q = queries.front();
        if (q.invocation_id != query_.invocation_id || q.launch != query_.launch ||
            q.context != query_.context || q.device != query_.device || q.profile != query_.profile || q.mode != query_.mode)
            return {{}, "observation identity mismatch"};
        return {{result_}, {}};
    }
private:
    KernelQuery query_;
    PredictorResult result_;
};
int fail(const char *message) { std::fprintf(stderr, "%s\n", message); return 1; }
bool digest_identity(const std::string &value, std::string_view prefix) {
    return value.starts_with(prefix) && value.size() == prefix.size()+64 &&
        std::all_of(value.begin()+prefix.size(), value.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
}
int main(int argc, char **argv) {
    if (argc != 2) return fail("usage: replay_probe kernels.probe");
    std::ifstream input(argv[1]);
    std::map<std::string, std::string> headers;
    std::set<std::uint64_t> ids;
    std::vector<Observation> observations;
    DeviceProfile profile; // Observed legal launches; no unmeasured capability inference.
    for (std::string line; std::getline(input, line);) {
        const auto equal = line.find('=');
        if (equal == line.npos) return fail("expected key=value");
        const auto key = line.substr(0, equal), value = line.substr(equal+1);
        if (key != "observation") {
            if (!observations.empty() || value.empty() ||
                (key != "schema" && key != "scope" && key != "source" && key != "code" && key != "hardware" && key != "conditions") ||
                !headers.emplace(key, value).second) return fail("invalid probe headers");
            continue;
        }
        if (headers.size() != 6 || headers["schema"] != "1" || headers["scope"] != "isolated_kernel_observations" ||
            !digest_identity(headers["source"], "ncu-report-sha256:") ||
            (!digest_identity(headers["code"], "declared-image-sha256:") &&
             !digest_identity(headers["code"], "captured-cubin-sha256:")) ||
            !digest_identity(headers["hardware"], "sha256:") || !digest_identity(headers["conditions"], "sha256:"))
            return fail("unsupported probe schema/scope");
        const auto fields = split(value, '\t');
        Observation observation{};
        Nanoseconds::rep duration;
        if (fields.size() != 9 || !number(fields[0], observation.id) || !observation.id ||
            !ids.insert(observation.id).second || !number(fields[1], observation.device) || observation.device < 0 ||
            !dimensions(fields[2], observation.launch.grid) || !dimensions(fields[3], observation.launch.block) ||
            !number(fields[4], observation.static_shared) || !number(fields[5], observation.launch.dynamic_shared_bytes) ||
            !number(fields[6], observation.registers) || !number(fields[7], duration) || fields[8].empty())
            return fail("invalid kernel observation");
        observation.launch.kernel = std::make_shared<KernelIdentity>(KernelIdentity{observation.id, fields[8]});
        observation.prediction = {Nanoseconds(duration), TimingScope::kernel, IncludedCosts::device_service,
            ConfidenceKind::measured_replay, headers.at("source"),
            MeasurementIdentity{fields[0], headers.at("code"), headers.at("hardware"), headers.at("conditions"), observations.size()},
            observation.id};
        if (!observation.prediction.supported() || observation.launch.validate(profile) != CUDA_SUCCESS)
            return fail("unsupported observation duration or launch");
        observations.push_back(std::move(observation));
    }
    if (!input.eof() || observations.empty()) return fail("unreadable or empty probe");
    std::puts("Independent kernel observations; host/arrival/transfer/dependency times were not collected.\n");
    std::printf("Source: `%s`\n\nCode: `%s`\n\nHardware: `%s`\n\nConditions: `%s`\n\n",
                headers.at("source").c_str(), headers.at("code").c_str(), headers.at("hardware").c_str(), headers.at("conditions").c_str());
    std::puts("| ID | Default service (ns) | Observed service (ns) | Replayed completion (ns) | Replay-call wall (ns) |\n|---:|---:|---:|---:|---:|");
    for (auto &observation : observations) {
        const KernelQuery query{reinterpret_cast<CUcontext>(1), observation.device, &profile,
                                &observation.launch, LaunchMode::eager, observation.id};
        const ReplayInvocation entry{query, 1, Nanoseconds{}, Nanoseconds{}, {}};
        SyntheticConstant baseline;
        ObservedDuration measured(query, observation.prediction);
        const auto before = replay_kernel_trace(baseline, std::span(&entry, 1));
        const auto after = replay_kernel_trace(measured, std::span(&entry, 1));
        if (!before.error.empty() || !after.error.empty()) return fail("isolated replay failed");
        std::printf("| %llu | %lld | %lld | %lld | %lld |\n", static_cast<unsigned long long>(observation.id),
                    static_cast<long long>(before.makespan.count()), static_cast<long long>(observation.prediction.service_time.count()),
                    static_cast<long long>(after.makespan.count()), static_cast<long long>(after.timing.simulator_run_time->count()));
    }
}
