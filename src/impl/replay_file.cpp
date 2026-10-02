#include "impl/replay_file.h"

#include <charconv>
#include <map>
#include <string_view>

namespace fake_cuda {
namespace {
std::vector<std::string_view> split(std::string_view value, char delimiter) {
    std::vector<std::string_view> fields;
    size_t start = 0;
    for (;;) {
        const auto end = value.find(delimiter, start);
        fields.push_back(value.substr(start, end == value.npos ? end : end - start));
        if (end == value.npos) return fields;
        start = end + 1;
    }
}
template<class T> bool number(std::string_view text, T &out) {
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
    return error == std::errc{} && end == text.data() + text.size();
}
bool dimensions(std::string_view text, std::array<unsigned int, 3> &out) {
    const auto fields = split(text, ',');
    if (fields.size() != out.size()) return false;
    for (size_t i = 0; i < out.size(); ++i)
        if (!number(fields[i], out[i]) || !out[i]) return false;
    return true;
}
bool parameters(std::string_view text, std::vector<std::byte> &out) {
    if (text == "-") return true;
    if (text.empty() || text.size() % 2) return false;
    for (size_t i = 0; i < text.size(); i += 2) {
        unsigned int value;
        const auto [end, error] = std::from_chars(text.data() + i, text.data() + i + 2, value, 16);
        if (error != std::errc{} || end != text.data() + i + 2) return false;
        out.push_back(static_cast<std::byte>(value));
    }
    return true;
}
bool identity(const std::string &value) {
    return !value.empty() && value.find_first_of("\t\r\n") == std::string::npos;
}
} // namespace

ReplayFile load_measured_replay(std::istream &input, std::span<const ReplayBinding> bindings) {
    size_t line_number = 0;
    auto fail = [&](const char *reason) -> ReplayFile {
        return {nullptr, "line " + std::to_string(line_number) + ": " + reason};
    };
    std::map<std::string, const ReplayBinding *, std::less<>> by_id;
    for (const auto &binding : bindings) {
        if (!identity(binding.id) || !identity(binding.code_identity) ||
            !identity(binding.hardware_identity) || !identity(binding.conditions) ||
            !binding.context || binding.device < 0 || !binding.profile || !binding.launch.kernel ||
            !identity(binding.launch.kernel->symbol) ||
            binding.launch.parameters != ParameterEncoding::packed_buffer ||
            binding.launch.validate(*binding.profile) != CUDA_SUCCESS)
            return fail("invalid or incomplete replay binding");
        if (!by_id.emplace(binding.id, &binding).second) return fail("duplicate binding ID");
    }
    std::map<std::string, std::string> headers;
    std::vector<ReplaySample> samples;
    std::string line;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        const auto equal = line.find('=');
        if (equal == line.npos) return fail("expected key=value");
        const auto key = line.substr(0, equal);
        const std::string_view value(line.data() + equal + 1, line.size() - equal - 1);
        if (key != "sample") {
            if (!samples.empty()) return fail("headers must precede samples");
            if (key != "schema" && key != "source" && key != "scope" && key != "included_costs")
                return fail("unknown header");
            if (value.empty() || !headers.emplace(key, value).second) return fail("empty or duplicate header");
            continue;
        }
        if (headers.size() != 4 || headers["schema"] != "1" || headers["scope"] != "kernel" ||
            headers["included_costs"] != "device_service")
            return fail("unsupported or incomplete schema/accounting headers");
        const auto fields = split(value, '\t');
        if (fields.size() != 11) return fail("sample requires 11 tab-separated fields");
        const auto found = by_id.find(fields[0]);
        if (found == by_id.end()) return fail("unknown binding ID");
        const auto &binding = *found->second;
        if (fields[1] != binding.code_identity || fields[2] != binding.hardware_identity ||
            fields[3] != binding.conditions || fields[4] != binding.launch.kernel->symbol)
            return fail("code, hardware, conditions or symbol mismatch");
        LaunchMode mode;
        if (fields[5] == "eager") mode = LaunchMode::eager;
        else if (fields[5] == "graph_replay") mode = LaunchMode::graph_replay;
        else return fail("unsupported launch mode");
        std::array<unsigned int, 3> grid, block;
        unsigned int shared;
        std::vector<std::byte> packed;
        Nanoseconds::rep duration;
        if (!dimensions(fields[6], grid) || !dimensions(fields[7], block) || !number(fields[8], shared) ||
            !parameters(fields[9], packed) || !number(fields[10], duration))
            return fail("invalid dimensions, shared memory, parameters or duration");
        if (grid != binding.launch.grid || block != binding.launch.block ||
            shared != binding.launch.dynamic_shared_bytes || packed != binding.launch.packed_parameters)
            return fail("launch metadata mismatch");
        PredictorResult timing{Nanoseconds(duration), TimingScope::kernel, IncludedCosts::device_service,
                               ConfidenceKind::measured_replay, headers.at("source"),
                               MeasurementIdentity{binding.id, binding.code_identity, binding.hardware_identity,
                                                   binding.conditions, samples.size()}};
        if (!timing.supported()) return fail("invalid service duration");
        samples.push_back({binding.context, binding.device, binding.profile, binding.launch, mode, std::move(timing)});
    }
    if (input.bad() || !input.eof()) return fail("input read failed");
    if (samples.empty()) return fail("no replay samples");
    return {std::make_unique<MeasuredReplay>(std::move(samples)), {}};
}
} // namespace fake_cuda
