#include "profile/device_profile.h"
#include "profile/gh200.h"

#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <string_view>

namespace fake_cuda {
namespace {
template<class T> bool number(std::string_view text, T &value) {
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc{} && end == text.data() + text.size();
}
bool read_profile(const char *path, DeviceProfile &profile) {
    std::ifstream input(path);
    if (!input) return false;
    std::set<std::string> fields;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        const auto split = line.find('=');
        if (split == std::string::npos) return false;
        const auto key = line.substr(0, split);
        const std::string_view value(line.data() + split + 1, line.size() - split - 1);
        if (value.empty() || !fields.insert(key).second) return false;
        if (key == "schema") { if (value != "1") return false; }
        else if (key == "name") profile.name = value;
        else if (key == "source") profile.source = value;
        else if (key == "memory_bytes") {
            if (!number(value, profile.memory_bytes) || !profile.memory_bytes) return false;
        } else if (key.starts_with("attribute.")) {
            int attribute = 0, attribute_value = 0;
            if (!number(std::string_view(key).substr(10), attribute) || attribute < 1 ||
                !number(value, attribute_value) ||
                !profile.attributes.emplace(attribute, attribute_value).second) return false;
        } else return false;
    }
    return !input.bad() && fields.contains("schema") && !profile.name.empty() &&
           !profile.source.empty() && profile.memory_bytes;
}
DeviceConfiguration configure() {
    DeviceConfiguration config;
    if (const char *count = std::getenv("FAKE_CUDA_DEVICE_COUNT")) {
        if (!number(std::string_view(count), config.count) || config.count < 1 || config.count > 1024) {
            config.count = 0;
            config.status = CUDA_ERROR_INVALID_VALUE;
            return config;
        }
    }
    if (const char *path = std::getenv("FAKE_CUDA_PROFILE")) {
        if (!read_profile(path, config.profile)) {
            config.count = 0;
            config.status = CUDA_ERROR_INVALID_VALUE;
        }
    } else {
        config.profile.name = gh200::name;
        config.profile.source = "GH200 Driver 13.2 snapshot (capabilities only)";
        config.profile.memory_bytes = gh200::memory_bytes;
        for (size_t i = 1; i < std::size(gh200::attributes); ++i)
            config.profile.attributes.emplace(static_cast<int>(i), gh200::attributes[i]);
    }
    return config;
}
} // namespace
const DeviceConfiguration &device_configuration() {
    static const DeviceConfiguration config = configure();
    return config;
}
} // namespace fake_cuda
