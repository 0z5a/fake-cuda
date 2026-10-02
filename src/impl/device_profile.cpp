#include "profile/device_profile.h"
#include "profile/gh200.h"

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>

namespace fake_cuda {
namespace {
template<class T> bool number(std::string_view text, T &value) {
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc{} && end == text.data() + text.size();
}
using Fields = std::map<std::string, std::string>;
bool read_fields(const std::filesystem::path &path, Fields &fields) {
    std::ifstream input(path);
    if (!input) return false;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        const auto split = line.find('=');
        if (split == std::string::npos || split == 0 || split + 1 == line.size() ||
            !fields.emplace(line.substr(0, split), line.substr(split + 1)).second) return false;
    }
    const auto schema = fields.find("schema");
    if (input.bad() || schema == fields.end() || schema->second != "1") return false;
    fields.erase(schema);
    return true;
}
bool read_profile(const std::filesystem::path &path, DeviceProfile &profile) {
    Fields fields;
    if (!read_fields(path, fields)) return false;
    for (const auto &[key, value] : fields) {
        if (key == "name") profile.name = value;
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
    return !profile.name.empty() && !profile.source.empty() && profile.memory_bytes;
}
bool read_system(const std::filesystem::path &path, DeviceConfiguration &config) {
    Fields fields;
    if (!read_fields(path, fields) || !fields.contains("source")) return false;
    config.topology_source = fields.at("source");
    fields.erase("source");
    std::map<int, std::string> devices;
    std::map<std::pair<int, int>, bool> peers;
    for (const auto &[key, value] : fields) {
        if (key.starts_with("device.")) {
            int device = -1;
            if (!number(std::string_view(key).substr(7), device) || device < 0 || device >= 1024 ||
                !devices.emplace(device, value).second) return false;
        } else if (key.starts_with("peer.")) {
            const auto pair = std::string_view(key).substr(5);
            const auto split = pair.find('.');
            int source = -1, peer = -1;
            if (split == std::string_view::npos || !number(pair.substr(0, split), source) ||
                !number(pair.substr(split + 1), peer) || source < 0 || peer < 0 || source == peer ||
                (value != "0" && value != "1") ||
                !peers.emplace(std::pair{source, peer}, value == "1").second) return false;
        } else return false;
    }
    if (devices.empty() || devices.rbegin()->first != static_cast<int>(devices.size()) - 1) return false;
    config.count = static_cast<int>(devices.size());
    if (peers.size() != devices.size() * (devices.size() - 1)) return false;
    config.peer_access.assign(devices.size() * devices.size(), false);
    for (const auto &[pair, access] : peers) {
        if (pair.first >= config.count || pair.second >= config.count) return false;
        config.peer_access[static_cast<size_t>(pair.first) * config.count + pair.second] = access;
    }
    for (const auto &[ordinal, file] : devices) {
        (void)ordinal;
        auto profile = std::make_shared<DeviceProfile>();
        if (!read_profile(path.parent_path() / file, *profile)) return false;
        config.profiles.push_back(std::move(profile));
    }
    return true;
}
bool read_legacy(DeviceConfiguration &config) {
    if (const char *count = std::getenv("FAKE_CUDA_DEVICE_COUNT"))
        if (!number(std::string_view(count), config.count) || config.count < 1 || config.count > 1024)
            return false;
    auto profile = std::make_shared<DeviceProfile>();
    if (const char *path = std::getenv("FAKE_CUDA_PROFILE")) {
        if (!read_profile(path, *profile)) return false;
    } else {
        profile->name = gh200::name;
        profile->source = "GH200 Driver 13.2 snapshot (capabilities only)";
        profile->memory_bytes = gh200::memory_bytes;
        for (size_t i = 1; i < std::size(gh200::attributes); ++i)
            profile->attributes.emplace(static_cast<int>(i), gh200::attributes[i]);
    }
    config.profiles.assign(config.count, profile);
    config.peer_access.assign(static_cast<size_t>(config.count) * config.count, true);
    for (int i = 0; i < config.count; ++i) config.peer_access[static_cast<size_t>(i) * config.count + i] = false;
    return true;
}
DeviceConfiguration configure() {
    DeviceConfiguration config;
    const char *system = std::getenv("FAKE_CUDA_SYSTEM");
    const bool valid = system
        ? !std::getenv("FAKE_CUDA_PROFILE") && !std::getenv("FAKE_CUDA_DEVICE_COUNT") && read_system(system, config)
        : read_legacy(config);
    if (!valid) {
        config.count = 0;
        config.profiles.clear();
        config.peer_access.clear();
        config.status = CUDA_ERROR_INVALID_VALUE;
    }
    return config;
}
} // namespace
const DeviceConfiguration &device_configuration() {
    static const DeviceConfiguration config = configure();
    return config;
}
} // namespace fake_cuda
