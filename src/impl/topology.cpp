#include "impl/topology.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace fake_cuda {
std::vector<GlobalDevice> visible_devices(const std::vector<GlobalDevice> &catalog,
                                         const std::string &node, const std::vector<size_t> &ordinals) {
    std::vector<GlobalDevice> local, visible;
    for (const auto &device : catalog) if (device.node == node) local.push_back(device);
    std::set<size_t> unique;
    for (auto ordinal : ordinals) {
        if (ordinal >= local.size() || !unique.insert(ordinal).second)
            throw std::invalid_argument("invalid process-local device mapping");
        visible.push_back(local[ordinal]);
    }
    if (local.empty() || visible.empty()) throw std::invalid_argument("empty process device mapping");
    return visible;
}
Topology::Topology(std::vector<GlobalDevice> devices, std::vector<PhysicalLink> links,
                   std::vector<TransferPath> paths) : devices_(std::move(devices)), links_(std::move(links)), paths_(std::move(paths)) {
    std::set<GlobalDevice> identities;
    for (const auto &d : devices_)
        if (d.node.empty() || d.uuid.empty() || !identities.insert(d).second) throw std::invalid_argument("invalid global device identity");
    std::set<std::string> link_ids;
    for (const auto &l : links_)
        if (l.id.empty() || !std::isfinite(l.bytes_per_ns) || l.bytes_per_ns <= 0 || !link_ids.insert(l.id).second)
            throw std::invalid_argument("invalid physical link");
    std::set<std::pair<GlobalDevice, GlobalDevice>> pairs;
    for (const auto &p : paths_) {
        std::set<size_t> unique(p.links.begin(), p.links.end());
        if (!identities.contains(p.source) || !identities.contains(p.destination) || p.links.empty() ||
            unique.size() != p.links.size() || *unique.rbegin() >= links_.size() || !pairs.emplace(p.source, p.destination).second)
            throw std::invalid_argument("invalid transfer path");
    }
}
std::vector<ResourceCapacity> Topology::capacities() const {
    std::vector<ResourceCapacity> result;
    for (const auto &link : links_) result.push_back({link.bytes_per_ns, 0});
    return result;
}
ResourcePhase Topology::transfer(const GlobalDevice &source, const GlobalDevice &destination, std::uint64_t bytes) const {
    auto path = std::find_if(paths_.begin(), paths_.end(), [&](const auto &p) { return p.source == source && p.destination == destination; });
    if (path == paths_.end() || !bytes) throw std::invalid_argument("transfer has no declared path or payload");
    long double duration = 0;
    std::vector<long double> demand(links_.size());
    for (auto link : path->links) { demand[link] = bytes; duration = std::max(duration, bytes / links_[link].bytes_per_ns); }
    if (duration > Nanoseconds::max().count()) throw std::overflow_error("transfer duration overflow");
    return {Nanoseconds(static_cast<Nanoseconds::rep>(std::ceil(duration))), std::move(demand), std::vector<std::uint64_t>(links_.size())};
}
Nanoseconds ring_all_reduce(size_t ranks, std::uint64_t bytes, Nanoseconds alpha, long double bandwidth) {
    if (!ranks || alpha.count() < 0 || !std::isfinite(bandwidth) || bandwidth <= 0) throw std::invalid_argument("invalid ring model");
    if (ranks == 1) return Nanoseconds{};
    const auto duration = 2.L * (ranks - 1) * alpha.count() + 2.L * (ranks - 1) / ranks * bytes / bandwidth;
    if (duration > Nanoseconds::max().count()) throw std::overflow_error("ring duration overflow");
    return Nanoseconds(static_cast<Nanoseconds::rep>(std::ceil(duration)));
}
} // namespace fake_cuda
