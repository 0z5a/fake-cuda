#ifndef FAKE_CUDA_IMPL_TOPOLOGY_H
#define FAKE_CUDA_IMPL_TOPOLOGY_H
#include "resource_engine.h"

namespace fake_cuda {
struct GlobalDevice {
    std::string node, uuid;
    auto operator<=>(const GlobalDevice &) const = default;
};
struct PhysicalLink { std::string id; long double bytes_per_ns; };
struct TransferPath { GlobalDevice source, destination; std::vector<size_t> links; };
// Explicit process visibility: ordinals index devices within this node, in
// catalog order. A reordered/subset map never changes their global identity.
std::vector<GlobalDevice> visible_devices(const std::vector<GlobalDevice> &catalog,
                                         const std::string &node, const std::vector<size_t> &ordinals);
class Topology {
public:
    Topology(std::vector<GlobalDevice>, std::vector<PhysicalLink>, std::vector<TransferPath>);
    std::vector<ResourceCapacity> capacities() const;
    ResourcePhase transfer(const GlobalDevice &, const GlobalDevice &, std::uint64_t bytes) const;
private:
    std::vector<GlobalDevice> devices_;
    std::vector<PhysicalLink> links_;
    std::vector<TransferPath> paths_;
};
Nanoseconds ring_all_reduce(size_t ranks, std::uint64_t payload_bytes, Nanoseconds stage_latency,
                            long double bytes_per_ns);
} // namespace fake_cuda
#endif
