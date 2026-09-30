#ifndef FAKE_CUDA_IMPL_RESOURCE_ENGINE_H
#define FAKE_CUDA_IMPL_RESOURCE_ENGINE_H

#include "perf_model.h"

#include <map>

namespace fake_cuda {
// Capacities and demands use the same units; rates are units/ns. Resident
// resources are held for a phase, throughput resources are shared while running.
struct ResourceCapacity { long double throughput; std::uint64_t resident; };
struct ResourcePhase {
    Nanoseconds isolated_service;
    std::vector<long double> demand;
    std::vector<std::uint64_t> resident;
};
struct ResourceWork {
    std::uint64_t id;
    Nanoseconds arrival;
    std::vector<std::uint64_t> dependencies;
    std::vector<ResourcePhase> phases;
    long double weight = 1;
};
enum class WorkState { pending, ready, running, completed };
struct CompletionToken {
    std::uint64_t generation = 0;
    std::optional<Nanoseconds> deadline;
    bool completed = false;
};
struct ResourceProgress {
    WorkState state = WorkState::pending;
    std::optional<Nanoseconds> ready, start, completion;
    size_t phase = 0;
    long double remaining = 1, rate = 0;
    CompletionToken token;
};
class ResourceEngine {
public:
    explicit ResourceEngine(std::vector<ResourceCapacity> capacities);
    void submit(ResourceWork work);
    void advance_to(Nanoseconds time);
    std::optional<Nanoseconds> next_event() const;
    const ResourceProgress &progress(std::uint64_t id) const { return work_.at(id).progress; }
    bool current_token(std::uint64_t id, std::uint64_t generation) const;
    Nanoseconds now() const { return now_; }
private:
    struct Entry { ResourceWork work; ResourceProgress progress; };
    void settle();
    void rates();
    std::vector<ResourceCapacity> capacities_;
    std::map<std::uint64_t, Entry> work_;
    Nanoseconds now_{};
};

// Ordinary CTA features only. Allocated registers/shared bytes must come from
// an explicit architecture rule or native occupancy oracle, not raw r*T.
struct SmCapacity {
    std::uint64_t sms, blocks, threads, warps, registers, shared_bytes;
    std::uint64_t block_threads, block_shared_bytes;
};
struct CtaAllocation {
    std::uint64_t grid_ctas, threads, allocated_registers, allocated_shared_bytes;
    bool cluster = false;
};
struct Residency { std::uint64_t ctas_per_sm, waves; long double occupancy; };
std::optional<Residency> ordinary_residency(const SmCapacity &, const CtaAllocation &);
} // namespace fake_cuda
#endif
