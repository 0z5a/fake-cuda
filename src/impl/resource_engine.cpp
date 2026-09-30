#include "impl/resource_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fake_cuda {
ResourceEngine::ResourceEngine(std::vector<ResourceCapacity> capacities) : capacities_(std::move(capacities)) {
    for (auto c : capacities_)
        if (!std::isfinite(c.throughput) || c.throughput < 0 || (!c.throughput && !c.resident))
            throw std::invalid_argument("invalid resource capacity");
}
void ResourceEngine::submit(ResourceWork work) {
    if (!work.id || work_.contains(work.id) || work.arrival < now_ || work.phases.empty() ||
        !std::isfinite(work.weight) || work.weight <= 0)
        throw std::invalid_argument("invalid resource work");
    for (auto id : work.dependencies)
        if (!work_.contains(id)) throw std::invalid_argument("dependency must already be registered");
    for (const auto &p : work.phases) {
        if (p.isolated_service.count() <= 0 || p.demand.size() != capacities_.size() || p.resident.size() != capacities_.size())
            throw std::invalid_argument("invalid resource phase");
        for (size_t r = 0; r < capacities_.size(); ++r)
            if (!std::isfinite(p.demand[r]) || p.demand[r] < 0 || (p.demand[r] && !capacities_[r].throughput) ||
                p.resident[r] > capacities_[r].resident)
                throw std::invalid_argument("phase exceeds resource support");
    }
    const auto id = work.id;
    work_.emplace(id, Entry{std::move(work), {}});
    settle();
}
void ResourceEngine::settle() {
    for (auto &[id, e] : work_) {
        auto &p = e.progress;
        if (p.state == WorkState::running && p.token.deadline <= now_) {
            if (++p.phase == e.work.phases.size()) {
                p.state = WorkState::completed;
                p.completion = now_;
                p.token.completed = true;
                p.token.deadline.reset();
                ++p.token.generation;
            } else {
                p.state = WorkState::ready;
                p.remaining = 1;
            }
        }
    }
    std::vector<std::uint64_t> used(capacities_.size());
    for (const auto &[id, e] : work_)
        if (e.progress.state == WorkState::running)
            for (size_t r = 0; r < used.size(); ++r) used[r] += e.work.phases[e.progress.phase].resident[r];
    for (auto &[id, e] : work_) {
        auto &p = e.progress;
        if (p.state == WorkState::pending && e.work.arrival <= now_ &&
            std::all_of(e.work.dependencies.begin(), e.work.dependencies.end(), [&](auto dep) {
                return work_.at(dep).progress.state == WorkState::completed;
            })) {
            p.state = WorkState::ready;
            p.ready = now_;
        }
        if (p.state != WorkState::ready) continue;
        const auto &phase = e.work.phases[p.phase];
        bool fits = true;
        for (size_t r = 0; r < used.size(); ++r)
            fits &= phase.resident[r] <= capacities_[r].resident - used[r];
        if (!fits) continue;
        for (size_t r = 0; r < used.size(); ++r) used[r] += phase.resident[r];
        p.state = WorkState::running;
        if (!p.start) p.start = now_;
    }
    rates();
}
void ResourceEngine::rates() {
    std::vector<Entry *> active;
    std::map<std::uint64_t, long double> previous;
    for (auto &[id, e] : work_) {
        previous[id] = e.progress.rate;
        e.progress.rate = 0;
        if (e.progress.state == WorkState::running) active.push_back(&e);
    }
    std::vector<bool> free(active.size(), true);
    size_t remaining = active.size();
    while (remaining) {
        long double step = std::numeric_limits<long double>::infinity();
        for (size_t i = 0; i < active.size(); ++i) if (free[i]) {
            auto &e = *active[i];
            step = std::min(step, (1.L / e.work.phases[e.progress.phase].isolated_service.count() - e.progress.rate) / e.work.weight);
        }
        std::vector<long double> used(capacities_.size()), slope(capacities_.size());
        for (size_t i = 0; i < active.size(); ++i) {
            auto &e = *active[i];
            const auto &demand = e.work.phases[e.progress.phase].demand;
            for (size_t r = 0; r < used.size(); ++r) {
                used[r] += demand[r] * e.progress.rate;
                if (free[i]) slope[r] += demand[r] * e.work.weight;
            }
        }
        for (size_t r = 0; r < used.size(); ++r) if (slope[r])
            step = std::min(step, std::max(0.L, capacities_[r].throughput - used[r]) / slope[r]);
        for (size_t i = 0; i < active.size(); ++i) if (free[i]) active[i]->progress.rate += step * active[i]->work.weight;
        size_t frozen = 0;
        for (size_t i = 0; i < active.size(); ++i) if (free[i]) {
            auto &e = *active[i];
            const auto &phase = e.work.phases[e.progress.phase];
            bool saturated = e.progress.rate * phase.isolated_service.count() >= 1 - 1e-15L;
            for (size_t r = 0; r < used.size(); ++r)
                if (phase.demand[r] && used[r] + step * slope[r] >= capacities_[r].throughput * (1 - 1e-15L))
                    saturated = true;
            if (saturated) { free[i] = false; ++frozen; }
        }
        if (!frozen) throw std::logic_error("resource allocation made no progress");
        remaining -= frozen;
    }
    for (auto *e : active) {
        auto &p = e->progress;
        const auto cap = 1.L / e->work.phases[p.phase].isolated_service.count();
        if (std::abs(p.rate - cap) <= 32 * std::numeric_limits<long double>::epsilon() * cap) p.rate = cap;
        if (p.rate == previous.at(e->work.id) && p.token.deadline && *p.token.deadline > now_) continue;
        auto duration = p.remaining / p.rate;
        const auto integer = std::round(duration);
        const auto tolerance = std::min(1e-6L, 32 * std::numeric_limits<long double>::epsilon() * std::max(1.L, duration));
        if (std::abs(duration - integer) <= tolerance) duration = integer;
        const long double ticks = std::ceil(duration);
        if (ticks > std::numeric_limits<Nanoseconds::rep>::max() - now_.count())
            throw std::overflow_error("resource timeline overflow");
        const auto end = now_ + Nanoseconds(std::max<Nanoseconds::rep>(1, static_cast<Nanoseconds::rep>(ticks)));
        if (p.token.deadline != end) { p.token.deadline = end; ++p.token.generation; }
    }
}
std::optional<Nanoseconds> ResourceEngine::next_event() const {
    std::optional<Nanoseconds> next;
    auto include = [&](Nanoseconds t) { if (!next || t < *next) next = t; };
    for (const auto &[id, e] : work_) {
        if (e.progress.state == WorkState::pending && e.work.arrival > now_) include(e.work.arrival);
        if (e.progress.state == WorkState::running) include(*e.progress.token.deadline);
    }
    return next;
}
void ResourceEngine::advance_to(Nanoseconds target) {
    if (target < now_) throw std::invalid_argument("resource clock moved backwards");
    while (now_ < target) {
        const auto next = next_event();
        const auto time = next ? std::min(target, *next) : target;
        for (auto &[id, e] : work_) if (e.progress.state == WorkState::running)
            e.progress.remaining = std::max(0.L, e.progress.remaining - e.progress.rate * (time - now_).count());
        now_ = time;
        settle();
    }
}
bool ResourceEngine::current_token(std::uint64_t id, std::uint64_t generation) const {
    const auto &p = progress(id);
    return !p.token.completed && p.token.generation == generation;
}
std::optional<Residency> ordinary_residency(const SmCapacity &c, const CtaAllocation &a) {
    if (a.cluster || !c.sms || !c.blocks || !c.threads || !c.warps || !c.registers || !c.shared_bytes ||
        !a.grid_ctas || !a.threads || a.threads > c.block_threads || a.allocated_shared_bytes > c.block_shared_bytes)
        return {};
    const auto warps = (a.threads + 31) / 32;
    auto blocks = std::min({c.blocks, c.threads / a.threads, c.warps / warps});
    if (a.allocated_registers) blocks = std::min(blocks, c.registers / a.allocated_registers);
    if (a.allocated_shared_bytes) blocks = std::min(blocks, c.shared_bytes / a.allocated_shared_bytes);
    if (!blocks || c.sms > std::numeric_limits<std::uint64_t>::max() / blocks) return {};
    const auto width = c.sms * blocks;
    return Residency{blocks, 1 + (a.grid_ctas - 1) / width, static_cast<long double>(blocks * warps) / c.warps};
}
} // namespace fake_cuda
