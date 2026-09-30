#include "impl/resource_engine.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace fake_cuda;
using namespace std::chrono_literals;
static void check(bool value, const char *expression, int line) {
    if (!value) { std::cerr << "FAIL line " << line << ": " << expression << '\n'; std::exit(1); }
}
#define require(expression) check((expression), #expression, __LINE__)
static ResourceWork serial(std::uint64_t id, Nanoseconds arrival, Nanoseconds service,
                           std::vector<std::uint64_t> deps = {}, size_t resource = 0) {
    std::vector<std::uint64_t> resident(2); resident[resource] = 1;
    return {id, arrival, std::move(deps), {{service, {0, 0}, resident}}};
}
int main() {
    // Q01: an earlier submitted operation waiting on a future event cannot
    // reserve the compute lane before an independent ready operation.
    ResourceEngine ready({{0, 1}, {0, 1}});
    ready.submit(serial(1, 0ns, 100ns, {}, 1));
    ready.submit(serial(2, 0ns, 10ns, {1}));
    ready.submit(serial(3, 0ns, 20ns));
    ready.advance_to(110ns);
    require(ready.progress(3).start == 0ns && ready.progress(3).completion == 20ns);
    require(ready.progress(2).ready == 100ns && ready.progress(2).completion == 110ns);
    // Q02/Q03: starts and finishes change rates and invalidate old completion
    // tokens; no immutable end timestamp may complete the delayed operation.
    ResourceEngine shared({{1, 0}});
    shared.submit({1, 0ns, {}, {{100ns, {100}, {0}}}});
    auto stale = shared.progress(1).token.generation;
    shared.advance_to(20ns);
    shared.submit({2, 20ns, {}, {{40ns, {40}, {0}}}, 2.5});
    require(!shared.current_token(1, stale));
    require(std::abs(shared.progress(1).rate * 100 + shared.progress(2).rate * 40 - 1) < 1e-12L);
    shared.advance_to(100ns);
    require(!shared.progress(1).completion && shared.progress(2).completion == 100ns);
    require(shared.progress(1).rate == .01L);
    shared.advance_to(140ns);
    require(shared.progress(1).completion == 140ns);
    // Phase transitions release residency and reschedule service, preserving
    // dependencies until the entire operation completes.
    ResourceEngine phases({{0, 1}, {0, 1}});
    phases.submit({1, 0ns, {}, {{10ns, {0, 0}, {1, 0}}, {20ns, {0, 0}, {0, 1}}}});
    phases.submit(serial(2, 0ns, 5ns));
    phases.submit(serial(3, 0ns, 5ns, {1}));
    phases.advance_to(35ns);
    require(phases.progress(2).start == 10ns && phases.progress(2).completion == 15ns);
    require(phases.progress(3).start == 30ns && phases.progress(3).completion == 35ns);
    // Weighted sharing and a separate bottleneck do not manufacture bandwidth.
    ResourceEngine weighted({{1, 0}, {2, 0}});
    weighted.submit({1, 0ns, {}, {{100ns, {100, 0}, {0, 0}}}, 3});
    weighted.submit({2, 0ns, {}, {{100ns, {100, 0}, {0, 0}}}, 1});
    weighted.submit({3, 0ns, {}, {{100ns, {0, 200}, {0, 0}}}, 1});
    require(std::abs(weighted.progress(1).rate / weighted.progress(2).rate - 3) < 1e-12L);
    require(weighted.progress(3).token.deadline == 100ns);
    // K03/K04: exact allocated-resource threshold; no occupancy time multiplier.
    SmCapacity sm{132, 32, 2048, 64, 65536, 228 * 1024, 1024, 228 * 1024};
    auto a = ordinary_residency(sm, {1024, 256, 20480, 65 * 1024});
    auto b = ordinary_residency(sm, {1024, 256, 24576, 65 * 1024});
    require(a && a->ctas_per_sm == 3 && a->waves == 3 && a->occupancy == .375L);
    require(b && b->ctas_per_sm == 2 && b->waves == 4 && b->occupancy == .25L);
    require(!ordinary_residency(sm, {1, 256, 65537, 0}));
    require(!ordinary_residency(sm, {1, 256, 0, 0, true}));
    std::cout << "PASS Q01/Q02/Q03/K03/K04: ready admission, shared rates, mutable completion, ordinary CTA features\n";
}
