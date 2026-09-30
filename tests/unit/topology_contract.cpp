#include "impl/topology.h"
#include <iostream>
#include <stdexcept>
using namespace fake_cuda;
using namespace std::chrono_literals;
int main() {
    for (size_t count : {1, 2, 4, 8, 16, 24, 32}) {
        std::vector<GlobalDevice> devices;
        for (size_t index = 0; index < count; ++index)
            devices.push_back({"run/node" + std::to_string(index / 8), "gpu" + std::to_string(index % 8)});
        Topology configured(devices, {}, {});
        const auto node = count > 8 ? "run/node1" : "run/node0";
        auto visible = visible_devices(devices, node, {0});
        if (visible[0] != devices[count > 8 ? 8 : 0]) throw std::runtime_error("local/global identity mismatch");
        if (count >= 4 && visible_devices(devices, "run/node0", {3, 1}) != std::vector<GlobalDevice>{devices[3], devices[1]})
            throw std::runtime_error("process-local reorder changed identity");
        try { visible_devices(devices, node, {8}); throw std::runtime_error("invalid local ordinal accepted"); }
        catch (const std::invalid_argument &) {}
    }
    GlobalDevice a{"node0", "gpu0"}, b{"node0", "gpu1"}, c{"node1", "gpu0"};
    Topology topology({a, b, c}, {{"pcie-upstream", 1}, {"nic", 2}}, {{a, b, {0}}, {a, c, {0, 1}}});
    ResourceEngine engine(topology.capacities());
    engine.submit({1, 0ns, {}, {topology.transfer(a, b, 100)}});
    engine.submit({2, 0ns, {}, {topology.transfer(a, c, 100)}});
    engine.advance_to(200ns);
    if (engine.progress(1).completion != 200ns || engine.progress(2).completion != 200ns ||
        ring_all_reduce(1, 100, 10ns, 1) != 0ns || ring_all_reduce(4, 100, 10ns, 1) != 210ns)
        throw std::runtime_error("topology contract failed");
    std::cout << "PASS H01/H02/N02: 1/2/4 and 8N layouts, explicit local/global mapping, shared path capacity, ring units\n";
}
