#include "impl/resource_engine.h"

#include <iostream>
#include <memory>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using namespace fake_cuda;
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const bool explicit_resources = std::string(argv[1]) == "--resources";
    size_t resources = explicit_resources ? 0 : std::stoul(argv[1]);
    if (!explicit_resources && (!resources || resources > 1024)) return 2;
    auto engine = explicit_resources ? nullptr : std::make_unique<ResourceEngine>(std::vector<ResourceCapacity>(resources, {0, 1}));
    std::cout << std::setprecision(20);
    std::string line;
    // A small embedding protocol, not a Driver export or network service.
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        if (command == "capacities" && explicit_resources && !engine) {
            if (!(input >> resources) || !resources || resources > 65536) return 2;
            std::vector<ResourceCapacity> capacities(resources);
            for (auto &capacity : capacities)
                if (!(input >> capacity.throughput >> capacity.resident)) return 2;
            engine = std::make_unique<ResourceEngine>(std::move(capacities));
            std::cout << "ok\n";
        } else if (!engine) return 2;
        else if (command == "work" && explicit_resources) {
            ResourceWork work; Nanoseconds::rep arrival;
            size_t count;
            if (!(input >> work.id >> arrival >> work.weight >> count) || count > 65536) return 2;
            work.arrival = Nanoseconds(arrival);
            work.dependencies.resize(count);
            for (auto &id : work.dependencies) if (!(input >> id)) return 2;
            work.held.resize(resources);
            for (auto &held : work.held) if (!(input >> held)) return 2;
            if (!(input >> count) || !count || count > 65536) return 2;
            work.phases.resize(count);
            for (auto &phase : work.phases) {
                Nanoseconds::rep duration;
                if (!(input >> duration)) return 2;
                phase.isolated_service = Nanoseconds(duration);
                phase.demand.resize(resources); phase.resident.resize(resources);
                for (auto &demand : phase.demand) if (!(input >> demand)) return 2;
                for (auto &resident : phase.resident) if (!(input >> resident)) return 2;
            }
            engine->submit(std::move(work));
            std::cout << "ok\n";
        } else if (command == "submit" && !explicit_resources) {
            std::uint64_t id, rank; Nanoseconds::rep arrival, cost;
            if (!(input >> id >> rank >> arrival >> cost) || rank >= resources) return 2;
            std::vector<std::uint64_t> resident(resources); resident[rank] = 1;
            engine->submit({id, Nanoseconds(arrival), {}, {{Nanoseconds(cost), std::vector<long double>(resources), resident}}});
            std::cout << "ok\n";
        } else if (command == "advance") {
            Nanoseconds::rep target;
            if (!(input >> target)) return 2;
            engine->advance_to(Nanoseconds(target));
            std::cout << "ok\n";
        } else if (command == "next") {
            auto event = engine->next_event();
            std::cout << (event ? event->count() : -1) << '\n';
        } else if (command == "status") {
            std::uint64_t id;
            if (!(input >> id)) return 2;
            const auto &p = engine->progress(id);
            std::cout << (p.start ? p.start->count() : -1) << ' ' << (p.completion ? p.completion->count() : -1) << '\n';
        } else if (command == "progress") {
            std::uint64_t id;
            if (!(input >> id)) return 2;
            const auto &p = engine->progress(id);
            std::cout << static_cast<int>(p.state) << ' ' << p.phase << ' ' << p.remaining << ' ' << p.rate << ' '
                      << p.token.generation << ' ' << (p.token.deadline ? p.token.deadline->count() : -1) << '\n';
        } else return 2;
        std::string extra;
        if (input >> extra) return 2;
        std::cout.flush();
    }
}
