#include "impl/resource_engine.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace fake_cuda;
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const auto ranks = std::stoul(argv[1]);
    if (!ranks || ranks > 1024) return 2;
    ResourceEngine engine(std::vector<ResourceCapacity>(ranks, {0, 1}));
    std::string line;
    // A small embedding protocol, not a Driver export or network service.
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        if (command == "submit") {
            std::uint64_t id, rank; Nanoseconds::rep arrival, cost;
            if (!(input >> id >> rank >> arrival >> cost) || rank >= ranks) return 2;
            std::vector<std::uint64_t> resident(ranks); resident[rank] = 1;
            engine.submit({id, Nanoseconds(arrival), {}, {{Nanoseconds(cost), std::vector<long double>(ranks), resident}}});
            std::cout << "ok\n";
        } else if (command == "advance") {
            Nanoseconds::rep target;
            if (!(input >> target)) return 2;
            engine.advance_to(Nanoseconds(target));
            std::cout << "ok\n";
        } else if (command == "next") {
            auto event = engine.next_event();
            std::cout << (event ? event->count() : -1) << '\n';
        } else if (command == "status") {
            std::uint64_t id;
            if (!(input >> id)) return 2;
            const auto &p = engine.progress(id);
            std::cout << (p.start ? p.start->count() : -1) << ' ' << (p.completion ? p.completion->count() : -1) << '\n';
        } else return 2;
        std::string extra;
        if (input >> extra) return 2;
        std::cout.flush();
    }
}
