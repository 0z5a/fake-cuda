#include "impl/resource_engine.h"
#include <iostream>
int main(int argc, char **argv) {
    if (argc != 13) return 2;
    std::uint64_t v[12];
    for (int i = 0; i < 12; ++i) v[i] = std::stoull(argv[i + 1]);
    const auto result = fake_cuda::ordinary_residency({v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]},
                                                    {v[8], v[9], v[10], v[11]});
    std::cout << (result ? result->ctas_per_sm : 0) << '\n';
}
