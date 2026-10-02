// Real Driver oracle: bounded buffers, cross-stream graph replay and host-wall timing.
#include <cuda.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

namespace {
void *driver;
template<class T> T symbol(const char *name) {
    auto function = reinterpret_cast<T>(dlsym(driver, name));
    if (!function) { std::fprintf(stderr, "missing symbol: %s\n", name); std::exit(1); }
    return function;
}
void check(CUresult result, const char *call) {
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "%s returned %d\n", call, result); std::exit(1);
    }
}
#define API(name, ...) ([] { static auto function = symbol<decltype(&name)>(#name); return function; }())(__VA_ARGS__)
#define OK(name, ...) check(API(name, __VA_ARGS__), #name)
constexpr size_t elements = 4096, bytes = elements * sizeof(std::uint32_t);
constexpr char kernel[] = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry double_u32(.param .u64 input, .param .u64 output) {
    .reg .b32 %r<5>;
    .reg .b64 %p<5>;
    ld.param.u64 %p0, [input];
    ld.param.u64 %p1, [output];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mad.lo.u32 %r2, %r1, 128, %r0;
    mul.wide.u32 %p2, %r2, 4;
    add.u64 %p3, %p0, %p2;
    add.u64 %p4, %p1, %p2;
    ld.global.u32 %r3, [%p3];
    shl.b32 %r4, %r3, 1;
    st.global.u32 [%p4], %r4;
    ret;
}
)ptx";
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s /path/to/real/libcuda.so.1 [kernel.cubin]\n", argv[0]); return 1;
    }
    std::vector<char> image;
    if (argc == 3) {
        std::ifstream file(argv[2], std::ios::binary);
        if (!file) { std::fprintf(stderr, "cannot open kernel image: %s\n", argv[2]); return 1; }
        image.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (file.bad() || image.empty()) { std::fprintf(stderr, "cannot read kernel image: %s\n", argv[2]); return 1; }
    }
    driver = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!driver) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    OK(cuInit, 0);
    int count = 0; OK(cuDeviceGetCount, &count);
    if (!count) return 1;
    std::vector<CUcontext> contexts(count);
    for (int ordinal = 0; ordinal < count; ++ordinal)
        OK(cuCtxCreate_v4, &contexts[ordinal], nullptr, 0, ordinal);
    std::puts("| Source | Peer | Driver peer access | Enable/disable contract |\n| ---: | ---: | ---: | --- |");
    for (int source = 0; source < count; ++source) {
        OK(cuCtxSetCurrent, contexts[source]);
        for (int peer = 0; peer < count; ++peer) if (source != peer) {
            int available = 0; OK(cuDeviceCanAccessPeer, &available, source, peer);
            const CUresult enabled = API(cuCtxEnablePeerAccess, contexts[peer], 0);
            const CUresult expected = available ? CUDA_SUCCESS : CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
            if (enabled != expected) {
                std::fprintf(stderr, "peer %d -> %d: enable returned %d, expected %d\n",
                             source, peer, enabled, expected);
                return 1;
            }
            if (available) {
                if (API(cuCtxEnablePeerAccess, contexts[peer], 0) != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
                    return 1;
                OK(cuCtxDisablePeerAccess, contexts[peer]);
            }
            if (API(cuCtxDisablePeerAccess, contexts[peer]) != CUDA_ERROR_PEER_ACCESS_NOT_ENABLED)
                return 1;
            std::printf("| %d | %d | %d | PASS |\n", source, peer, available);
        }
    }
    for (CUcontext context : contexts) { OK(cuCtxDestroy_v2, context); }
    std::puts("\n| Device | Four changed-input replays | Eager µs | Graph µs | Speedup |\n"
              "| ---: | --- | ---: | ---: | ---: |");
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        CUcontext context; CUmodule module; CUfunction function; CUstream a, b;
        CUevent fork, join; CUdeviceptr input, output; void *host_input, *host_output;
        OK(cuCtxCreate_v4, &context, nullptr, 0, ordinal);
        OK(cuModuleLoadData, &module, image.empty() ? static_cast<const void *>(kernel) : image.data());
        OK(cuModuleGetFunction, &function, module, "double_u32");
        if (!image.empty()) {
            int binary_version = 0;
            OK(cuFuncGetAttribute, &binary_version, CU_FUNC_ATTRIBUTE_BINARY_VERSION, function);
            std::printf("Loaded cubin: device %d, binary version %d\n", ordinal, binary_version);
        }
        OK(cuStreamCreate, &a, CU_STREAM_NON_BLOCKING); OK(cuStreamCreate, &b, CU_STREAM_NON_BLOCKING);
        OK(cuEventCreate, &fork, CU_EVENT_DISABLE_TIMING); OK(cuEventCreate, &join, CU_EVENT_DISABLE_TIMING);
        OK(cuMemAlloc_v2, &input, bytes); OK(cuMemAlloc_v2, &output, bytes);
        OK(cuMemAllocHost_v2, &host_input, bytes); OK(cuMemAllocHost_v2, &host_output, bytes);
        auto *source = static_cast<std::uint32_t *>(host_input);
        auto *target = static_cast<std::uint32_t *>(host_output);
        auto update = [&](unsigned int iteration) {
            for (size_t i = 0; i < elements; ++i) source[i] = static_cast<unsigned int>(i) + iteration * 17;
        };
        auto validate = [&](const std::uint32_t *values, unsigned int iteration) {
            for (size_t i = 0; i < elements; ++i)
                if (values[i] != (static_cast<unsigned int>(i) + iteration * 17) * 2) {
                    std::fprintf(stderr, "device %d: wrong value at %zu\n", ordinal, i); std::exit(1);
                }
        };
        void *parameters[] = {&input, &output};
        auto eager = [&] {
            OK(cuMemcpyHtoDAsync_v2, input, source, bytes, a);
            OK(cuLaunchKernel, function, 32, 1, 1, 128, 1, 1, 0, a, parameters, nullptr);
            OK(cuEventRecord, fork, a); OK(cuStreamWaitEvent, b, fork, 0);
            OK(cuMemcpyDtoHAsync_v2, target, output, bytes, b);
            OK(cuEventRecord, join, b); OK(cuStreamWaitEvent, a, join, 0);
        };
        update(0);
        for (int i = 0; i < 5; ++i) eager();
        OK(cuStreamSynchronize, a); validate(target, 0);
        CUgraph graph; CUgraphExec executable;
        OK(cuStreamBeginCapture_v2, a, CU_STREAM_CAPTURE_MODE_GLOBAL);
        eager();
        OK(cuStreamEndCapture, a, &graph);
        OK(cuGraphInstantiateWithFlags, &executable, graph, 0);
        OK(cuGraphDestroy, graph);
        std::array<std::array<std::uint32_t, elements>, 4> saved;
        for (unsigned int i = 0; i < saved.size(); ++i) {
            update(i + 1);
            CUstream launch = i % 2 ? b : a;
            OK(cuGraphLaunch, executable, launch); OK(cuStreamSynchronize, launch);
            std::copy_n(target, elements, saved[i].begin());
            validate(target, i + 1);
        }
        for (unsigned int i = 0; i < saved.size(); ++i) validate(saved[i].data(), i + 1);
        std::array<std::vector<double>, 2> times;
        for (int round = 0; round < 4; ++round)
            for (int arm : round % 2 ? std::array{1, 0, 0, 1} : std::array{0, 1, 1, 0}) {
                auto submit = [&] { if (arm) { OK(cuGraphLaunch, executable, a); } else eager(); };
                for (int i = 0; i < 5; ++i) submit();
                OK(cuStreamSynchronize, a);
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 100; ++i) submit();
                OK(cuStreamSynchronize, a);
                times[arm].push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 100);
                validate(target, 4);
            }
        for (auto &samples : times) std::sort(samples.begin(), samples.end());
        const double eager_us = (times[0][3] + times[0][4]) / 2, graph_us = (times[1][3] + times[1][4]) / 2;
        std::printf("| %d | PASS | %.3f | %.3f | %.3f× |\n", ordinal, eager_us, graph_us, eager_us / graph_us);
        std::fflush(stdout);
        OK(cuGraphExecDestroy, executable); OK(cuModuleUnload, module);
        OK(cuMemFree_v2, input); OK(cuMemFree_v2, output);
        OK(cuMemFreeHost, host_input); OK(cuMemFreeHost, host_output);
        OK(cuEventDestroy_v2, fork); OK(cuEventDestroy_v2, join);
        OK(cuStreamDestroy_v2, a); OK(cuStreamDestroy_v2, b); OK(cuCtxDestroy_v2, context);
    }
    dlclose(driver);
}
