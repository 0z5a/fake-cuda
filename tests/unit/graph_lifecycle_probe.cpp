// Standalone CUDA 13 Driver ABI graph-lifecycle probe; links only libdl, not libcuda.
// Build: c++ -std=c++23 -Wall -Wextra -Werror -I.venv-cu130/lib/python3.12/site-packages/nvidia/cu13/include tests/unit/graph_lifecycle_probe.cpp -ldl -o /tmp/graph_lifecycle_probe
// Run:   /tmp/graph_lifecycle_probe build/libcuda.so.1
#include <cuda.h>
#include <dlfcn.h>
#include <signal.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>

#if CUDA_VERSION < 13000 || CUDA_VERSION >= 14000
#error "graph_lifecycle_probe requires CUDA 13 Driver headers"
#endif

namespace {
void *library = nullptr;

[[noreturn]] void fail(const char *what, int line, int actual = -1, int expected = -1) {
    std::fprintf(stderr, "graph_lifecycle_probe:%d: %s (actual %d, expected %d)\n",
                 line, what, actual, expected);
    std::exit(1);
}
#define CHECK(x) do { if (!(x)) fail(#x, __LINE__); } while (false)
#define EXPECT(x, code) do { CUresult r = (x); if (r != (code)) \
    fail(#x, __LINE__, static_cast<int>(r), static_cast<int>(code)); } while (false)
#define OK(x) EXPECT(x, CUDA_SUCCESS)

template <typename T> T symbol(const char *name) {
    dlerror();
    void *p = dlsym(library, name);
    if (const char *error = dlerror(); error || !p) {
        std::fprintf(stderr, "missing Driver ABI export %s: %s\n", name, error ? error : "null");
        std::exit(1);
    }
    return reinterpret_cast<T>(p);
}
#define API(name, ...) symbol<decltype(&::name)>(#name)(__VA_ARGS__)

void timed_out(int) {
    constexpr char message[] = "graph_lifecycle_probe: timed out\n";
    (void)write(STDERR_FILENO, message, sizeof(message) - 1);
    _exit(1);
}

void check_replay(CUgraphExec exec, CUstream launch, CUstream observer,
                  CUevent start, CUevent first, CUevent second, CUevent done) {
    OK(API(cuGraphLaunch, exec, launch));
    OK(API(cuEventRecord, done, launch));
    OK(API(cuStreamWaitEvent, observer, done, 0));
    OK(API(cuStreamSynchronize, observer));
    OK(API(cuStreamSynchronize, launch));
    OK(API(cuEventQuery, done));
    OK(API(cuEventQuery, first));
    OK(API(cuEventQuery, second));
    float milliseconds = 0;
    OK(API(cuEventElapsedTime_v2, &milliseconds, start, first));
    CHECK(milliseconds >= 5.0f); // Virtual kernel on the origin lane (10 ms model).
    OK(API(cuEventElapsedTime_v2, &milliseconds, first, second));
    CHECK(milliseconds >= 5.0f); // The imported lane's kernel follows the event wait.
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: graph_lifecycle_probe /path/to/libcuda.so.1\n");
        return 2;
    }
    CHECK(signal(SIGALRM, timed_out) != SIG_ERR);
    alarm(10);
    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    OK(API(cuInit, 0));
    CUdevice device = -1;
    OK(API(cuDeviceGet, &device, 0));
    CUcontext context = nullptr;
    OK(API(cuDevicePrimaryCtxRetain, &context, device));
    OK(API(cuCtxSetCurrent, context));
    CUstream a = nullptr, b = nullptr;
    OK(API(cuStreamCreate, &a, CU_STREAM_NON_BLOCKING));
    OK(API(cuStreamCreate, &b, CU_STREAM_NON_BLOCKING));
    CUevent start = nullptr, first = nullptr, second = nullptr, done = nullptr;
    OK(API(cuEventCreate, &start, CU_EVENT_DEFAULT));
    OK(API(cuEventCreate, &first, CU_EVENT_DEFAULT));
    OK(API(cuEventCreate, &second, CU_EVENT_DEFAULT));
    OK(API(cuEventCreate, &done, CU_EVENT_DEFAULT));

    // PTX is a virtual handle here; the simulator schedules kernels but never executes them.
    static constexpr char ptx[] = ".version 8.0\n.target sm_90\n.address_size 64\n"
                                  ".visible .entry lifecycle_kernel() { ret; }\n";
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    OK(API(cuModuleLoadData, &module, ptx));
    OK(API(cuModuleGetFunction, &kernel, module, "lifecycle_kernel"));

    OK(API(cuStreamBeginCapture_v2, a, CU_STREAM_CAPTURE_MODE_GLOBAL));
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
    cuuint64_t capture_id = 0;
    CUgraph capture_graph = nullptr;
    const CUgraphNode *dependencies = nullptr;
    const CUgraphEdgeData *edges = nullptr;
    size_t dependency_count = 0;
    OK(API(cuStreamGetCaptureInfo_v3, a, &status, &capture_id, &capture_graph,
           &dependencies, &edges, &dependency_count));
    CHECK(status == CU_STREAM_CAPTURE_STATUS_ACTIVE && capture_id && capture_graph);
    CHECK(dependency_count == 0 && !dependencies && !edges);
    size_t count = 99;
    OK(API(cuGraphGetNodes, capture_graph, nullptr, &count));
    CHECK(count == 0);

    // A: start -> kernel -> first; B: wait(first) -> kernel -> second; A: wait(second).
    OK(API(cuEventRecord, start, a));
    OK(API(cuLaunchKernel, kernel, 1, 1, 1, 1, 1, 1, 0, a, nullptr, nullptr));
    OK(API(cuEventRecord, first, a));
    OK(API(cuStreamWaitEvent, b, first, 0));
    OK(API(cuLaunchKernel, kernel, 1, 1, 1, 1, 1, 1, 0, b, nullptr, nullptr));
    OK(API(cuEventRecord, second, b));
    OK(API(cuStreamWaitEvent, a, second, 0));
    OK(API(cuStreamGetCaptureInfo_v3, a, &status, &capture_id, &capture_graph,
           &dependencies, &edges, &dependency_count));
    CHECK(status == CU_STREAM_CAPTURE_STATUS_ACTIVE && dependency_count == 1 && dependencies && edges);
    constexpr size_t expected_nodes = 7;
    count = 0;
    OK(API(cuGraphGetNodes, capture_graph, nullptr, &count));
    CHECK(count == expected_nodes);
    std::array<CUgraphNode, expected_nodes> captured{};
    OK(API(cuGraphGetNodes, capture_graph, captured.data(), &count));
    CHECK(count == expected_nodes);
    for (CUgraphNode node : captured) CHECK(node != nullptr);
    CHECK(dependencies[0] == captured.back());

    CUgraph graph = nullptr;
    OK(API(cuStreamEndCapture, a, &graph));
    CHECK(graph == capture_graph);
    count = 0;
    OK(API(cuGraphGetNodes, graph, nullptr, &count));
    CHECK(count == expected_nodes);
    std::array<CUgraphNode, expected_nodes> finished{};
    OK(API(cuGraphGetNodes, graph, finished.data(), &count));
    CHECK(count == expected_nodes && finished == captured);

    CUgraphExec one = nullptr, two = nullptr, default_priority = nullptr;
    CUDA_GRAPH_INSTANTIATE_PARAMS params{};
    OK(API(cuGraphInstantiateWithParams, &one, graph, &params));
    CHECK(one && params.result_out == CUDA_GRAPH_INSTANTIATE_SUCCESS);
    OK(API(cuGraphInstantiateWithFlags, &default_priority, graph,
           CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY));
    OK(API(cuGraphExecDestroy, default_priority));
    OK(API(cuGraphInstantiateWithFlags, &two, graph, 0));
    CHECK(two && two != one);
    OK(API(cuGraphDestroy, graph));
    EXPECT(API(cuGraphGetNodes, graph, nullptr, &count), CUDA_ERROR_INVALID_HANDLE);

    // Both executions must own their captured nodes independently of the source graph.
    // Replay sequentially: captured event handles are reused, not concurrently raced.
    check_replay(one, a, b, start, first, second, done);
    check_replay(two, b, a, start, first, second, done);

    OK(API(cuGraphExecDestroy, one));
    OK(API(cuGraphExecDestroy, two));
    CUstream high;
    OK(API(cuStreamCreateWithPriority, &high, CU_STREAM_NON_BLOCKING, -1));
    OK(API(cuStreamBeginCapture_v2, high, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(API(cuLaunchKernel, kernel, 1, 1, 1, 1, 1, 1, 0, high, nullptr, nullptr));
    OK(API(cuStreamEndCapture, high, &graph));
    CUgraphExec unsupported = nullptr;
    EXPECT(API(cuGraphInstantiateWithFlags, &unsupported, graph,
               CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY), CUDA_ERROR_NOT_SUPPORTED);
    CHECK(unsupported == nullptr);
    OK(API(cuGraphDestroy, graph));
    OK(API(cuStreamDestroy_v2, high));
    OK(API(cuModuleUnload, module));
    OK(API(cuEventDestroy_v2, done));
    OK(API(cuEventDestroy_v2, second));
    OK(API(cuEventDestroy_v2, first));
    OK(API(cuEventDestroy_v2, start));
    OK(API(cuStreamDestroy_v2, b));
    OK(API(cuStreamDestroy_v2, a));
    OK(API(cuCtxSetCurrent, nullptr));
    OK(API(cuDevicePrimaryCtxRelease_v2, device));
    CHECK(dlclose(library) == 0);
    alarm(0);
    std::puts("PASS: two-stream capture, node lifetime, independent execs after graph destroy, event waits and synchronization");
}
