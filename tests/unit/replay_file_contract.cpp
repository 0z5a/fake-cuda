#include "impl/core.h"
#include "impl/replay_file.h"
#include "impl/scheduler.h"
#include "impl/virtual_work.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); std::exit(1); \
} } while (0)
#define OK(expr) CHECK((expr) == CUDA_SUCCESS)
using namespace fake_cuda;
using namespace fake_cuda::detail;
using namespace std::chrono_literals;

int main(int argc, char **argv) {
    CHECK(argc == 2);
    std::ifstream file(argv[1]);
    CHECK(file.good());
    const std::string text{std::istreambuf_iterator<char>(file), {}};
    OK(core_init(0));
    CUcontext context; CUstream stream;
    OK(core_context_create(&context, 0, 0));
    OK(core_stream_create(&stream, CU_STREAM_NON_BLOCKING, 0));
    CUmodule module; CUfunction function;
    OK(virtual_module_load_data(&module, "contract code, explicit binding only"));
    OK(virtual_module_get_function(&function, module, "timing_contract"));
    size_t size = 0;
    void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, nullptr,
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
    auto launch = [&] { return virtual_launch_kernel(function, 1, 1, 1, 32, 1, 1, 0, stream, nullptr, extra); };
    OK(launch());
    auto *device = virtual_core_device(0);
    auto original = device->compute_queue.last_operation();
    CHECK(original && original->launch);
    ReplayBinding binding{"case0", "contract-code-v1", "contract-hardware-v1", "fixed-empty-arguments",
                          context, 0, &device->profile(), *original->launch};
    auto load = [&](const std::string &data) {
        std::istringstream input(data);
        return load_measured_replay(input, std::span(&binding, 1));
    };
    auto loaded = load(text);
    CHECK(loaded.model && loaded.error.empty());
    auto &s = scheduler();
    s.performance_model = std::move(loaded.model);
    OK(launch());
    auto eager = device->compute_queue.last_operation();
    CHECK(eager && eager->end - eager->start == 7ms);
    CHECK(eager->prediction->source == "synthetic replay-file contract");
    const auto &identity = eager->prediction->measurement;
    CHECK(identity && identity->binding == "case0" && identity->sample_index == 0);
    CHECK(identity->code == binding.code_identity && identity->hardware == binding.hardware_identity);
    CHECK(identity->conditions == binding.conditions);
    CUgraph graph; CUgraphExec exec;
    OK(virtual_stream_begin_capture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    OK(launch()); OK(launch());
    OK(virtual_stream_end_capture(stream, &graph));
    OK(virtual_graph_instantiate(&exec, graph));
    OK(virtual_graph_destroy(graph));
    OK(virtual_graph_launch(exec, stream));
    auto replay = device->compute_queue.last_operation();
    CHECK(replay && replay->end - replay->start == 13ms);
    CHECK(replay->prediction->measurement->sample_index == 2);
    CHECK(s.timing.service_time == 41ms); // 10 ms default + 7 + 11 + 13.
    CHECK(virtual_graph_launch(exec, stream) == CUDA_ERROR_NOT_SUPPORTED);

    size_t rejected = 0;
    auto reject = [&](const std::string &data) {
        auto result = load(data);
        CHECK(!result.model && !result.error.empty());
        CHECK(result.error.starts_with("line "));
        ++rejected;
    };
    auto replace = [&](const std::string &from, const std::string &to) {
        auto data = text;
        const auto position = data.find(from);
        CHECK(position != data.npos);
        data.replace(position, from.size(), to);
        return data;
    };
    for (const auto &[from, to] : std::vector<std::pair<std::string, std::string>>{
        {"schema=1", "schema=2"}, {"source=synthetic replay-file contract", "source="},
        {"scope=kernel", "scope=whole_forward"}, {"included_costs=device_service", "included_costs=end_to_end"},
        {"schema=1", "schema=1\nschema=1"}, {"schema=1", "unknown=1"},
        {"schema=1\n", ""}, {"scope=kernel", "scope kernel"},
        {"case0", "missing"}, {"contract-code-v1", "other-code"},
        {"contract-hardware-v1", "other-hardware"}, {"fixed-empty-arguments", "different-cache-state"},
        {"timing_contract", "different-symbol"}, {"\teager\t", "\tunknown\t"},
        {"\t1,1,1\t", "\t2,1,1\t"}, {"\t32,1,1\t", "\t64,1,1\t"},
        {"\t0\t-\t", "\t16\t-\t"}, {"\t-\t", "\t00\t"},
        {"\t1,1,1\t", "\t0,1,1\t"}, {"\t1,1,1\t", "\t1,1\t"},
        {"\t32,1,1\t", "\t4294967296,1,1\t"}, {"\t0\t-\t", "\t-1\t-\t"},
        {"\t-\t", "\tx0\t"}, {"\t-\t", "\t0\t"},
        {"7000000", "-1"}, {"7000000", "9223372036854775808"},
        {"7000000", "9223372036854775807"}, {"7000000", "7ms"},
        {"7000000", "7\textra"}, {"7000000", ""}})
        reject(replace(from, to));
    reject("");
    reject(text.substr(0, text.find("sample=")));
    reject(text + "source=late header\n");
    reject(text + "sample=incomplete\n"); // No partially loaded provider escapes.
    auto crlf = text;
    for (size_t i = 0; (i = crlf.find('\n', i)) != crlf.npos; i += 2) crlf.insert(i, "\r");
    CHECK(load(crlf).model);
    auto zero = load(replace("7000000", "0"));
    CHECK(zero.model);
    std::istringstream duplicate_input(text);
    const std::array duplicate{binding, binding};
    CHECK(!load_measured_replay(duplicate_input, duplicate).model);
    std::istringstream failed_input(text);
    failed_input.setstate(std::ios::badbit);
    CHECK(!load_measured_replay(failed_input, std::span(&binding, 1)).model);
    auto saved = binding;
    binding.code_identity.clear();
    reject(text);
    binding = saved;
    binding.launch.parameters = ParameterEncoding::unknown_layout;
    reject(text);
    binding = saved;
    // Binary packed bytes, including zero and 0xff, survive text loading unchanged.
    binding.launch.packed_parameters = {std::byte{0}, std::byte{255}};
    auto binary_text = replace("\t-\t", "\t00ff\t");
    binary_text.resize(binary_text.find("\nsample=", binary_text.find("sample=")));
    auto binary = load(binary_text);
    CHECK(binary.model);
    OK(virtual_graph_exec_destroy(exec));
    OK(virtual_module_unload(module));
    OK(core_context_destroy(context));
    std::printf("PASS: file-backed eager/graph replay, provenance and %zu rejected inputs\n", rejected);
}
