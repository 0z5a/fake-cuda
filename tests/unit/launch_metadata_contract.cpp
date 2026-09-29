#include "impl/core.h"
#include "impl/scheduler.h"
#include "impl/virtual_work.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); std::exit(1); \
} } while (0)
#define OK(expr) CHECK((expr) == CUDA_SUCCESS)

static std::shared_ptr<const fake_cuda::KernelLaunch> last_launch() {
    auto op = fake_cuda::virtual_core_device(0)->compute_queue.last_operation();
    CHECK(op && op->launch);
    return op->launch;
}
int main() {
    // Overflow checks still apply when a partial profile has no dimension limits.
    fake_cuda::DeviceProfile unknown;
    fake_cuda::KernelLaunch overflow{nullptr, {UINT32_MAX, UINT32_MAX, UINT32_MAX}, {1, 1, 1}, 0,
                                    fake_cuda::ParameterEncoding::unknown_layout, {}};
    CHECK(overflow.validate(unknown) == CUDA_ERROR_INVALID_VALUE);
    overflow.grid = {1, 1, 1};
    overflow.block = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    CHECK(overflow.validate(unknown) == CUDA_ERROR_INVALID_VALUE);
    fake_cuda::DeviceProfile limited;
    limited.attributes.emplace(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 64);
    limited.attributes.emplace(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 32);
    overflow.block = {32, 2, 1};
    CHECK(overflow.validate(limited) == CUDA_SUCCESS);
    overflow.block = {33, 1, 1};
    CHECK(overflow.validate(limited) == CUDA_ERROR_INVALID_VALUE);
    overflow.block = {32, 3, 1};
    CHECK(overflow.validate(limited) == CUDA_ERROR_INVALID_VALUE);
    OK(core_init(0));
    CUcontext context; CUstream stream;
    OK(core_context_create(&context, 0, 0));
    OK(core_stream_create(&stream, CU_STREAM_NON_BLOCKING, 0));
    CUmodule module; CUfunction function, again;
    OK(virtual_module_load_data(&module, "opaque image"));
    OK(virtual_module_get_function(&function, module, "contract_kernel"));
    OK(virtual_module_get_function(&again, module, "contract_kernel"));
    CHECK(function == again);
    CUmodule other; CUfunction other_function;
    OK(virtual_module_load_data(&other, "opaque image"));
    OK(virtual_module_get_function(&other_function, other, "contract_kernel"));
    CHECK(function != other_function);
    struct Arguments { CUdeviceptr pointer; std::uint64_t scalar; } args{0x100000000000ULL, 7};
    size_t bytes = sizeof(args);
    void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &args,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes, CU_LAUNCH_PARAM_END};
    OK(virtual_launch_kernel(function, 3, 2, 1, 32, 2, 1, 1024, stream, nullptr, extra));
    const auto eager = last_launch();
    CHECK((eager->grid == std::array<unsigned int, 3>{3, 2, 1}));
    CHECK((eager->block == std::array<unsigned int, 3>{32, 2, 1}));
    CHECK(eager->dynamic_shared_bytes == 1024);
    CHECK(eager->kernel->symbol == "contract_kernel");
    CHECK(eager->kernel->load_id == reinterpret_cast<std::uintptr_t>(module));
    CHECK(eager->parameters == fake_cuda::ParameterEncoding::packed_buffer);
    CHECK(eager->packed_parameters.size() == bytes);
    CHECK(std::memcmp(eager->packed_parameters.data(), &args, bytes) == 0);
    args.scalar = 9;
    CHECK(std::memcmp(eager->packed_parameters.data(), &args, bytes) != 0);

    CUgraph graph; CUgraphExec exec;
    OK(virtual_stream_begin_capture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    {
        Arguments captured{args.pointer, 11};
        void *capture_extra[] = {CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes,
            CU_LAUNCH_PARAM_BUFFER_POINTER, &captured, CU_LAUNCH_PARAM_END};
        OK(virtual_launch_kernel(function, 3, 2, 1, 32, 2, 1, 1024, stream, nullptr, capture_extra));
        captured.scalar = 99;
    }
    OK(virtual_stream_end_capture(stream, &graph));
    OK(virtual_graph_instantiate(&exec, graph));
    OK(virtual_graph_destroy(graph));
    for (int i = 0; i < 2; ++i) {
        OK(virtual_graph_launch(exec, stream));
        const auto replay = last_launch();
        CHECK(replay->kernel == eager->kernel);
        CHECK(replay->grid == eager->grid && replay->block == eager->block);
        CHECK(replay->dynamic_shared_bytes == eager->dynamic_shared_bytes);
        Arguments saved{};
        std::memcpy(&saved, replay->packed_parameters.data(), sizeof(saved));
        CHECK(saved.pointer == args.pointer && saved.scalar == 11);
    }
    void *params[] = {&args.pointer, &args.scalar};
    CHECK(virtual_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0, stream, params, extra) == CUDA_ERROR_INVALID_VALUE);
    CHECK(virtual_launch_kernel(function, 0, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE);
    CHECK(virtual_launch_kernel(function, 1, 1, 1, 1024, 2, 1, 0, stream, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE);
    CHECK(virtual_launch_kernel(function, 1, 1, 1, 1, 1, 1, 1000000, stream, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE);
    void *bad_extra[] = {CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes, CU_LAUNCH_PARAM_END};
    CHECK(virtual_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, bad_extra) == CUDA_ERROR_INVALID_VALUE);
    OK(virtual_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0, stream, params, nullptr));
    CHECK(last_launch()->parameters == fake_cuda::ParameterEncoding::unknown_layout);
    CHECK(last_launch()->packed_parameters.empty());
    CUlibrary library; CUkernel kernel; CUfunction library_function;
    OK(virtual_library_load_data(&library, "opaque image", nullptr, nullptr, 0, nullptr, nullptr, 0));
    OK(virtual_library_get_kernel(&kernel, library, "contract_kernel"));
    OK(virtual_kernel_get_function(&library_function, kernel));
    OK(virtual_launch_kernel(library_function, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
    const auto library_launch = last_launch();
    OK(virtual_launch_kernel(reinterpret_cast<CUfunction>(kernel), 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
    CHECK(last_launch()->kernel == library_launch->kernel);
    CHECK(library_launch->kernel->load_id != eager->kernel->load_id);
    CUmodule library_module; CUfunction through_module;
    OK(virtual_library_get_module(&library_module, library));
    OK(virtual_module_get_function(&through_module, library_module, "contract_kernel"));
    OK(virtual_launch_kernel(through_module, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
    CHECK(last_launch()->kernel->load_id == library_launch->kernel->load_id);
    CHECK(last_launch()->kernel->symbol == library_launch->kernel->symbol);
    OK(virtual_library_unload(library));
    CHECK(library_launch->kernel->symbol == "contract_kernel");
    CHECK(virtual_launch_kernel(library_function, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr) == CUDA_ERROR_INVALID_HANDLE);
    OK(virtual_graph_exec_destroy(exec));
    OK(virtual_module_unload(module));
    OK(virtual_module_unload(other));
    CHECK(eager->kernel->symbol == "contract_kernel");
    OK(core_context_destroy(context));
    std::puts("launch metadata contract passed");
}
