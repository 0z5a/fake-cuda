# Driver API unit / contract tests

Build and run with `cmake -S . -B build && cmake --build build -j4 && ctest --test-dir build --output-on-failure`.

All probes compile as C++23 against CUDA Driver headers and load this project's `libcuda.so.1` through `dlopen`; they do not link the CUDA Runtime or require a physical GPU. The C ABI bridge remains C. `driver_probe` is built separately with CUDA 13 and, if installed, CUDA 12 headers to catch ABI/version-selection regressions.

- `api_contract.cpp`: one-to-one inventory of the concrete Driver exports; resolves them via `dlsym` and `cuGetProcAddress`, directly invokes every entry point, and checks representative success, invalid-input, unsupported, and lifetime behavior by API category. Historical ABI variants (`cuCtxCreate_v2`, `cuEventElapsedTime`, `cuGetProcAddress`, `cuDeviceGetUuid`, and `cuStreamGetCaptureInfo`) are exercised via explicitly typed symbols. CUDA 12/13 capture-info resolver boundaries and virtual library/kernel handle lifetimes are tested. Unknown private export-table UUIDs must fail; undocumented Runtime UUID protocols are not treated as a stable public API.
- `driver_probe.cpp`: CUDA 12/13 device, primary-context, and resolver compatibility.
- `virtual_driver_probe.cpp`: queue delays, copies, allocations, events, kernels, capture and replay.
- `virtual_multistream_probe.cpp`: stream ordering, barriers, and multi-stream capture dependencies.
- `multidevice_probe.cpp`: context isolation, cross-device events, P2P transfers (runs with `FAKE_CUDA_DEVICE_COUNT=2`).
- `virtual_thread_probe.cpp`: concurrent submission and context retirement.

A passing copy/launch test checks address validation and virtual scheduling **only**: device memory stores no bytes, and CUDA kernels do not execute. When adding an exported Driver API, add a direct invocation and relevant error/lifetime assertions to `api_contract.cpp`, then a focused scheduling test when needed.

`semantic_contract.py` checks operator ledgers, fresh Graph bindings, chunk prefill and single-charge fusion coverage. `serving_contract.py` checks FCFS/chunk/cancellation timing, per-card pools and request/token conservation. `ep_contract.py` uses `resource_engine_bridge --resources` for shared link/SM capacity, held chunk buffers, remaining-work generations and discrete combined costs. CMake registers all three; the no-kill `tests/integration/tooling/run_contracts.py /path/to/build` executes registered commands through natural exit without CTest timeout termination.

`model_functional_contract.py` checks the evidence verifier with temporary CPU-only fixtures. Incomplete summaries, summaries copied from another policy and changed native samples must fail with saved replay-integrity differences. Removing cancelled-request output must retain strict token failure. These fixtures test verification, not numerical model inference or execution of the original Scheduler.
