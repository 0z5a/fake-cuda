# Fake CUDA Driver

`fake_cuda` builds `libcuda.so.1`, a GPU-free CUDA **Driver API** simulator targeting CUDA 13 with tested CUDA 12.8 client compatibility. Its C entry points feed a C++23 virtual device and scheduler. It is intended for studying CUDA control flow and timing, not for computing tensor values.

## Architecture

```text
Driver API calls
      |
      v
Device 0 (selected profile)                         Device 1 ... (optional)
  |                                              |
  +-- primary Context / other Contexts          +-- Contexts
  |     +-- Stream A                             |     +-- Streams
  |     +-- Stream B                             |
  |     `-- ...                                  `-- device-owned resources
  |
  `-- device-owned simulated resources (shared by all its contexts/streams)
        +-- h2d_queue             host -> device transfers
        +-- d2h_queue             device -> host transfers
        +-- compute_queue         kernels, local D2D, memset
        `-- p2p_queues[peer]       outgoing device -> peer transfers
```

A **context** associates work with a device; a **stream** determines operation order. Streams are *not* execution queues: submitted work goes to the appropriate device resource queue while retaining its stream and event dependencies. Independent queues can overlap; synchronizing one stream waits for that stream's work, not unrelated streams. `Capture` records graph dependencies, `Graph` holds a completed definition and replays it, and `GraphManager` owns capture sessions and graph/executable handles. These queues are a **simulation model**, not a claim about the number of physical GH200 hardware queues.

**CUDA ordering versus this simulator:** CUDA preserves in-stream order and explicit event/default-stream dependencies, but does not guarantee FCFS order or concurrency between independent streams. Stream priority (lower number means higher priority) is a non-preemptive *hint* for preferentially launching compute kernels when possible, not a guarantee of execution order; NVIDIA states that it does not affect H2D/D2H copies. Our resource queues currently reserve work in submission order without using stream priority, as a deterministic **timing heuristic**, not a model of CUDA hardware arbitration. [Driver stream priorities](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html) · [default-stream synchronization](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html).

One virtual GH200-profile device is exposed by default. `FAKE_CUDA_DEVICE_COUNT=1..1024` selects more independent virtual devices; the P2P queues model outgoing transfers between them. H2D/D2H delays depend on transfer size and configurable virtual bandwidth; kernels currently cost 10 ms of virtual time.

## Device profiles and launch records

`FAKE_CUDA_PROFILE=/absolute/path/device.profile` replaces the default capability snapshot for all virtual devices. Configure it before loading the Driver. Device name, attributes, total/free memory, allocation admission and launch dimension ceilings share this profile. Device storage grows with the configured count; outgoing P2P queues are created on use. The count limit of 1024 is a simulator configuration bound. UUIDs distinguish ordinals within one process, not hosts across a cluster.

Profiles are UTF-8 `key=value` files with `schema=1`, `name`, `source`, `memory_bytes`, and optional `attribute.<CUDA enum number>` entries. No whitespace trimming or escaping is applied; blank lines and `#` comments are allowed. Memory uses bytes and attributes retain CUDA's documented units (for example, clock rate is kHz). Missing attributes known to the build headers return `CUDA_ERROR_NOT_SUPPORTED`; other unknown enum values return `CUDA_ERROR_INVALID_VALUE`. Explicitly supplied numeric attributes can come from newer CUDA headers; malformed files/counts make `cuInit` and `cuDeviceGetCount` return `CUDA_ERROR_INVALID_VALUE`. The [synthetic fixture](tests/fixtures/synthetic.profile) is for contracts only.

The build includes a read-only collector for a real Driver (queries attributes known to its build headers):

```sh
build/capture_device_profile /path/to/real/libcuda.so.1 0 > device.profile
FAKE_CUDA_DEVICE_COUNT=4 FAKE_CUDA_PROFILE="$PWD/device.profile" your-command
```

For per-device profiles and measured direct peer access, use `FAKE_CUDA_SYSTEM=/absolute/path/devices.system` instead. It is mutually exclusive with `FAKE_CUDA_PROFILE` and `FAKE_CUDA_DEVICE_COUNT`. The device entries determine the count and must cover ordinals 0 through N−1; paths resolve relative to the manifest. Every directed non-self pair must be specified, including unavailable pairs. Missing pairs are rejected, not assumed connected.

```text
schema=1
source=measured direct peer access
device.0=device-0.profile
device.1=device-1.profile
peer.0.1=0
peer.1.0=0
```

Capture `device-N.profile` for each ordinal using the command above, then run `build/capture_device_profile /path/to/real/libcuda.so.1 --system > devices.system` beside those files. `cuDeviceCanAccessPeer` and `cuCtxEnablePeerAccess` use this directed matrix; an unavailable direction returns `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` when enabled. [CUDA peer-access semantics](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__PEER__ACCESS.html).

These files describe capabilities and direct peer access, not transport routes, measured bandwidth or runtime operating state. Peer-copy timing still uses the existing outgoing-queue heuristic, without modeling host staging or NUMA contention. Legacy configuration retains the synthetic all-to-all peer matrix for compatibility.

Eager and captured kernels now retain the same immutable launch record: load identity and symbol, grid/block dimensions, dynamic shared memory, and an owned copy of an explicitly packed `extra` parameter buffer. Library kernels keep the same load identity through context-specific modules. Captured records survive temporary host argument storage and source graph destruction. [CUDA launch parameter conventions](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html).

Images remain opaque: `kernelParams` has an unknown layout and is never retained or dereferenced without an ABI. Static resources, parameter types, pointed-to allocation generations and content hashes are not decoded yet. Packed snapshots copy parameter bytes, not tensor data. Load IDs are process-local identities, not predictor cache keys. The default kernel duration remains the 10 ms heuristic.

A standalone real-Driver oracle, `build/driver_graph_contract /path/to/real/libcuda.so.1`, checks peer query/enable/disable state, cross-stream capture, changed-input replay values and eager/graph timing on every visible GPU. It uses bounded pinned buffers and requires CUDA 13 Driver entry points. Results cover [four RTX 5060 Ti devices](results/2026-09-29/four-device-validation.md) and [cross-validation on four A100 SXM4 devices](results/2026-09-29/a100-cross-validation.md).

## Kernel service-time providers

The internal C++ `PerformanceModel` interface supplies one `PredictorResult` per kernel invocation, shared by eager submission and graph replay. Results declare duration in nanoseconds, accounting scope, included costs, confidence kind and source. Each query has a nonzero invocation ID; its result must name that same ID as its sole accounting owner. Duplicate, missing, reordered or foreign coverage rejects the whole batch. Each graph replay receives fresh IDs. Only kernel device-service costs are accepted here; whole-forward and end-to-end estimates are rejected to prevent charging their host/communication work again. Transfer timing remains the existing bandwidth model.

`SyntheticConstant` preserves `synthetic_constant_v1` at 10 ms. `MeasuredReplay` is a finite ordered oracle with explicit bindings to this process's context, device/profile, module load and symbol, grid/block, dynamic shared memory, packed parameters and eager/graph mode. A missing, reordered or mismatched invocation returns `CUDA_ERROR_NOT_SUPPORTED`, with a reason retained by the scheduler. Unknown parameter layouts are rejected. It does not match kernels by symbol alone or infer measurements from opaque images.

An embedding harness installs the provider through `Scheduler::performance_model` under its mutex. `load_measured_replay(input, bindings)` reads an ordered timing file and returns either a complete provider or a line-numbered error. Each `ReplayBinding` pairs an external binding ID, code identity, hardware identity and measurement conditions with the current context/device/profile and owned launch descriptor. The loader matches every row against those bindings. The harness must verify these identities against its known code bytes, hardware and conditions, including input contents and cache state; the shim cannot derive them from opaque handles or inspect pointed-to data. There is no environment selector or framework adapter.

Timing files use the following four required headers, followed by one `sample=` row per invocation:

```text
schema=1
source=measurement provenance
scope=kernel
included_costs=device_service
```

Each row has exactly eleven tab-separated fields: binding ID, code identity, hardware identity, conditions, symbol, mode (`eager` or `graph_replay`), grid (`x,y,z`), block (`x,y,z`), dynamic shared bytes, packed parameter hex (`-` for empty), and service nanoseconds. See the [synthetic format fixture](tests/fixtures/kernel.timing). Blank lines, `#` comments, LF and CRLF are accepted; no whitespace trimming or escaping is applied. Headers precede samples. Unknown/duplicate headers, incomplete bindings, identity/metadata mismatches, invalid durations and read errors reject the entire file, including when earlier rows were valid.

Loaded predictions retain the source, binding/code/hardware/condition identities and zero-based sample index on the scheduled operation. Runtime invocation matching, graph preflight and finite-sample exhaustion still apply after loading. This makes file reuse explicit; matching caller-supplied identity strings is not independent proof of hardware calibration. [File-loader validation and timings](results/2026-09-30/replay-file.md).

Capture stores metadata without querying or charging the provider. Replay predicts all kernel nodes once, uses those same results in temporal preflight and submission, and consumes samples only after successful scheduling. A failed preflight neither advances the replay cursor nor charges kernel service. Each scheduled kernel retains its prediction and provenance.

`TimingLedger` separates the sum of committed kernel service durations from predictor query wall time. Target-host time is explicitly unknown until a host model exists. The service sum is not elapsed makespan and excludes transfers; query wall time is not total simulator overhead. Scheduling remains paced with `steady_clock`, so slow queries can delay wall-clock submission even though their cost is not added to kernel service. This is not yet coordinated offline time. [Contract checks and before/after timings](results/2026-09-30/predictor-contract.md).

For a bounded diagnostic, `replay_kernel_trace(model, trace)` provides **fixed-trace offline replay**. The caller supplies a topologically ordered kernel trace with invocation IDs, normalized stream labels, explicit earlier dependencies, arrival times and target-host service durations. One host submission lane advances through those declared CPU costs; one serialized compute queue per device consumes kernel predictions. Independent devices can overlap. Every returned interval records host start/submission, dependency-ready time, device start/completion and prediction provenance. No target timestamp comes from wall time.

Its ledger reports kernel service sum, explicit host-service sum, and replay-call wall time; predictor query time is a subset of that last value and must not be added again. Makespan is computed from dependencies/resources, not by adding the three ledgers. Invalid coverage, malformed dependencies or timeline overflow return no partial timeline and do not consume replay samples. A file-backed provider can be used directly; see `kernel_replay_contract` for the complete binding/load/replay path.

This diagnostic preserves the submission-order compute-queue heuristic. It does not generate CUDA dependencies, interpret default-stream handles, model host synchronization or transfer costs, or coordinate live framework actors. Changing its timings leaves the supplied trace fixed; it is not closed-loop serving or a ready-driven resource engine. The paced Driver API remains unchanged. [P2 accounting and query-delay invariance](results/2026-09-30/p2-accounting.md).

## Probe artifacts and measured replay

`tests/integration/tooling/probe_ncu.py` captures a native command with NCU, saves its report and metrics, snapshots hardware identity and measurement settings, and writes `kernels.probe`. For an existing compiled module/extension, supply `--image`; the artifact keeps a copy and SHA-256, checks that the source did not change, and labels the association as caller-declared. For one CuTe JIT specialization, use `--capture-cute` to save the PTX/cubin generated by that capture process. Keep one compiled variant per CuTe command; this path requires one cubin and one selected kernel observation.

```sh
python tests/integration/tooling/probe_ncu.py \
  --output /absolute/path/probe-run --image /absolute/path/module.cubin \
  --conditions 'fixed workload, input seed and operating conditions' \
  --kernel 'regex:my_kernel.*' -- /absolute/path/native-workload

# Or replace --image with --capture-cute for a single CuTe specialization.
# In a separate no-GPU container, using the same absolute mounted paths:
build/replay_probe /absolute/path/probe-run/kernels.probe
```

The default capture uses the third matching launch per launch configuration, kernel replay, flushed caches and unchanged clocks. NCU may execute multiple collection passes; do not use the profiled application's host/event timers as kernel service samples. `--report`, `--image`, `--hardware` and `--conditions` import an existing report with its original identity/settings. Raw artifacts stay outside git.

`replay_probe` feeds each observation through `PerformanceModel`, coverage validation and `replay_kernel_trace`, comparing the default 10 ms with the captured service time. Its provider is confined to that observation; unknown parameter contents remain unknown, and it does not match future launches by symbol. Every observation is replayed independently from zero with host work excluded. NCU's selected samples do not establish arrival times, full stream/event dependencies, transfers or a whole-workload makespan. Resources and provenance remain in the artifact; no occupancy multiplier is applied to measured duration. [Local capture-to-replay validation and research rationale](results/2026-09-30/probe-replay.md).

## Build and try it

On Linux aarch64, install `uv`, CMake, a C/C++ compiler, Python 3.12 and Docker with a local `ubuntu:24.04` image. The setup script installs separate PyTorch CUDA 12.8 and 13.0 environments (several GB of downloads); CUDA 13 headers from the latter are required to build.

```sh
bash tests/setup_uv.sh
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
bash tests/run_no_gpu.sh all tensor
bash tests/run_no_gpu.sh all graph
```

The PyTorch probes run in Docker with no GPU devices and no network; `all` checks both CUDA wheels. Other modes cover discovery, memory API paths, streams/events, and multidevice behavior. See [integration tests](tests/integration/README.md) for the complete probe list and framework startup checks.

## What works (and what does not)

- PyTorch CUDA 12.8 and 13.0 device discovery, tensor **allocation/metadata**, virtual memory-operation submission, streams/events, and a supported subset of CUDA Graph capture/replay have passed no-GPU probes. Other CUDA 12 versions have not been verified.
- Device allocations return non-dereferenceable virtual addresses. Copies consume simulated time **but do not copy bytes**; kernels are not executed. Tensor values, numerical PyTorch operations and model inference do **not** work.
- vLLM and SGLang can import and see the virtual device, but neither has been shown to perform inference. vLLM CUDA platform detection is blocked by NVML (outside the Driver API); SGLang Engine import is not model execution. See [framework probe stages](tests/integration/README.md#framework-startup-stages-not-inference).

## Debugging findings

- **Compiled identity and profiling scope:** Native inspection on RTX 5090 found attention variants with identical symbols, parameter layouts and launch dimensions but different cubin hashes and local-memory usage. The generated PTX also identifies a different compiler from system `nvcc`. Preserve the actual compiled artifact, verified parameter ABI and measurement conditions when binding timing samples. NCU's kernel durations must be kept separate from host/event intervals collected under its instrumentation. The generic `tests/integration/tooling/inspect_compiled_kernel.py` reads real-Driver function resources and parameter offsets from a supplied cubin. [Compilation evidence, nine kernel profiles and unprofiled speed comparisons](results/2026-09-30/compiled-kernel-profile.md).

These are conclusions from observed failures and targeted probes, not a claim of full CUDA compatibility:

- **Measured peer availability:** The four RTX 5060 Ti host reports all 12 directed pairs unavailable; the four A100 SXM4 host reports all 12 available and passes native peer enable/disable checks. The A100 host reports NODE topology with inactive NVLinks, so SXM packaging alone does not establish an active NVLink path. `FAKE_CUDA_SYSTEM` reproduces both matrices and all per-card attributes. Legacy configuration and peer-copy timing retain their simulation heuristics. [Cross-validation evidence](results/2026-09-29/a100-cross-validation.md).

- **Device configuration:** Multi-digit device counts previously fell back to one, and memory accounting used a fixed GH200 capacity. Strict count parsing and one profile-backed memory path now cover discovery, admission and free-memory queries. Contract tests cover 1/2/4/8/16/24/32/257 devices, unique ordinal UUIDs, independent accounting and malformed configuration. The original 148 GH200 attributes remain queryable even when the build headers end at a lower attribute number.

- **Replay ordering across launch streams:** Two launches of the same executable previously overlapped on different resource queues. Executables now retain their previous completion independently of the shared graph definition. Replay uses private lanes with launch-stream entry and exit dependencies. A two-copy regression changes the incorrect 80 ms completion separation to 160 ms; distinct executables retain 80 ms separation under the same synthetic resource model. This is a correctness correction, not a speedup.
- **PyTorch 2.13 capture initialization:** This version's RNG-state setup uses `cuThreadExchangeStreamCaptureMode` to allocate outside the graph under a relaxed guard. The missing entry point blocked `capture_begin()`. Thread-local mode exchange, allocation restrictions and capture invalidation now support that path. A real Driver oracle also verifies recovery after invalidation and that a wrong-thread EndCapture ends capture with an error. PyTorch copy and vector-add capture/four-replay fixtures pass in a GPU-free container; the matching real-CUDA fixtures check values. Default-priority kernel nodes accept the node-priority instantiate flag; nonzero captured priorities still return not-supported.

- **CUDA 12/13 ABI:** Building against CUDA 13 headers is not enough to serve CUDA 12 clients. In particular, `cuCtxCreate` resolves to the three-argument `cuCtxCreate_v2` for CUDA 12, but the four-argument `cuCtxCreate_v4` for CUDA 13; elapsed-time and stream-capture-info names also have versioned entry points. `cuGetProcAddress` must choose the matching signature, not merely return an exported name. The split C bridge keeps shared entries in `src/driver_api/common.c` and differing signatures in `cu12.c` / `cu13.c`; both ABI probes pass.
- **`CUDAGraph.capture_begin()` reporting `cudaErrorCallRequiresNewerDriver`:** The first failure was *before capture*, during the CUDA Runtime's lazy library load, which required `cuLibraryLoadData`. After handling that path, capture/replay also needed the observed `cuStreamGetCaptureInfo`, `cuGraphGetNodes`, and `cuGraphInstantiateWithFlags` paths. The current no-GPU probes pass virtual D2D/event capture and two replays with both PyTorch wheels; a Driver ABI lifecycle probe also verifies that two executable graphs remain replayable after their source graph is destroyed. No bytes are copied or kernels executed. [Probe details](tests/integration/README.md).
- **A missing lookup is not necessarily a blocker:** PyTorch prefetches many optional Driver symbols. `FAKE_CUDA_TRACE=1` reports lookups; `FAKE_CUDA_TRACE_CALLS=1` reports calls to implemented entries. Investigate the failing call or import stage before adding an API.
- **vLLM is blocked outside the Driver API:** Its normal CUDA platform selection asks native NVML to discover devices; installing the Python `pynvml` binding alone does not create a virtual NVML device. A *test-only* bypass exposed another blocker, `cuTensorMapEncodeTiled`, which encodes a hardware TMA descriptor and cannot honestly be implemented as an unconditional success stub. [Probe stages and limits](tests/integration/README.md#framework-startup-stages-not-inference).
- **SGLang import is not engine execution:** Supporting the observed `cuPointerGetAttribute` call allowed its `Engine` class to import; no model startup or inference has been verified. Imports and tensor metadata tests cannot validate values that this driver does not store.
- **General graph validation:** A shared PyTorch fixture checks capture, cross-stream replay, and updated inputs; the real-GPU benchmark accepts an operation, reference, and input-update callback. External operator probes expose remaining library/attribute API gaps before capture. [Validation and speed comparisons](results/2026-09-29/README.md).
- **Priority is not an FCFS contract:** NVIDIA documents priority as a non-preemptive compute-kernel scheduling hint, not an execution-order guarantee; it does not specify equal-priority FCFS or require copy/P2P prioritization. The current scheduler reserves resources at submission time, so a waiting operation can reserve ahead of a later-ready operation; priorities do not influence virtual timing. The Driver does clamp requested priorities to the advertised `-1..0` range, and API tests check both boundaries. Simulated queue order should not be presented as real CUDA scheduling.
- **Timing is a model, not measured hardware topology:** The three named queues are simulated resources, not a verified count of GH200 engines. Independent H2D/D2H and directed P2P timelines may overestimate overlap; one compute queue may underestimate concurrent kernel execution. H2D/D2H defaults come from one pinned-buffer GH200 measurement, while HBM/P2P bandwidth defaults are unmeasured. Even accurate delay does not imply a byte copy.
- **CUDA Runtime private ABI and teardown:** PyTorch initialization uses undocumented `cuGetExportTable` UUID/integrity paths, so supporting published Driver entry points alone was insufficient. The implementation isolates this version-sensitive handling; unknown tables fail. The scheduler survives process teardown because Runtime `atexit` handlers may still call the Driver. Normal interpreter teardown was checked with both tested wheels; other Runtime releases need revalidation.
- **Editor diagnostics versus compiler:** After the C bridge split, editor diagnostics reported `driver_api/all.h` missing, while the CMake build, 7/7 CTest cases and no-GPU Docker discovery/tensor/graph probes passed. That discrepancy does not establish a source compilation failure; check the editor's configured include paths before changing code to silence it.

Contributor-facing constraints, configuration and verification procedures live in [AGENTS.md](AGENTS.md).
