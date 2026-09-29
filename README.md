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

Images remain opaque: `kernelParams` has an unknown layout and is never retained or dereferenced without an ABI. Static resources, parameter types, pointed-to allocation generations and content hashes are not decoded yet. Packed snapshots copy parameter bytes, not tensor data. Load IDs are process-local identities, not predictor cache keys. Kernel duration remains the 10 ms heuristic; this is the first profile/metadata slice, not calibrated execution prediction.

A standalone real-Driver oracle, `build/driver_graph_contract /path/to/real/libcuda.so.1`, checks peer access, cross-stream capture, changed-input replay values and eager/graph timing on every visible GPU. It uses bounded pinned buffers and requires CUDA 13 Driver entry points. Four RTX 5060 Ti profiles and real/no-GPU checks are documented in the [hardware report](results/2026-09-29/four-device-validation.md).

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

These are conclusions from observed failures and targeted probes, not a claim of full CUDA compatibility:

- **Measured peer availability:** All 12 directed peer-access queries on the four RTX 5060 Ti test host returned zero, despite four visible devices. `FAKE_CUDA_SYSTEM` now preserves that matrix and each card's PCI/NUMA attributes. Legacy configuration and peer-copy timing retain their simulation heuristics. [Hardware evidence](results/2026-09-29/four-device-validation.md).

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
