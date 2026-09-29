# Graph contracts and kernel integration — RTX 5090

Baseline: `monsoon235/fake-cuda@75fd2cd27713373baef17007547a5dda03219223`.
Changes live on `pcie-test`. This report covers graph control flow and a real-kernel benchmark; numerical kernel execution remains outside the simulator.

## Environment

Existing `/home/gongji/0z5a/bin/python`: PyTorch `2.13.0+cu130` (source `cf30153c4c131c8164ee7798e5022d810682e2cb`), vLLM `0.29.0`, Triton `3.7.1`, flashinfer-python `0.6.18`. GPU measurements use RTX 5090 device 7 (`GPU-c1e2d922-6fb4-92a2-c9e3-9f4ef9d9c079`), driver `580.82.07`. Other GPUs are shared; no clocks or power settings were changed. Build: GCC 11.4, CMake, existing CUDA 13 headers. No environment packages were installed or updated.

GPU-free tests use the existing `python:3.12-slim` image (`sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9`), `--runtime=runc --network=none --pull=never`, read-only environment/source mounts and no device mounts. Processes exit normally; the validation commands use no timeout/kill wrapper.

## Correctness

| Test | Baseline | Updated implementation | Evidence |
| --- | --- | --- | --- |
| Same GraphExec, different launch streams; 80 ms H2D then 80 ms D2H per replay | FAIL: 80 ms completion separation | PASS: 160 ms separation | `base-same.log`, `contracts-no-gpu.log` |
| Independent GraphExec objects | No new serialization required | PASS: 80 ms separation | `contracts-no-gpu.log` |
| Capture does not submit; destroyed template/exec; event re-record snapshot | Existing/extended contracts | PASS | `contracts-no-gpu.log` |
| Mode exchange, thread isolation, relaxed allocation, forbidden allocation, invalidation/recovery, wrong-thread EndCapture | Missing exchange; incomplete state handling | PASS against fake and real CUDA | `capture-mode-fake.log`, `capture-mode-real-safe.log` |
| PyTorch 2.13 cross-stream copy graph, four replays | Capture initialization blocked by missing mode exchange | PASS without GPU; real CUDA values PASS | `fake-torch-fixed.log`, `real-torch-graph.log` |
| PyTorch vector-add graph, four replays | Node-priority flag rejected for every kernel | PASS for default-priority nodes without GPU; real CUDA values PASS | `fake-add-fixed.log`, `real-add.log` |
| Native ABI regression suite | — | 10/10 invocations PASS in no-GPU container | `contracts-no-gpu.log` |

The 160 ms result corrects an invalid overlap; it is not a performance optimization. The 80 ms independent-exec overlap is a simulator heuristic, not a hardware concurrency guarantee. Nonzero captured kernel priorities are still rejected when requested at instantiation. No CUDA 12 wheel rerun was performed in this existing environment.

During oracle development, a probe containing a null mode pointer crashed the real Driver; a separate out-of-domain mode value (`99`) returned success. These inputs are excluded from real-CUDA conformance claims and tested only as simulator argument validation. Valid-mode real-CUDA runs then exposed and verified that wrong-thread EndCapture clears capture; subsequent EndCapture returns `CUDA_ERROR_ILLEGAL_STATE`. See `oracle-development-notes.md` for the failed-run observations; final conformance logs are separate. No process was externally terminated.

## Speed comparison: real Marlin W4A16

Shapes use `K=1024`, `N=2048`, group size 128, FP16 activation/output and `uint4b8` weights. Values are checked against the dequantized reference (`atol=rtol=0.02`), including input updates at stable addresses across replays. Each of three fresh processes performs four APPA/PAAP blocks, 100 calls per arm, with warmup outside timing. The table is the median of each process's median arm wall time; speedup is the median of per-process ratios. Raw CUDA-event and wall-clock measurements are retained. The window includes host submission and final completion, so these are **op submission-path** measurements, not isolated kernel durations. No profiler is enabled in timed windows.

| M | Eager µs/call | Graph µs/call | Speedup | Observed process speedup range |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31.387 | 6.334 | 4.801× | 4.727–4.956× |
| 16 | 31.423 | 6.350 | 4.811× | 4.741–4.949× |
| 64 | 31.617 | 10.463 | 3.020× | 2.881–3.161× |

Raw records: `marlin-real.json`, `marlin-real-2.json`, `marlin-real-3.json`. Each includes all arms and the extension SHA256 (`0375f3ec05f823961476dfe3450cb67df385eb9541eeed29e5daccc28a142ecb`). These three small synthetic cases demonstrate the installed Marlin operator's graph path. They are not full-model E2E, evidence of a faster FakeCUDA patch, or a calibrated Marlin timing profile. No inferential interval or broad performance claim is made from three sessions.

## Kernel and model boundaries

| Path | Status |
| --- | --- |
| PyTorch vector-add binary through FakeCUDA capture/replay | PASS for control flow; values are not computed |
| Marlin on real CUDA | Correctness and eager/Graph timing PASS for the three shapes above |
| Installed Marlin binary through FakeCUDA | BLOCKED at extension load: unresolved `cuTensorMapEncodeTiled`; see `fake-marlin.log` |
| FA4 on real CUDA | Correctness and eager/Graph timing PASS from task-local official source; [follow-up report](FA4-SVDQuant.md) |
| SVDQuant W4A4 + rank-32 on real CUDA | Correctness and changed-input Graph replay PASS via official core objects/test binding; shared-GPU timing is variable; [all runs](FA4-SVDQuant.md) |
| SVDQuant through FakeCUDA | BLOCKED at dynamic shared-memory attribute setup before capture; `svdquant-binding-fake.log` |
| FA4 through FakeCUDA | BLOCKED during CuTe/TVM library initialization after unresolved `cuKernelGetAttribute`; no capture reached |
| YuE2 / BAGEL / MiniCPM full-model execution | NOT RUN in FakeCUDA; numerical execution is deferred, and old real-model results are not simulator evidence |
| Kernel argument layout/values, parameter updates, allocation reuse generations, calibrated durations | Not added by this patch; existing 10 ms virtual kernel duration remains uncalibrated |

Existing model work was checked first under `/home/gongji/0z5a/work/{yue2,bagel,minicpm}-cg-20260927`. BAGEL has eager thinker outputs; MiniCPM's recorded `encoder_capture_axes_available=false` rules out treating its prior output as Graph acceleration. No checkpoint was downloaded or newly created for this kernel test, so there are no task-owned model files to delete. Existing shared model files and `/home/lcpu` NFS were left untouched.

## Reproduce

Build in the existing environment, without dependency installation:

```sh
cmake -S . -B build -DCUDA_DRIVER_INCLUDE_DIR=/usr/local/cuda/include
cmake --build build -j4
build/graph_ordering_probe build/libcuda.so.1 same
build/graph_ordering_probe build/libcuda.so.1 independent
build/capture_mode_probe build/libcuda.so.1
# Explicit physical Driver ordinal (CUDA_VISIBLE_DEVICES does not remap this probe):
build/capture_mode_probe /usr/lib/x86_64-linux-gnu/libcuda.so.1 7
CUDA_VISIBLE_DEVICES=7 /home/gongji/0z5a/bin/python tests/integration/pytorch/graph_contract.py --mode real --kernel add
CUDA_VISIBLE_DEVICES=7 /home/gongji/0z5a/bin/python tests/integration/tooling/benchmark_marlin_graph.py --output /tmp/marlin.json
```

For the no-GPU fixture, substitute the checked-out project path below. This mounts the existing interpreter and environment read-only and uses no GPU runtime:

```sh
PROJECT=/home/gongji/0z5a/work/fake-cuda-pcie-test/patched
BUILD=/home/gongji/0z5a/work/fake-cuda-pcie-test/build-patched
docker run --rm --pull=never --runtime=runc --network=none \
  --mount type=bind,src=/home/gongji/0z5a,dst=/home/gongji/0z5a,readonly \
  --mount type=bind,src=/home/gongji/.local/share/uv/python,dst=/home/gongji/.local/share/uv/python,readonly \
  -e LD_PRELOAD="$BUILD/libcuda.so.1" -e PYTHONDONTWRITEBYTECODE=1 \
  -e FAKE_CUDA_TRACE_CALLS=1 python:3.12-slim /home/gongji/0z5a/bin/python \
  "$PROJECT/tests/integration/pytorch/graph_contract.py" --mode fake --kernel add
```

## Implementation map

| Responsibility | Existing module and change |
| --- | --- |
| Capture status/owner/mode | `Capture` in `include/impl/capture.h`; invalidation and mode checks in `GraphManager` |
| Template vs executable lifetime | `Graph` remains shared/immutable; each executable now owns its completion token |
| Replay dependencies | `src/impl/graph.cpp`: private replay lanes, explicit launch entry and lane-exit joins |
| Driver ABI | `cuThreadExchangeStreamCaptureMode` in stream header, C bridge/resolver, C++ implementation; covered by existing `cu[A-Z]*` exports rule |
| Unsafe synchronous allocation/free and synchronization | `src/impl/virtual_work.cpp`: invalidate affected captures; relaxed allocation stays outside the graph |
| Node priority | Capture the kernel's stream priority; accept default-priority nodes, reject unsupported nonzero arbitration |

Semantics were checked against the [NVIDIA Driver stream documentation](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html) and [Graph documentation](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html), then against the installed real Driver for the declared cases. NVIDIA's live skills catalog was inspected; it had no directly applicable Driver Graph simulator skill.
