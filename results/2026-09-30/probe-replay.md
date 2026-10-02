# Native probe artifacts and isolated replay

Baseline: `ce8a5fd`. Implementation: `54e7c62`. Native measurements use gongji GPU 0 (RTX 5090, 170 SMs, driver 580.82.07), the existing `0z5a` environment, PyTorch 2.13.0+cu130, CUTLASS DSL 4.6.2 and Nsight Compute 2025.3.1. Replay and regression checks use offline Docker with `--runtime=runc --network=none --pull=never`, without GPUs. No packages, clocks or services were changed, and no model weights were downloaded.

The new path consumes probe artifacts instead of inventing per-symbol timing constants. Six existing native workload cases produced nine kernel observations. All nine now pass through the shared performance-model interface, coverage validator and fixed-trace scheduler without a GPU. Each is an independent observation with time origin zero; host submission, transfers and dependencies are not reconstructed from NCU rows.

```text
native workload + fixed input conditions
  +-- compiler -> PTX/cubin, or caller-declared existing image
  `-- NCU -> per-kernel duration, launch resources, hardware snapshot
                    |
                    v
          kernels.probe + artifact digests
                    |
                    v
       one observed invocation -> PerformanceModel
                    |
                    v
          coverage check -> fixed-trace replay
                    +-- target kernel completion
                    `-- separate simulator wall time
```

## Artifact production and consumption

`tests/integration/tooling/probe_ncu.py` captures an existing native command or imports an existing NCU report with its original hardware snapshot and declared conditions. Captures use kernel replay, unlocked clocks, flushed caches, LaunchStats/Occupancy/SpeedOfLight, and the third matching launch per launch configuration. The unprofiled native workload checks numerical results independently.

`--capture-cute` saves the selected process's actual generated PTX and cubin in an isolated compiler directory. It requires one compiled specialization and one selected kernel observation per run. All three attention captures reproduced the cubin hashes recorded in the [compiler investigation](compiled-kernel-profile.md). For prebuilt extensions, `--image` snapshots and hashes the explicit image; this association is caller-declared, not verified dynamic-loader mapping.

Raw output contains `capture.ncu-rep`, `metrics.csv`, `hardware.csv`, `capture.txt`, `capture.log`, `compiled.image`, and `kernels.probe`; CuTe captures also retain `compiler/`. Raw files remain outside git. The schema-1 replay artifact declares `isolated_kernel_observations` and report/code/hardware/conditions digests. Each tab-separated observation contains ID, device ordinal, grid, block, static shared bytes, dynamic shared bytes, registers/thread, duration in nanoseconds, and symbol. Original NCU process/context/stream IDs remain in the raw CSV; they do not establish a complete execution graph.

`build/replay_probe kernels.probe` validates the complete artifact before replay. Its measured provider belongs to one observed invocation; unknown parameter contents remain unknown. It cannot be installed as an automatic symbol cache or bind an unrelated live launch. The compiled inspector queries the real Driver's function resources and parameter layout separately; it does not infer tensor data or argument contents.

## Measured-service comparison

The default provider assigns **10,000 µs per kernel**. The new artifact path reproduces the observed service durations below. These are in-sample replay checks, not held-out prediction accuracy or a hardware speedup. NCU samples use different cache/profiling conditions from the unprofiled native eager/graph comparison.

| Workload case | Kernel | Default service (µs) | Probe service (µs) | Replayed completion (µs) | Replay-call wall (µs) |
|---|---|---:|---:|---:|---:|
| Attention, L=512, non-causal | attention | 10,000 | 18.528 | 18.528 | 5.544 |
| Attention, L=2048, non-causal | attention | 10,000 | 61.568 | 61.568 | 5.275 |
| Attention, L=2048, causal | attention | 10,000 | 66.176 | 66.176 | 6.447 |
| Quantized, M=256 | quantize + low-rank down | 10,000 | 9.600 | 9.600 | 6.013 |
| Quantized, M=256 | GEMM + low-rank up | 10,000 | 132.640 | 132.640 | 2.159 |
| Quantized, M=1024 | quantize + low-rank down | 10,000 | 8.032 | 8.032 | 5.576 |
| Quantized, M=1024 | GEMM + low-rank up | 10,000 | 127.488 | 127.488 | 2.244 |
| Quantized, M=4096 | quantize + low-rank down | 10,000 | 12.608 | 12.608 | 6.160 |
| Quantized, M=4096 | GEMM + low-rank up | 10,000 | 268.448 | 268.448 | 2.382 |

Sources, relative to the task's raw `evidence/compiler-ncu/` directory: `attention-{0,1,2}-captured/replay.md`, `quantized-0-artifact/replay.md`, and `quantized-{1,2}-probe/replay.md`. Quantized case 0 exercises report import. Each replay-call wall value is one diagnostic observation excluding CLI startup, file parsing and output; the default call precedes the measured call. These values are not a controlled before/after CPU speed benchmark.

The independently measured native eager/graph ratios are **1.186×–3.866× for attention** and **1.011×–1.028× for quantized workloads** across three processes per workload family. See the complete [speed comparison table](compiled-kernel-profile.md#unprofiled-eagergraph-speed-comparison). Those ratios measure CUDA Graph benefit on the real GPU. This patch does not optimize the native kernels or establish faster FakeCUDA execution; its concrete improvement is using captured kernel service times with explicit provenance.

## Validation

| Check | Result |
|---|---|
| Registered ABI, graph, profile, predictor, replay and probe contracts | 26/26 passed without GPUs |
| CSV units and artifact round trip | Passed, including decimal kilo-byte and nanosecond conversion |
| Invalid CSV/artifact inputs | 17 cases rejected; no partial replay output |
| Declared-image and captured-cubin identities | Both accepted by the isolated consumer; malformed identities rejected |
| Native capture and existing-report import | Both exercised on real NCU reports |
| Six native cases → artifact → no-GPU replay | 9/9 kernel observations reproduced |
| Real Driver inspection of captured attention cubins | All three passed; ten-parameter ABI recorded |
| Saved four-device 5060 Ti system | Attributes and directed peer matrix passed |
| PyTorch copy/add graph processes on saved system | 8/8 passed, four graph replays each |

The full suite passed before the captured-cubin identity prefix was added; the probe contract was rerun after that addition and covers both prefixes. Earlier A100 cross-validation remains documented separately; it was not rerun for this tooling-only iteration.

## Research informing the boundary

- [Maya](https://arxiv.org/html/2503.20191v2) separates kernel runtime prediction from trace collation and dependency handling. This implementation follows that separation: NCU supplies observed service time, while missing invocation arguments and dependencies remain unknown. The paper's prediction accuracy is not a result of this implementation.
- [Accel-Sim](https://accel-sim.github.io/) uses dynamic SASS traces, performance modeling and hardware correlation. Static disassembly and aggregate NCU counters do not provide those dynamic instruction/memory traces. This probe therefore does not claim a cycle-level Blackwell model.
- [Revati](https://arxiv.org/html/2601.00397v1) coordinates actor/observer progress with a timekeeper. Selected NCU kernel observations alone cannot reconstruct host blocking or live causality; that requires additional trace capture and coordination.
- [NCU profiling guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#workload-durations) distinguishes per-kernel GPU duration from host/event timing affected by profiling overhead. The artifact uses `gpu__time_duration.sum` and records cache/clock settings; profiler process wall time is not substituted for target service time.

Next validation needs independent repeated or held-out captures, compiler-qualified live invocation binding, and host/transfer/dependency traces before claiming full-workload timing accuracy. Resource counts can explain measurements, but must not multiply a measured full-kernel duration and count their effects twice.
