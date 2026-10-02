# Compiled kernel identity and native profiling

This investigation precedes further predictor changes. It uses the existing `0z5a` environment on gongji, an idle RTX 5090 (GPU 0, driver 580.82.07), PyTorch `2.13.0+cu130`, CUTLASS DSL 4.6.2 and Nsight Compute 2025.3.1. No environments, clocks or running services were changed. No model checkpoints were needed. GPU 0 returned to 0 MiB used after the runs.

Six workload cases passed independent numerical checks in three unprofiled processes per workload family. Nine kernel launches were profiled separately. These observations informed the generic cubin inspector and the subsequent [probe-artifact replay path](probe-replay.md); they are not a simulator speedup or a kernel optimization.

## Compilation evidence

```text
CuTe Python specialization -> compiler IR -> NVVM/PTX -> compiled cubin
CUDA C++ + NVCC flags     -> device compilation       -> cubin in extension
                                                         |
                       exact image bytes + SHA-256 + symbol
                                                         |
                      real Driver function resources / parameter layout
                                                         |
                      launch configuration + input/measurement conditions
                                                         |
                      per-kernel NCU service-time observations
```

System `nvcc --version` reports **13.0.88**. However, the captured attention PTX explicitly reports **NVVM compiler 13.3.27**, build `CL-37800683`, NVVM 23.0.0, `.version 9.3` and `.target sm_120a`. The installed Python NVVM distribution reports yet another version, 13.4.92. Neither the system compiler version, installed package version nor `torch.version.cuda` alone identifies the compiler that produced this artifact.

The quantized extension's saved Ninja recipe instead invokes `/usr/local/cuda/bin/nvcc` with `compute_120a,code=sm_120a`, C++20 and expensive ptxas optimizations. Its SHA-256 remains `c341dd753891a4334a29b90f941c748387c317f2ddf3e8020397302a11f91121`. This identifies the extension container, not an individually extracted kernel cubin. Host `-Og`/`-UNDEBUG` and device `-g` are recorded build options, not proof that device optimizations were disabled. CUDA source, PTX, cubin and fatbinary are distinct compilation artifacts. [NVCC compilation phases](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#the-cuda-compilation-trajectory).

For the attention cases, batch=1, heads=8, head dimension=64 and dtype=FP16:

| Case | Cubin SHA-256 | Driver registers/thread | Driver local bytes/thread |
|---|---|---:|---:|
| Length 512, non-causal | `ff3f64afca02eaebbfb771a4631aae382560d03331eda3603b0385588bfdcb5a` | 255 | 80 |
| Length 2048, non-causal | same as above | 255 | 80 |
| Length 2048, causal | `4315d90588b81b033152a01d700fdfe569add359dbcf69b6a2e80456b2ce9f1a` | 255 | 200 |

All three have the **same exported symbol and parameter layout**. The two length-2048 cases also have identical grid/block dimensions and dynamic shared memory. Thus symbol/shape/resource-count matching would conflate different compiled programs. Conversely, one cubin serves both non-causal lengths with different grids and durations. Keep both artifact identity and invocation metadata.

Native `cuFuncGetParamInfo` returns ten `(offset, size)` pairs: `(0,48)`, `(48,48)`, `(96,48)`, `(144,48)`, `(192,4)`, `(196,4)`, `(200,4)`, `(204,4)`, `(208,12)`, `(220,4)`. The 224-byte ABI includes by-value descriptors; treating every `kernelParams` entry as an eight-byte tensor pointer would be wrong. The Driver reports zero user static shared bytes and a 49,152-byte dynamic limit. `cuobjdump`'s ELF shared allocation includes additional reserved space; it must not replace the Driver's user-static attribute. [Driver function and parameter queries](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__EXEC.html).

SASS inspection finds 18 static `STL*` and 20 `LDL*` instructions in the non-causal cubin versus 80 and 82 in the causal cubin. These are static instruction counts, not executed traffic. NCU separately reports local spilling requests. This establishes a compiled-code difference; it does not isolate spilling as the sole cause of the latency difference.

## NCU observations

Each case was run in a separate process and dump directory, avoiding overwrite of identically named cubins and profiler grouping of same-name/same-launch variants. Settings: kernel replay, `--clock-control none`, `--cache-control all`, third matching launch per configuration, LaunchStats/Occupancy/SpeedOfLight/MemoryWorkloadAnalysis/InstructionStats. Each collected kernel required ten passes. Values below are single profiled observations under unlocked clocks and flushed caches, not steady-state performance estimates.

| Workload | Kernel | Grid | Threads/block | Registers/thread | Static / dynamic shared bytes | NCU duration (µs) | Waves/SM | Local spilling requests |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FA4, L=512, non-causal | attention | 4×8×1 | 128 | 255 | 0 / 49,152 | 18.784 | 0.09 | 10,496 |
| FA4, L=2048, non-causal | attention | 16×8×1 | 128 | 255 | 0 / 49,152 | 57.248 | 0.38 | 177,152 |
| FA4, L=2048, causal | attention | 16×8×1 | 128 | 255 | 0 / 49,152 | 66.656 | 0.38 | 234,048 |
| SVDQuant, M=256 | quantize + low-rank down | 1×8×1 | 256 | 168 | 512 / 34,816 | 9.152 | 0.05 | 0 |
| SVDQuant, M=256 | GEMM + low-rank up | 1×16×1 | 256 | 250 | 17,408 / 0 | 132.384 | 0.09 | 0 |
| SVDQuant, M=1024 | quantize + low-rank down | 4×8×1 | 256 | 168 | 512 / 34,816 | 8.416 | 0.19 | 0 |
| SVDQuant, M=1024 | GEMM + low-rank up | 4×16×1 | 256 | 250 | 17,408 / 0 | 134.240 | 0.38 | 0 |
| SVDQuant, M=4096 | quantize + low-rank down | 16×8×1 | 256 | 168 | 512 / 34,816 | 12.512 | 0.75 | 0 |
| SVDQuant, M=4096 | GEMM + low-rank up | 16×16×1 | 256 | 250 | 17,408 / 0 | 258.688 | 1.51 | 0 |

Quantized cases use K=1024, N=2048, rank=32 and group size=64. The GEMM's per-SM register residency bound is one block; the 256-block case exceeds the device's 170 SMs. Nearly doubled duration is consistent with additional work waves in this fixture, but is not a validated universal wave multiplier. Small grids leave much of the device unused even when achieved occupancy on active SMs appears similar.

The old host-loop event interval includes submission gaps. Under NCU, both host and CUDA-event intervals additionally include profiling overhead. Only NCU's per-kernel duration is used in the preceding table. These flushed-cache observations are not added to the warm whole-pipeline times below. [NCU measurement overhead, serialization and cache policy](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#workload-durations).

## Unprofiled eager/graph speed comparison

The existing numerical workloads ran outside NCU, with 100 repetitions per arm, APPA/PAAP ordering repeated twice, in three fresh processes. The table reports the median of each process's arm median. All changed-input numerical checks passed; the quantized low-rank branch remained nonzero. These are native kernel-pipeline measurements, not model E2E or a before/after FakeCUDA patch comparison.

| Workload | Eager wall (µs) | Graph wall (µs) | Eager / graph | Graph range across processes (µs) |
|---|---:|---:|---:|---:|
| FA4, L=512, non-causal | 64.211 | 16.610 | 3.866× | 16.603–16.614 |
| FA4, L=2048, non-causal | 63.484 | 53.518 | 1.186× | 53.509–53.645 |
| FA4, L=2048, causal | 74.774 | 57.613 | 1.298× | 57.597–57.703 |
| SVDQuant, M=256 | 133.726 | 131.252 | 1.019× | 131.247–131.284 |
| SVDQuant, M=1024 | 135.107 | 131.369 | 1.028× | 131.366–131.382 |
| SVDQuant, M=4096 | 263.285 | 260.441 | 1.011× | 260.395–260.662 |

## Consequences for the simulator

The next runtime work should use build-qualified artifact identities and explicitly verified ABI layouts to bind real invocations, then validate separate kernel-service and host-submission costs. Quantization and GEMM need distinct accounting owners. Residency and spilling explain why profile attributes alone cannot supply a duration; measured full-kernel time already includes their effects. The current fixed-trace replay remains a diagnostic, and these samples do not establish cross-shape or cross-device prediction accuracy.

`tests/integration/tooling/inspect_compiled_kernel.py IMAGE.cubin SYMBOL` hashes the exact bytes loaded into the real Driver and prints resource/parameter tables. It was validated against all three attention cubins above. It neither launches a kernel nor injects metadata into FakeCUDA. Use `--library` to select the real Driver and `--device` for the ordinal. Keep raw PTX, cubin, SASS, NCU reports and logs outside git; only Markdown results and the generic tool are committed.
