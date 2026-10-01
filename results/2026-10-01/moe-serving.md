# M1 operator accounting and H20 validation

M1's qualified paths pass independent shape validation with zero unsupported timing stages. Source and dependency identity are in the [baseline lock](moe-baseline.lock.md); [usage and boundaries](../../sim/README.md) describe the schema and replay API. Only source and Markdown evidence are submitted.

| Delivery | Evidence |
| --- | --- |
| P0 | Actual PR #2 diff locked; baseline builds and passes 32 contracts |
| P1 | Owned Graph annotations, fresh bindings, schema/registry validation, integer replay, fusion accounting and rescheduled counterfactuals |
| P2 | Recorded/uniform/hot routes, conserved counts, actual per-expert padding, CPU permutation/weighted-combine reference |
| P3 | KDA state recurrence/reference, live slots and allocated capacity, FP32/BF16 byte accounting, core/projector separation and partial hybrid layout |
| P4 | Frozen profiles, new whole-shape holdouts, numerical native checks and qualified profile export |
| Regression | 34/34 registered contracts; 12 operator CPU cases; 100 identical replay hashes for every scored family |

## Native protocol and supported paths

Two H20 devices, Driver 580.105.08, unmodified clocks. MoE uses UUID `edf64e5f-21ef-a1f2-6601-e8620b5664ff`; KDA/top-k use `8ee84e7d-143f-dd29-1097-85943783e027`. Profiles do not cross-match UUIDs. Numerical collection loads the real Driver in a private process; CPU replay loads no GPU framework.

Each case uses 30 full warmups, 200 unprofiled CUDA-event measurements per stage/whole operator and five independent process sessions. Graphs have fixed addresses and steady reuse. A queued prefill prevents an idle GPU from timing the Python submission gap; host submission is recorded separately. CUDA launch diagnostics precede warmup. Raw sessions include actual routes, source/binary hashes, peak allocated memory and measurement arrays.

| Family | Qualified numerical backend / shape |
| --- | --- |
| MoE | PyTorch BF16 per-expert gated matrices, D=128, E=256, k=8, m=128, tile_M=4; softmax routing/renormalization, no shared expert |
| Short-row top-k | FP32, E=256, k=8, sorted; finite random scores; ties have backend-unspecified index order |
| KDA core | FLA 0.5.2 forward, Hqk=Hv=32, dk=dv=128, FP32 key/value state, FP32 accumulation, fixed slots |
| Declared KDA module | D=2304 BF16 QKV/O matrices, four-tap depthwise convolution, normalization, fixed alpha/beta and elementwise output gate |

The KDA module is the stated surrogate block, not the entire production Kimi attention module: trainable gate projections, MLA, indexer, prefill and LM head are not covered. No whole-model attention fraction is reported. BF16 state bytes, shared experts, alternative E/k/dtypes and rotating slots have algebra/contracts but no qualified timing path here; strict lookup rejects absent profiles. DRAM/workspace/graph-buffer totals remain unknown rather than being filled from logical bytes.

## Independent prediction accuracy

MoE calibration uses batches 1/4/8/16/17/32/64/128. Validation uses **3/10/28/56/112**, plus hot routes at **10/28/56/112** (64-expert hot set). KDA/top-k calibrate at 1/8/32/64/128 and validate at **2/12/40/80**. Parameters and feature bounds freeze before validation. Earlier validation recordings never enter fitting; only prior calibration sessions are reused. No validation durations are used to repair a fit.

Per-stage gates: long stages (native >=10 us) p50 relative error <=15%, p95 <=30%; short-stage p95 absolute error <=2 us. Whole-operator p50 error <=20%, at least three held-out configurations, five sessions and strict unknown count zero.

| Family | Held-out configurations / case-session observations | Long-stage p50 / p95 error | Short-stage p95 absolute error | Whole-operator p50 error | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| MoE | 9 / 45 | 3.45% / 6.33% | 0.277 us | 2.83% | PASS |
| KDA | 4 / 20 | 0.99% / 23.36% | 0.585 us | 2.31% | PASS |
| Top-k | 4 / 20 | 2.27% / 5.96% | no short stages | 2.56% | PASS |

| Stage | p50 error | p95 error |
| --- | ---: | ---: |
| MoE router | 2.43% | 4.61% |
| MoE score | 2.69% | 5.10% |
| MoE top-k / renormalize | 3.51% | 13.63% |
| MoE counts / offsets / sorting | 1.11% | 4.38% |
| MoE permute | 4.47% | 13.67% |
| MoE gate/up | 3.34% | 4.61% |
| MoE activation | 4.30% | 7.21% |
| MoE down | 4.46% | 5.57% |
| MoE combine | 3.38% | 5.59% |
| KDA core | 1.48% | 24.28% |
| KDA QKV | 0.75% | 4.67% |
| KDA convolution / gates | 1.09% | 4.33% |
| KDA output normalization / gate | 0.55% | 1.41% |
| KDA O projection | 1.49% | 6.84% |

## Routing failure and compiled-path correction

The first MoE holdout failed despite 1.92% whole-operator p50 error: aggregate long-stage p95 was 38.14%, with routing p95 41.06%. CUDA launch traces showed the pinned binary selecting 32-item, 128-item and 1024-item radix-sort instantiations. The [pinned dispatcher source](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/aten/src/ATen/native/cuda/Sort.cu) explains the fixed-size branch boundaries. Counts were never inferred from those names: they remain explicit recorded input; the ordered launch signature supplies a backend variant key.

An intermediate profile correctly rejected an untrained tiny-sort interpolation endpoint. Fresh boundary calibration completed that domain before the final, new holdout. Final routing p95 is 4.38%. This compares validation histories on different datasets, not paired errors on the same test set. The original failed/unsupported sessions remain retained.

NCU LaunchStats and counter collection both returned `ERR_NVGPUCTRPERM`. No host Driver setting was changed. Compiled-path evidence comes from CUDA launch traces and pinned binaries/source; it is not an NCU counter measurement. [NCU profiling semantics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) require treating instrumentation separately; no profiler event interval enters the calibrated timing files.

## Complete CPU trace-process speed

Both arms use the same qualified profile, descriptors, replay ledger and registered completion protocol. The baseline uses existing `PacedCoordinator`; the candidate uses existing `Coordinator`. Only waiting for virtual time differs. Each process replays the selected native session 100 times. Five independent APPA/PAAP sessions per family produce 20 process runs, with one identical combined timeline/ledger/reply digest across every arm.

Wall time includes interpreter startup, imports, profile/trace loading, replay, hashing, delivery, output and natural teardown. Profiles are frozen and reused in both arms; fitting is not included. This measures the complete fixed operator-trace process, not numerical inference or the live Driver.

| Trace | Operator replays / process | Paced process median | Offline process median | Speed ratio | Wall reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoE | 500 | 0.921743 s | 0.286015 s | 3.223x | 68.97% |
| KDA | 400 | 0.192311 s | 0.141458 s | 1.359x | 26.44% |
| Top-k | 400 | 0.090468 s | 0.089244 s | 1.014x | 1.35% |

Per-session speed ratios span 3.148–3.331x for MoE, 1.342–1.411x for KDA and 0.992–1.053x for top-k. Top-k has no consistent wall-time improvement at this scale; process and metadata overhead dominate its short target time.

## Artifact identities and remaining scope

| Family | Frozen profile SHA256 | Qualified profile SHA256 |
| --- | --- | --- |
| MoE | `55eecd42102de9a3df884b1862393290bb2ae6c6474aa5819e642cc2489ed3ef` | `14d0f000bc4e04cd99ca9356d14f3d19bfec649c9be7a73528cdad081e9172ac` |
| KDA | `3ef120a36baf7e1e74cd5dd60267d4d164a1c2c23cf009d7a736b33be64fd69b` | `eacb464653c915f82ad2ef2d43f7de761584dabe86d0cc70d81e9a183a196bbb` |
| Top-k | `3d862ab6403fc2035f85d3c28e216ea89f479034aafc6ce84bf6d952765ca91d` | `2dfcc29b95016ecb5c42fc20e3a6da08a92e200bab003a23df4cb543564c8d08` |

Original session manifests, compiled caches, profile files and diagnostics remain outside git, with local copies and hashes. Only configuration metadata was downloaded; numerical weights are process-local generated tensors, released after each case/session. No process was terminated to complete a benchmark. P5 serving and P6 EP/resource overlap remain later milestones; cold-cache and broader backend timing require their own calibration and validation.
