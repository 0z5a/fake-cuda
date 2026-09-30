# A100 serving prediction and configuration sweep

vLLM 0.30.0; Qwen2.5-0.5B bf16, TP1, eager TRITON_ATTN; P64/O64; fixed-length token oracle. Native collection was restricted to physical GPU0 (A100-SXM4-40GB) with no other process on that card; the concurrent Qwen-Image task was restricted to GPU1. CPU replay initializes no CUDA. This is U1: original Scheduler/KV manager with adapted worker completion and an explicit coordinator. Whole-step costs include host and communication; neither is billed again.

Calibration: 24 separate runs, batch 1/8/16/32, prompt 32/64/128, output 96. Both predictors were frozen before reading 18 validation runs: three repeats of each workload and max-sequence setting. AISimulate 0.12.0 uses its native regression explicitly, not an op-level database for a different backend version.

| Predictor | Workload | Max seq | Native s | CPU s | Reuse speedup | Duration error | Throughput error | TTFT median error | ITL median error | TTFT P95 error | ITL P95 error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| measured-step | burst-128 | 8 | 16.334 | 0.413 | 39.54× | -0.53% | +0.53% | -0.79% | -0.15% | -0.57% | -1.74% |
| AIS regression | burst-128 | 8 | 16.334 | 0.375 | 43.70× | -0.79% | +0.79% | -1.04% | -0.41% | -0.83% | -2.07% |
| measured-step | two-waves-96 | 8 | 12.240 | 0.298 | 41.05× | -0.45% | +0.45% | -0.55% | -0.11% | -0.47% | -1.33% |
| AIS regression | two-waves-96 | 8 | 12.240 | 0.290 | 42.09× | -0.70% | +0.71% | -0.80% | -0.37% | -0.73% | -1.66% |
| measured-step | burst-128 | 16 | 8.247 | 0.269 | 30.61× | -0.25% | +0.25% | -0.60% | +0.15% | -0.25% | -1.09% |
| AIS regression | burst-128 | 16 | 8.247 | 0.266 | 30.98× | -0.18% | +0.18% | -0.52% | +0.20% | -0.17% | -1.05% |
| measured-step | two-waves-96 | 16 | 6.191 | 0.209 | 29.69× | -0.35% | +0.35% | -0.60% | +0.00% | -0.37% | -1.25% |
| AIS regression | two-waves-96 | 16 | 6.191 | 0.206 | 30.12× | -0.27% | +0.28% | -0.52% | +0.05% | -0.29% | -1.20% |
| measured-step | burst-128 | 32 | 4.250 | 0.202 | 21.00× | -0.50% | +0.50% | -1.19% | +0.05% | -0.63% | -1.39% |
| AIS regression | burst-128 | 32 | 4.250 | 0.202 | 21.01× | -0.09% | +0.09% | -0.81% | +0.51% | -0.23% | -0.85% |
| measured-step | two-waves-96 | 32 | 3.189 | 0.162 | 19.73× | -0.56% | +0.57% | -1.35% | +0.07% | -0.78% | -1.64% |
| AIS regression | two-waves-96 | 32 | 3.189 | 0.167 | 19.13× | -0.15% | +0.15% | -0.97% | +0.53% | -0.37% | -1.10% |

CPU workload wall time includes original scheduling, oracle, predictor and bridge IPC/startup/teardown. Reuse speedup excludes prior downloads/calibration and process imports; it is a timing simulation speedup, not numerical inference acceleration. Reported rows are medians of three runs. Tail percentiles use linear interpolation of all request TTFTs and token intervals in each run; three repeats under shared host load do not establish confidence bounds.

| Predictor | Held-out step coverage | Conditional WAPE | Conditional MAPE | Completed workloads | ≤10% duration / throughput / TTFT / ITL / P95 TTFT / P95 ITL |
|---|---:|---:|---:|---:|---| 
| measured-step | 9408/9408 | 0.51% | 0.49% | 18/18 | 18/18 / 18/18 / 18/18 / 18/18 / 18/18 / 18/18 |
| AIS regression | 9408/9408 | 0.63% | 0.61% | 18/18 | 18/18 / 18/18 / 18/18 / 18/18 / 18/18 / 18/18 |

| Workload | Previous core CPU s | Optimized core CPU s | End-to-end simulator speedup |
|---|---:|---:|---:|
| burst-128, max seq 8 | 0.742 | 0.413 | 1.80× |
| two-waves-96, max seq 8 | 0.448 | 0.298 | 1.50× |
| burst-128, max seq 16 | 0.321 | 0.269 | 1.19× |
| two-waves-96, max seq 16 | 0.233 | 0.209 | 1.12× |
| burst-128, max seq 32 | 0.219 | 0.202 | 1.07× |
| two-waves-96, max seq 32 | 0.171 | 0.162 | 1.05× |

All 18/18 baseline/optimized runs have identical virtual token timestamps and batch membership. Core-only history scaling is reported separately.

Sum of all 18 complete serving-run wall times: previous core 7.065 s, optimized 5.016 s, 1.41× (29.00% less wall time). Imports, fitting and independent scoring are excluded from this identical-work comparison.

| Predictor | Workload | Chosen max seq | Measured best max seq | Throughput regret |
|---|---|---:|---:|---:|
| measured-step | burst-128 | 32 | 32 | 0.00% |
| measured-step | two-waves-96 | 32 | 32 | 0.00% |
| AIS regression | burst-128 | 32 | 32 | 0.00% |
| AIS regression | two-waves-96 | 32 | 32 | 0.00% |

Regret is restricted to the three independently measured settings, 8/16/32; no optimum is inferred for unmeasured configurations. Native request admission runs at step boundaries while the simulator delivers declared arrivals exactly; native add-request overhead is outside the step-cost scope. Neither discrepancy is fitted on validation data.

| Retained input | SHA-256 |
|---|---|
| scheduler-config.pkl | `a97291111c939492577cd218137449c7221ebd6f13495203300aa7735c53a0bf` |
| calibration.json | `98a6f23dbc2f3c57a226f16db7b39758db70830d35e65d805035dc572fa83201` |
| validation.json | `e8cc5a83b4616f832321da281f171c035a65d7f06d3e3c5cb172b8921285ed34` |
| device-0.profile | `84e2b59b685cf4ac1ac850b63043a617a2dd5340661b75e4368c838c018536a8` |
| Canonical model configuration | `d61858b8764f47d9c67d21d9a20aa9a39c1e025da15291704ee16b7d1120509f` |

Complete comparison process wall: 29.133 s, including imports, both fits, all baseline/comparator runs and scoring.

Whole-step intervals cover 99.52–99.87% of native validation elapsed time (median 99.75%). The remaining admission/loop time is outside this predictor; 9408/9408 step coverage is not a claim that every host operation is modeled. Calibration contains 2304 steps. The native validation workloads total 151.386 s; native campaign startup, warmup, calibration and validation total 210.442 s. Task-owned 988,097,824 bytes of weights were deleted after evidence backup.
