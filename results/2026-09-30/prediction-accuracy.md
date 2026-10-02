# Independent kernel timing accuracy

The previous nine probe durations were frozen before collecting **24 new NCU reports containing 36 kernel observations**. On the 27 observations with calibrated configurations, frozen-duration reuse has **3.654% MAPE**, **13.746% P95 absolute percentage error**, and **14.915% maximum error**. Twenty-four of 27 are within 10%. The nine observations from two unseen shapes are unsupported: overall coverage is **27/36 (75%)**, and unseen-shape coverage is **0/9**.

This tests same-configuration prediction from an earlier measurement. It is not the earlier in-sample replay equality check, a learned cross-shape predictor, or full-workload latency accuracy. The runtime remains unchanged from `b8b516f`; the new generic evaluator scores independently captured artifacts offline.

See the [comparison with Maya, Revati, SGLang and Accel-Sim](simulator-comparison.md) for differences in prediction scope, datasets and metrics.

## Protocol

- Hardware/environment: gongji GPU 0, RTX 5090, UUID `GPU-a015762a-f0d5-9065-109d-37898af80ecf`, driver 580.82.07, existing `0z5a` Python/PyTorch environment and NCU 2025.3.1. GPU 0 was idle before the experiment and returned to 0 MiB/0% afterwards. Other GPUs' services were left running.
- Calibration: the six original reports from [probe replay](probe-replay.md), containing one sample for each of nine kernel/configuration keys. Artifact hashes and copies were frozen before any validation capture; originals were checked against those snapshots before scoring. No validation observation was used to update a prediction.
- Validation: three fresh native processes per case, eight cases per round. Case order alternated forward, reverse, then a four-case rotation. The six original cases used unchanged workload scripts and seed 20260929. Two held-out cases changed only attention length to 1024 or quantized M to 2048. Numerical checks passed in all 24 processes, including the nonzero low-rank branch.
- Measurement: the same third matching launch per configuration, kernel replay, `--clock-control none`, `--cache-control all`, LaunchStats/Occupancy/SpeedOfLight. Attention captured the process's generated cubin; quantized workloads used the same prebuilt extension hash. Profiler host/event intervals were not used as kernel targets.
- Estimator: the evaluator freezes the calibration median for each exact code/hardware/device/symbol/grid/block/static-shared/dynamic-shared/register key. Here each key has one calibration sample, so predictions equal those original samples. Unknown keys abstain. Known-case input and profiling-condition equivalence was audited explicitly; metadata alone does not establish argument equivalence.
- Isolation: calibration and validation NCU report digests must be disjoint. Duplicate reports within either split and changed report/hardware/manifest digests fail before output. Scoring and artifact replay ran in a no-GPU, offline `runc` container.

## Aggregate accuracy

For prediction `p` and new observation `y`, APE is `100 × |p−y| / y`; MAPE averages APE across observations. P95 uses nearest rank. WAPE is `100 × Σ|p−y| / Σy`; bias is mean signed percentage error. Unsupported rows are excluded from conditional error and retained in coverage. All nine calibrated keys have three validation observations, so their contribution to MAPE is balanced.

| Estimator | Evaluated observations | MAPE | P95 APE | Maximum APE | WAPE | Signed bias | Within 10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default 10 ms, same covered set | 27 | 44,205.830% | 108,031.488% | 108,031.488% | 12,788.173% | +44,205.830% | 0/27 |
| Frozen probe duration | 27 | **3.654%** | **13.746%** | **14.915%** | **2.662%** | **−0.723%** | **24/27** |
| Default 10 ms, all validation | 36 | 44,725.681% | 108,031.488% | 108,031.488% | 13,599.131% | +44,725.681% | 0/36 |

The default is a synthetic timing constant, not a calibrated competing model. Error reduction relative to it is not an execution-speed improvement. Native eager/graph speed comparisons remain in the [compiler report](compiled-kernel-profile.md#unprofiled-eagergraph-speed-comparison).

## Per-configuration results

Attention uses the existing FA4 fixture: B=1, H=8, D=64, FP16. Quantized workloads use the existing SVDQuant fixture: K=1024, N=2048, rank=32, group=64. Each row summarizes three independent validation processes; MAPE is computed against all three observations, not only their median.

| Case | Kernel | Frozen prediction (µs) | Validation median (µs) | Validation range (µs) | MAPE |
|---|---|---:|---:|---:|---:|
| Attention L512, non-causal | attention | 18.528 | 18.208 | 18.144–18.720 | 1.63% |
| Attention L2048, non-causal | attention | 61.568 | 61.472 | 57.120–61.888 | 2.82% |
| Attention L2048, causal | attention | 66.176 | 67.616 | 66.048–68.992 | 2.14% |
| Quantized M256 | quantize + low-rank down | 9.600 | 9.248 | 9.248–9.312 | 3.57% |
| Quantized M256 | GEMM + low-rank up | 132.640 | 128.032 | 127.200–131.008 | 3.04% |
| Quantized M1024 | quantize + low-rank down | 8.032 | 9.312 | 9.248–9.440 | **13.94%** |
| Quantized M1024 | GEMM + low-rank up | 127.488 | 128.800 | 128.000–134.336 | 2.17% |
| Quantized M4096 | quantize + low-rank down | 12.608 | 12.736 | 12.704–12.768 | 1.00% |
| Quantized M4096 | GEMM + low-rank up | 268.448 | 259.488 | 257.984–267.872 | 2.57% |
| Held-out attention L1024 | attention | unsupported | 34.272 | 32.672–35.488 | — |
| Held-out quantized M2048 | quantize + low-rank down | unsupported | 9.728 | 9.408–10.208 | — |
| Held-out quantized M2048 | GEMM + low-rank up | unsupported | 131.104 | 130.880–139.200 | — |

All three errors above 10% come from the M1024 quantization kernel. Its single calibration sample, 8.032 µs, is lower than every new observation. The measurements show that one sample can produce a persistent bias; this experiment does not establish its cause. Clocks remained unlocked and caches were flushed by NCU. A future repeated-calibration experiment must collect a separate validation set rather than replacing this frozen value using these results.

## Reproduction and checks

Use `tests/integration/tooling/evaluate_probe.py --replay /absolute/path/build/replay_probe --conditions 'audited input and capture settings' --calibration CALIBRATION_PATHS --validation VALIDATION_PATHS`. Each path names `kernels.probe` beside its original `capture.ncu-rep`, `hardware.csv` and `capture.txt`. See the README command example. The tool reports aggregate metrics and every observation; it never reads validation durations while freezing the calibration model.

Raw evidence is retained on gongji under `/home/gongji/0z5a/work/fake-cuda-device-profile/evidence/prediction-accuracy/`: `frozen-calibration.tsv`, `protocol.txt`, `*-r{0,1,2}/`, `accuracy.md`, workload correctness outputs and regression logs. Calibration paths are recorded in the frozen manifest. Generic evaluator code, tests and this Markdown report are committed; raw logs, JSON and binaries are not.

| Validation | Result |
|---|---|
| Full registered no-GPU contract suite | 27/27 passed |
| Accuracy contract | Hand-computed metrics, median freezing, unsupported coverage, report leakage/duplication and tampered-report rejection passed |
| Native captures and numerical checks | 24/24 fresh processes passed |
| Independent report observations | 36 validated; 27 scored, 9 unsupported |
| Saved four-device 5060 Ti system and graph regression | Attribute/peer checks and 8/8 graph processes passed |

No environments were updated, processes killed, or model files downloaded. These results cover one GPU, one fixed input seed and NCU's stated measurement conditions. Cross-device, input-content, unseen-shape and end-to-end timing accuracy remain unmeasured or unsupported.
