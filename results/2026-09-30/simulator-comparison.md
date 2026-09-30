# Comparison with published simulators

The current 3.654% MAPE measures reuse of frozen durations on 27 independent observations of nine previously calibrated kernel/configuration keys. Two unseen shapes contribute nine unsupported observations. It does not establish parity with learned kernel predictors or end-to-end simulators. Other systems' numbers below are reported by their authors; these systems were not run head-to-head in this experiment.

| System | Prediction and validation scope | Reported result | Difference from this iteration |
|---|---|---|---|
| FakeCUDA | Frozen kernel durations on one RTX 5090 and fixed input seed | MAPE 3.654%, P95 APE 13.746%; coverage 27/36, unseen-shape coverage 0/9 | Independent same-configuration validation; no cross-shape model or serving-latency validation |
| Maya v2 | Random-forest kernel predictors and distributed training simulation | H100 GEMM MAPE 3.65% and 2.22%; end-to-end runtime error reported below 5% | Random 80:20 kernel train/test split; about 42k training points for heavy-hitter kernels, plus complete execution/dependency modeling |
| Revati v1 | Real vLLM/SGLang control flow with coordinated virtual time; three model configurations | TTFT/TPOT median prediction error below 5%; 5–17× execution speedup over real GPU runs | Serving distributions and causality are evaluated, rather than isolated kernel durations |
| SGLang #33824 | Real scheduler/cache/request lifecycle with AIC, ML or batch replay forward-latency prediction | Qwen3-8B TTFT errors 2.38–4.15%; four long-context traces have mean-TTFT MAPE 5.89% | Forward and serving metrics, cache behavior and logical-time execution are integrated |
| Accel-Sim 2.0 | Dynamic SASS traces and detailed GPU microarchitecture modeling | Official H100 validation reports 13.4% mean absolute cycle error over 34,000+ kernel instances | Hardware-cycle modeling and counter correlation; NCU aggregate-duration reuse does not provide this capability |

Sources: [our independent accuracy experiment](prediction-accuracy.md), [Maya v2 Appendix B and evaluation](https://arxiv.org/html/2503.20191v2), [Revati v1 §6](https://arxiv.org/html/2601.00397v1), [SGLang PR #33824](https://github.com/sgl-project/sglang/pull/33824), and [Accel-Sim official validation](https://accel-sim.github.io/). Public comparison checked on 2026-09-30. SGLang's integration and predictor scope can also be inspected at its [fixed merge revision](https://github.com/sgl-project/sglang/blob/59799a368793b9f795baf59b067233a65ad8e38e/tools/sglang-simulator/README.md).

These percentages cannot be ranked as one benchmark. Maya also reports large relative errors on some short kernels without a comparable end-to-end impact. Revati's median serving-latency error, SGLang's per-trace errors, Accel-Sim's cycle error and our conditional kernel MAPE have different targets and aggregation. Likewise, native CUDA Graph speed ratios in our compiler report do not measure simulator execution speed.

The useful distinction is architectural. FakeCUDA retains Driver calls, compiled-image evidence and kernel accounting as its foundation. SGLang replaces model forward execution while preserving its own scheduler and cache. Maya adds trace collation and learned operation timing; Revati coordinates live actors' virtual time. Accel-Sim models execution below the kernel boundary. A low-level interface alone does not establish broader validated coverage.

## Next comparable experiments

1. Separate calibration and validation by shape, not merely by process. Report MAPE, duration-weighted error, tail error and unsupported coverage together; keep hardware and compiled-code identities explicit.
2. Replay the same complete workload trace with measured kernel durations and then with predicted durations. The first experiment tests scheduling/accounting fidelity; the second adds predictor error. Include host, transfer and synchronization costs.
3. After integrating real request scheduling, compare TTFT, TPOT, throughput and simulator wall time on matched hardware/model/workload settings. Keep predictor query wall time outside target-system time.

These are future validation steps, not completed results. This submission establishes the probe/evidence path and independent same-configuration timing evaluation.
