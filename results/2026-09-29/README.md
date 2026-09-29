# CUDA Graph validation — RTX 5090

Baseline: `monsoon235/fake-cuda@75fd2cd27713373baef17007547a5dda03219223`. Changes are on `pcie-test`. Tests use the existing `/home/gongji/0z5a` environment: PyTorch 2.13.0+cu130, CUDA 13, driver 580.82.07, RTX 5090 device 7. No environment packages, clocks, or power settings were changed. Raw outputs and external operator adapters are archived locally; only this report and general test tools are submitted.

## Graph contracts

| Contract | Baseline | Updated implementation | Fixture |
| --- | --- | --- | --- |
| Same GraphExec launched on different streams; 80 ms H2D + 80 ms D2H | FAIL: 80 ms completion separation | PASS: 160 ms | `graph_ordering_probe` |
| Independent GraphExec objects | No added serialization needed | PASS: 80 ms separation | `graph_ordering_probe` |
| Capture does not submit; template/exec lifetime; event re-record snapshot | Existing/extended coverage | PASS | Driver contract probes |
| Thread-local mode, owner thread, invalidation and recovery | Missing/incomplete behavior | PASS against real and fake Driver | `capture_mode_probe` |
| Cross-stream copy/add capture, four replays, changed inputs and saved outputs | Capture initialization / node-priority flags blocked | No-GPU control flow PASS; real-CUDA values PASS | `graph_contract.py` |
| Native ABI regression suite | — | 10/10 invocations PASS | Standalone Driver probes |

No-GPU checks run in an offline Docker container with `--runtime=runc --network=none --pull=never`, read-only environment mounts and no GPU devices. The 160 ms result fixes invalid overlap; it is not a speedup. The independent-exec overlap is a simulator heuristic. Real-Driver conformance covers valid capture modes; malformed/null mode arguments are excluded. Nonzero captured priorities and captured allocation/free remain unsupported.

## Eager versus Graph measurements

The following external-operator measurements validate real-CUDA capture/replay, including changed inputs at stable addresses. Three fresh processes per reported group use four alternating APPA/PAAP blocks and 100 calls per arm, with compilation, capture and warmup outside timing. Times include host submission and final completion. Entries are medians of per-process medians; speedups are medians of per-process ratios. They are historical operator measurements, not a speedup from the simulator patch or full-model E2E.

| Workload | Case | Eager µs | Graph µs | Speedup |
| --- | --- | ---: | ---: | ---: |
| Marlin W4A16 | M=1 | 31.387 | 6.334 | 4.801× |
| Marlin W4A16 | M=16 | 31.423 | 6.350 | 4.811× |
| Marlin W4A16 | M=64 | 31.617 | 10.463 | 3.020× |
| FA4 | 512 tokens, noncausal | 64.400 | 16.593 | 3.882× |
| FA4 | 2048 tokens, noncausal | 65.697 | 53.507 | 1.228× |
| FA4 | 2048 tokens, causal | 64.917 | 57.683 | 1.126× |
| SVDQuant W4A4 + rank-32 | M=256 | 134.951 | 131.326 | 1.028× |
| SVDQuant W4A4 + rank-32 | M=1024 | 135.431 | 131.409 | 1.031× |
| SVDQuant W4A4 + rank-32 | M=4096 | 262.850 | 260.434 | 1.009× |

Linear cases use K=1024/N=2048, FP16 activation/output and group sizes 128 (W4A16) or 64 (W4A4). The latter includes activation quantization and a nonzero low-rank branch through official core objects with a small test binding. Attention uses batch 1, eight heads and head dimension 64. Reference checks pass with atol=rtol=0.02 for linear cases and 0.003 for attention.

The W4A4 rows are a follow-up after another scheduler became resident on GPU 7 and the device was observed idle. Initial shared-GPU runs ranged from 0.767× to 1.052×, including regressions; all six runs remain in the local archive. Later improvements are conditional shared-GPU observations. Attention's observed per-process ranges were 3.788–3.984×, 1.176–2.090× and 1.100–1.160× respectively; CPU compilation overlapped those measurements.

External binaries still encounter missing TMA encoding or function/kernel attribute APIs (`cuTensorMapEncodeTiled`, `cuKernelGetAttribute`, `cuFuncSetAttribute` / `cuKernelSetAttribute`) before capture under FakeCUDA. Import or allocation success does not establish numerical execution. No success stubs, model checkpoints, or calibrated kernel timing were added; virtual kernels retain their existing 10 ms duration.

The refactored generic runner was verified on real CUDA with preallocated copy/add outputs and an external matrix-multiplication callback that allocates its output. All eager/reference and changed-input replay checks passed.

## General test entry points

```sh
# Existing environment; no package installation required.
cmake -S . -B build -DCUDA_DRIVER_INCLUDE_DIR=/usr/local/cuda/include
cmake --build build -j4
build/graph_ordering_probe build/libcuda.so.1 same
build/graph_ordering_probe build/libcuda.so.1 independent
build/capture_mode_probe build/libcuda.so.1
# Explicit physical Driver ordinal:
build/capture_mode_probe /usr/lib/x86_64-linux-gnu/libcuda.so.1 7
CUDA_VISIBLE_DEVICES=7 /home/gongji/0z5a/bin/python tests/integration/pytorch/graph_contract.py --mode real --kernel add
CUDA_VISIBLE_DEVICES=7 /home/gongji/0z5a/bin/python tests/integration/tooling/benchmark_graph.py --operation add
```

`graph_contract.py --mode fake --kernel copy|add` belongs in the no-GPU container with this project's `libcuda.so.1` preloaded. The real-only `benchmark_graph` function accepts three typed callbacks: run the operation, compute an independent reference, and update inputs in place. It validates eager output and three changed-input replays before alternating eager/Graph timing. External adapters supply those callbacks without adding package-specific branches or build dependencies to this repository.
