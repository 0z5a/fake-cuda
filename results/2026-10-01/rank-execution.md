# Independent Scheduler processes and rank lifecycle reuse

This follow-up adds independent CPU rank processes to the controlled U1 adapter. Each process owns the original vLLM 0.30.0 Scheduler/KV manager and fixed-length token control ledger. Commands and replies carry a monotonic sequence, run epoch and virtual timestamp. All rank replies must arrive before the coordinator certifies their safe points; delayed IPC does not move virtual token timestamps. Every new epoch reconstructs the original Scheduler, KV state and ledger. Pipes close on EOF and processes exit naturally.

The two-process contract completes 40 requests / 160 oracle tokens with arrival during service. Adding 1 ms to every rank reply preserves the entire local token/batch timeline, exact arrivals and 320 verified control reads. Repeated epochs and paced/coordinated runs match; an out-of-order command produces protocol failure. CPU campaign entry points hide CUDA before framework imports, and spawned ranks require zero visible devices. These are registered synchronous rank actors, not interception of arbitrary OS/runtime clocks or numerical GPU execution.

## Complete process speed

Three repeats per mode in alternating order; 3 complete retained workloads per process, with CUDA hidden in both modes. Both modes use identical source, original vLLM schedulers, frozen costs, rank transport and target work. Fresh starts all CPU ranks for each workload; reused starts once, then resets every Scheduler, KV state and control ledger under a new run epoch. Outer wall time includes interpreter/imports, fitting, every rank startup, scheduling, bridge IPC, local parity oracle, and natural teardown. Prior native calibration/download/build are excluded.

| Complete workloads | Fresh ranks s | Reused ranks s | Speedup | Wall reduction |
|---:|---:|---:|---:|---:|
| 3 | 37.760 | 20.832 | 1.813× | 44.83% |

| Repeat | Fresh ranks s | Reused ranks s |
|---:|---:|---:|
| 1 | 39.637 | 20.832 |
| 2 | 37.760 | 20.826 |
| 3 | 37.414 | 21.188 |

All six runs have identical virtual finish, token, batch, arrival and control-read digest: `8223e059cf606a42e038fa3cecdfd4a2be0c6f0c8ea776d0f4bee972f5bf1da9`. CPU rank PID and IPC wall delay are excluded from the digest. This measures timing simulation execution, not numerical inference acceleration.

The benchmark uses **one CPU rank** and the first three retained TP1 validation cases (burst-128, two-waves-96, burst-128; max seq 8, P64/O64). The two-rank process correctness fixture is separate. The performance comparison measures the new protocol with fresh ranks versus the same protocol with explicit epoch reuse; it is not a comparison against the old single-process adapter. Each measured subprocess additionally checks its complete targets against the local original-scheduler oracle.

## Retained independent prediction validation

Calibration and native validation are the unchanged, independently collected September 30 TP1 inputs. This is CPU regression on the new machine, not a new hardware calibration or TP2 accuracy claim. All 18 workload timelines match their local protocol reference exactly. Both modes use whole-step costs including host/communication; no extra kernel/NCCL time is charged.

| Coverage | Step WAPE | Step MAPE | Completed workloads | ≤10% duration / throughput / median TTFT / median ITL / P95 TTFT / P95 ITL |
|---:|---:|---:|---:|---|
| 9408/9408 | 0.51% | 0.49% | 18/18 | 18/18 / 18/18 / 18/18 / 18/18 / 18/18 / 18/18 |

| Workload | Max seq | Duration error | Throughput error | TTFT median error | ITL median error | TTFT P95 error | ITL P95 error |
|---|---:|---:|---:|---:|---:|---:|---:|
| burst-128 | 8 | -0.53% | +0.53% | -0.79% | -0.15% | -0.57% | -1.74% |
| burst-128 | 16 | -0.25% | +0.25% | -0.60% | +0.15% | -0.25% | -1.09% |
| burst-128 | 32 | -0.50% | +0.50% | -1.19% | +0.05% | -0.63% | -1.39% |
| two-waves-96 | 8 | -0.45% | +0.45% | -0.55% | -0.11% | -0.47% | -1.33% |
| two-waves-96 | 16 | -0.35% | +0.35% | -0.60% | +0.00% | -0.37% | -1.25% |
| two-waves-96 | 32 | -0.56% | +0.57% | -1.35% | +0.07% | -0.78% | -1.64% |

Rows are medians of three retained native observations, with P95 measured per run by linear interpolation. They do not establish tail confidence bounds. The frozen predictor, original arrivals and fixed output budgets are unchanged; native add-request/outer-loop overhead remains outside the step boundary. The masked local adapter also completes all 18 cases with target digest `860b206438e601d421f622b685a62985e6a12ca208aaa5d60e0d1498b7e512c5`; the masked Graph/KV contract retains the original manager's two-request versus one-request admission result.

## Native TP2 validation and observed conditions

The machine has two A100 SXM4 40 GB devices, driver 580.105.08, CUDA 13.0, Torch 2.13.0+cu130 and vLLM 0.30.0. Its observed topology is **SYS across two NUMA nodes**, with directed Driver peer access advertised for both pairs. Before this task, existing workers occupied both GPUs at 100% utilization and had stopped progressing at NCCL initialization.

The collector now accepts TP1/TP2 and launches each native TP rank through vLLM's `external_launcher`. It retains the original numerical worker, synchronizes native admission decisions across ranks, and checks batch/token counts across native ranks after each workload. The parent only waits for natural exit. An explicit `--transport socket` mode disables P2P/SHM/custom all-reduce and requests NCCL Socket transport; it is recorded as a separate collection condition.

Both Socket campaigns completed model loading, 24 calibration runs and 18 independent numerical validation workloads, then exited naturally. Each workload checks native rank batch/shape and token-count agreement. CPU process/local replay timelines also agree on every scored workload. This checks the controlled scheduler boundary; the CPU oracle does not reproduce numerical token IDs.

| Native TP2 Socket campaign | Native campaign s | Calibration runs / steps | Scored workloads / steps | Step WAPE | Step MAPE | Accuracy target |
|---|---:|---:|---:|---:|---:|---|
| First; no per-workload background snapshots | 2133.006 | 24 / 2304 | 18 / 9408 | 86.61% | 163.60% | Fail |
| Second; matching background snapshots | 864.578 | 24 / 2304 | 12 / 5440 | 44.81% | 31.08% | Fail |

The second collector snapshots other GPU process IDs, UUIDs and allocated memory before/after each workload, excluding its native rank PIDs. Seven calibration workloads changed conditions and are discarded before fitting. Six validation workloads changed conditions and are unsupported, retaining their 3968 steps in the denominator. Every remaining case uses a separately frozen model for its observed background condition. Shape extrapolation, unseen conditions and incomplete snapshots also fail coverage explicitly. Older input with neither snapshot remains a separately identified legacy condition; it is not evidence of equivalent GPU load. Empty coverage reports unknown error, not zero error. The condition contract confirms separate frozen models, unchanged supported predictions despite a 1000× changed-condition calibration sample, snapshot-order invariance, missing/unknown/legacy validation rejection, and refusal to publish a speed table with skipped workloads.

The snapshots show transient GPU processes during collection. Their provenance and effect on latency were not established. GPU-visible CPU imports are one possible source; hiding devices before all new CPU rank imports removes that exposure. Even matching endpoint snapshots cannot establish unchanged utilization, clocks or communication contention within a workload. The default TP2 workers remained present in both campaigns, and their interference is not modeled.

The two campaign errors are **not an improvement comparison**: they use different observations and coverage. On the exact same 12-workload / 5440-step subset of the second campaign, pooling all calibration samples gives **44.22% WAPE**, versus **44.81%** with condition isolation. The new coverage guard prevents unsupported claims; it does not solve TP2 timing prediction. No validation duration is used to fit either model.

| Second campaign workload | Max seq | Scored repeats | Duration error | Throughput error | Median TTFT error | Median ITL error | P95 TTFT error | P95 ITL error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| burst-128 | 8 | 2 | -32.34% | +48.16% | -31.70% | -4.29% | -31.70% | -77.97% |
| burst-128 | 16 | 1 | -38.06% | +61.45% | -37.33% | -25.26% | -36.57% | -79.43% |
| burst-128 | 32 | 3 | -47.28% | +89.67% | -50.40% | -5.61% | -45.24% | -79.01% |
| two-waves-96 | 8 | 1 | -31.31% | +45.59% | -30.65% | -8.84% | -32.09% | -75.57% |
| two-waves-96 | 16 | 2 | -35.08% | +54.14% | -34.31% | -19.73% | -32.65% | -76.40% |
| two-waves-96 | 32 | 3 | -26.40% | +35.88% | -20.01% | +20.29% | -30.26% | -78.60% |

Rows are medians over the scored repeats. Within 10%: duration 0/12, throughput 0/12, median TTFT 1/12, median ITL 6/12, P95 TTFT 1/12, P95 ITL 0/12. TP2 has not met the proposed 10% target. Communication/host timing remains included in the measured whole step; no extra NCCL duration is added.

The default TP2 attempt has not passed communicator initialization. This task stopped no process, updated no base environment, and accessed no NFS path. That incomplete attempt still needs its task-owned checkpoint; completed September 30 weights were already removed. Both completed Socket campaigns and CPU results are backed up locally outside git. The collector now prints workload start/completion explicitly; sparse earlier logs did not establish a blocked inference step.

| Retained input | SHA-256 |
|---|---|
| rank-masked-validation.json | `25d747c45c50630223c81811cbd855963542c57a28afc1b8b423c3caf9fa713b` |
| rank-masked-speed.md | `be4c2eed97602d63f5ef041497467c9b373cc70d2abf5e83b249812b64344a0f` |
| reference-scheduler-config.pkl | `a97291111c939492577cd218137449c7221ebd6f13495203300aa7735c53a0bf` |
| tp2/device-0.profile | `a069d5d8eff9e066ef21e86bff0d975d8b255d83947b7899aca66846df274dc6` |
| tp2/device-1.profile | `ffac3e46d81e922fa7e901a27c5b1aa17329e6930bf75c6a798055c1544f80b0` |
| tp2/device.system | `d6bce8a93934d414c687ea1b18cb176f5a32acf5d570561f39fb7f9ad3149767` |
| tp2-socket/scheduler-config.pkl | `7c2767e20cf4f8d193b744c890f8373b0730b2b36c2c3ca27df9af3d0e80a3ef` |
| tp2-socket/calibration.json | `9af0da6f3b2e61a21fa8c2a3ac36a51d5cbe4dd5bcead831ec24519e2b7c818f` |
| tp2-socket/validation.json | `83d2ce7b17f0511771d24a712d9d22d816c58e0436d22be7266b45c75948dc51` |
| tp2-stable-socket/scheduler-config.pkl | `59f50b718d19c451f0f3028971ee6a8a915c47c16552079c85b1e3364c9f02e3` |
| tp2-stable-socket/calibration.json | `9e78a2fef44e26c391af3eff2d00db46c4334f450ba41b7abcae2f48b409a1e0` |
| tp2-stable-socket/validation.json | `b9f86a5f5fa97e7ffa51c3e42aa2a6db7aeb7b68a5da3c84993169173578e596` |
| tp2-qualified-final-validation.json | `339c3890930bc04b4b2361d08ff8c87ef48e5aaa3af82348337e77e8ef45544c` |
| resource_engine_bridge | `03311606d35a97a852768e273e94586a4088cf4a1e24de59ae4030759bbfa57d` |
| libcuda.so.1 | `7c2d376ebb41f5410af6168efe3513265e0e1a0fa1424b9b1f7f307704642404` |

All 18 process/local target timelines: `ed6c4d6c8dc9b3c4887c932c77a3939e6593e20dece89faf2b591cf6f984e51a`. Original calibration/validation identities and numerical collection conditions are retained in [the TP1 accuracy report](../2026-09-30/serving-validation.md). Logs, JSON, profiles, binaries and checkpoints are not submitted.
