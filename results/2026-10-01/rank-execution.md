# Independent Scheduler processes and rank lifecycle reuse

This follow-up adds independent CPU rank processes to the controlled U1 adapter. Each process owns the original vLLM 0.30.0 Scheduler/KV manager and fixed-length token control ledger. Commands and replies carry a monotonic sequence, run epoch and virtual timestamp. All rank replies must arrive before the coordinator certifies their safe points; delayed IPC does not move virtual token timestamps. Every new epoch reconstructs the original Scheduler, KV state and ledger. Pipes close on EOF and processes exit naturally.

The two-process contract completes 40 requests / 160 oracle tokens with arrival during service. Adding 1 ms to every rank reply preserves the entire local token/batch timeline, exact arrivals and 320 verified control reads. Repeated epochs and paced/coordinated runs match; an out-of-order command produces protocol failure. These are registered synchronous rank actors, not interception of arbitrary OS/runtime clocks or numerical GPU execution.

## Complete process speed

Three repeats per mode in alternating order; 3 complete retained workloads per process. Both modes use identical source, original vLLM schedulers, frozen costs, rank transport and target work. Fresh starts all CPU ranks for each workload; reused starts once, then resets every Scheduler, KV state and control ledger under a new run epoch. Outer wall time includes interpreter/imports, fitting, every rank startup, scheduling, bridge IPC, local parity oracle, and natural teardown. Prior native calibration/download/build are excluded.

| Complete workloads | Fresh ranks s | Reused ranks s | Speedup | Wall reduction |
|---:|---:|---:|---:|---:|
| 3 | 40.911 | 21.525 | 1.901× | 47.39% |

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

Rows are medians of three retained native observations, with P95 measured per run by linear interpolation. They do not establish tail confidence bounds. The frozen predictor, original arrivals and fixed output budgets are unchanged; native add-request/outer-loop overhead remains outside the step boundary.

## New machine native TP2 attempt

The machine has two A100 SXM4 40 GB devices, driver 580.105.08, CUDA 13.0, Torch 2.13.0+cu130 and vLLM 0.30.0. Its observed topology is **SYS across two NUMA nodes**, with directed Driver peer access advertised for both pairs. Before this task, existing workers occupied both GPUs at 100% utilization and had stopped progressing at NCCL initialization.

The collector now accepts TP1/TP2 and launches each native TP rank through vLLM's `external_launcher`. It retains the original numerical worker, synchronizes native admission decisions across ranks, and checks batch/token counts across native ranks after each workload. The parent only waits for natural exit. An explicit `--transport socket` mode disables P2P/SHM/custom all-reduce and requests NCCL Socket transport; it is recorded as a separate collection condition.

The default TP2 attempt has not passed communicator initialization. The Socket attempt completed initialization, model loading and all 24 calibration runs; independent validation is progressing. The existing collector only logged completed validation workloads, so the earlier long interval without output did not establish a blocked inference step. **Native TP2 accuracy remains unvalidated until the independent campaign finishes**. No process was stopped, no base environment was updated, and no NFS path was accessed. Incomplete native attempts still retain their task-owned checkpoint; the completed September 30 model checkpoint was already removed. Raw evidence is backed up locally outside git. Future collector runs now print workload start/completion explicitly.

| Retained input | SHA-256 |
|---|---|
| rank-validation.json | `f933694319450339da83f576ea8b8deeb625332400c1b29f590a769d452b1b87` |
| rank-speed.md | `a58d61bb9ebc3cc8723bc05c0d922174d9a60de88f4378a303ea138d98858a80` |
| reference-scheduler-config.pkl | `a97291111c939492577cd218137449c7221ebd6f13495203300aa7735c53a0bf` |
| tp2/device-0.profile | `a069d5d8eff9e066ef21e86bff0d975d8b255d83947b7899aca66846df274dc6` |
| tp2/device-1.profile | `ffac3e46d81e922fa7e901a27c5b1aa17329e6930bf75c6a798055c1544f80b0` |
| tp2/device.system | `d6bce8a93934d414c687ea1b18cb176f5a32acf5d570561f39fb7f9ad3149767` |
| tp2-socket/scheduler-config.pkl | `7c2767e20cf4f8d193b744c890f8373b0730b2b36c2c3ca27df9af3d0e80a3ef` |

All 18 process/local target timelines: `ed6c4d6c8dc9b3c4887c932c77a3939e6593e20dece89faf2b591cf6f984e51a`. Original calibration/validation identities and numerical collection conditions are retained in [the TP1 accuracy report](../2026-09-30/serving-validation.md). Logs, JSON, profiles, binaries and checkpoints are not submitted.
