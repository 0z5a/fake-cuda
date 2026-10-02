# Short-trace CPU optimization and whole-model validation

The short selection trace was dominated by report serialization rather than virtual device work. A 4000-operator CPU profile spent 0.537 s in report digests, including 0.447 s in recursive `dataclasses.asdict` conversion, out of 0.971 s total instrumented execution. The report now passes dataclass fields directly to the JSON encoder. Canonical bytes, SHA-256 digests, virtual timelines and accounting remain identical. These instrumented times identify the bottleneck; the table below uses separate, unprofiled runs.

## Equal-target complete process comparison

Both versions run offline with the same frozen stage profile and held-out trace. The baseline is `b6637b3`; the candidate changes only report serialization. Five independent APPA/PAAP sessions launch 20 fresh processes per trace size, ten per version, on the AMD EPYC 9K84 host with Python 3.12.3. Wall time includes interpreter startup, imports, profile/trace loading, replay, hashing, actor delivery, output and natural teardown. Native calibration and profile fitting are excluded equally from both arms. CUDA is hidden in every CPU process.

| Operators per process | Before, median s | After, median s | CPU speedup | Wall-time reduction | Session speedup range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 0.091140 | 0.079732 | 1.143× | 12.52% | 1.116–1.179× |
| 4000 | 0.376063 | 0.267963 | 1.403× | 28.75% | 1.374–1.463× |

All 40 processes preserve their corresponding timeline/ledger/reply digests. Every session improves at both sizes. This comparison measures the source change; the earlier [paced/offline comparison](moe-serving.md) measures a different change and cannot be substituted for this baseline. Neither comparison measures numerical neural-network acceleration.

| Evidence | SHA-256 |
| --- | --- |
| Frozen profile | `2dfcc29b95016ecb5c42fc20e3a6da08a92e200bab003a23df4cb543564c8d08` |
| Held-out trace | `47ccea2b9608bbcd1135753819398ab709d09104e2f87c403c5fb46c58aecd9f` |
| 400-operator target | `e687baa09f7059ba0612dd232d72db03c602aa017d321a8365640a0dbe876eef` |
| 4000-operator target | `45743e2352b04101e0fc2fe78d49c8cc7b6ce9f52306229f6aaf3a62dd1e8476` |

The canonical-byte regression covers MoE, KDA, selection and unknown operators across fresh execution IDs. All 13 Python semantic cases and all 34 registered contracts pass. Raw profiles, timings, logs and JSON stay outside Git.

## Whole-model protocol

The checkpoint is [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/tree/e1df551a447157d4658b573f9a695d57658590e9), revision `e1df551a447157d4658b573f9a695d57658590e9`. All 20 shards, totalling 98,248,224,120 file bytes, must match the immutable HF LFS SHA-256 values before native inference starts. ModelScope's per-file checksums match this revision; the mirror changes the transport only. The architecture contains all 27 layers: 20 KDA, seven full-attention layers, one initial dense FFN and 26 MoE FFNs.

The private environment uses vLLM 0.30.0, PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.18.0 and FlashInfer 0.6.18.post1. The host has two H20s connected by NV18; trained-checkpoint validation uses physical GPU 1, UUID `GPU-8ee84e7d-143f-dd29-1097-85943783e027`, with TP1. FlashInfer's NVCC JIT uses the separate NVCC 13.0.88 / runtime-header 13.0.96 toolkit; Triton uses its own compiler and selected PTX assembler, whose binary hash is recorded separately. Runtime 13.0 headers paired with the initially supplied NVCC 13.4 failed CCCL's version check. Matching toolkit files, runtime library aliases and cuRAND headers pass the complete architecture preflight; its random-weight results are excluded from checkpoint validation.

Native execution uses the complete BF16 checkpoint without quantization or CPU offload, eager synchronous scheduling, prefix/chunked prefill disabled, maximum eight sequences and model length 512. Loaded parameters occupy 91.55 GiB. An explicit 512 MiB KV budget provides 4352 tokens / 8.50 requests at the declared maximum length. The native engine selects Triton unquantized MoE and FlashAttention MLA; TP1 does not measure TP communication. The separate random-weight TP2 preflight uses 1 GiB KV per rank and exercises FlashInfer MNNVL all-reduce fusion. Loaded library/source/toolchain hashes and actual visible GPU UUIDs are retained per rank. Whole-step measurements include host work and cannot be installed as Driver kernel-service costs.

On this SM90 target, Triton's selected assembler reports CUDA 12.8.93 despite the CUDA 13 PyTorch runtime and FlashInfer toolkit. This is an independently bundled code-generation path, not the earlier NVCC/header mismatch. System `nvcc` alone does not identify the compiler for every executed kernel.

Calibration records batches 1/4/8 × prompts 32/64/128 × 64 output tokens, twice: 18 workloads, 1152 measured engine steps and 4992 generated tokens. The resulting table is written before warming held-out shapes. Validation uses batches 2/3/6 × prompts 48/96/112 × 32 output tokens, three times: 27 workloads, 864 steps and 3168 generated tokens. Those durations never enter fitting. The collector retains native token IDs and compares batch/token digests across ranks when TP exceeds one. Two additional chat questions check real generation; fixed-length timing workloads ignore EOS.

CPU replay verifies the trusted scheduler configuration and exact frozen table before opening validation. One independent original-Scheduler/KV child process replays all 27 TP1 workloads and asserts exact batch membership and declared arrivals. CPU token values come from the explicitly labelled fixed-length oracle; only the native engine produces numerical checkpoint outputs. The native and CPU processes finish naturally. Model downloads, logs and raw JSON remain outside the submitted source tree.

## Checkpoint E2E and prediction accuracy

The complete trained checkpoint passes native TP1 prefill/decode and real text generation. Asked whether 123 is prime, it returns `No.`; asked for the capital of France, it returns `Paris`. All 27 CPU workloads preserve native batch membership and declared arrivals. Coverage is 864/864 steps, with zero unknown costs. Frozen whole-step prediction has **1.40% median error, 6.00% P95, 2.15% MAPE and 2.20% WAPE**. Median absolute complete-workload duration error is 1.61%.

Each row is the median of three held-out workloads. CPU request replay wall includes that request's scheduler reconstruction, prediction, IPC, bridge execution and token-oracle delivery, with imported/fitted state and the rank process already resident. It excludes process startup and fitting; it is timing simulation, not numerical inference. Predicted seconds and tokens/s use the virtual timeline.

| Batch | Prompt tokens | Native seconds | CPU request replay seconds | Predicted seconds | Absolute duration error | Native tokens/s | Predicted tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 48 | 1.242824 | 0.017610 | 1.219173 | 1.90% | 51.50 | 52.49 |
| 2 | 96 | 1.236623 | 0.017936 | 1.216656 | 1.61% | 51.75 | 52.60 |
| 2 | 112 | 1.264415 | 0.017687 | 1.217743 | 3.69% | 50.62 | 52.56 |
| 3 | 48 | 1.238744 | 0.018850 | 1.223690 | 1.22% | 77.50 | 78.45 |
| 3 | 96 | 1.252142 | 0.018211 | 1.220316 | 2.54% | 76.67 | 78.67 |
| 3 | 112 | 1.253467 | 0.018315 | 1.224242 | 2.33% | 76.59 | 78.42 |
| 6 | 48 | 1.260480 | 0.019912 | 1.246418 | 1.12% | 152.32 | 154.04 |
| 6 | 96 | 1.280059 | 0.020372 | 1.233319 | 3.65% | 149.99 | 155.68 |
| 6 | 112 | 1.250426 | 0.019960 | 1.239302 | 0.89% | 153.55 | 154.93 |

The native complete process takes 188.062 s, including imports, weight loading, JIT/warmups, calibration, validation, chat generation and natural teardown. The complete CPU validation process takes 15.199 s, including imports, fitting the retained calibration, child startup, all 27 replays, output and natural teardown. These processes perform different campaigns, so their total durations do not define an equal-work speedup. The report-serialization speedup above measures a separate source change.

| Checkpoint evidence | SHA-256 |
| --- | --- |
| Trusted scheduler configuration | `b9e3e8bd8bc66ace9ce1c45fbe7f6c2ac562280b2c16a25c6a3b0c4ec40f6ad3` |
| Calibration capture | `617cb6d6320549f980f26d7f2baf71cdd7fc2241adb53ed1fbdf6e9434e321ab` |
| Table frozen before holdouts | `c10b436a1548fec558433933cef024724065869f42198c32a3ed4d81aaff7295` |
| Independent validation capture | `4c488e0d290d6ec122acc70a8ec21b1ab29cae7923695b2d7471fa4eb5479d45` |
| CPU timeline/token-oracle/batch target | `5a61016c01491e6dbbb8fd6e346078ccc001971f4b93d223c7cbd65f49d6cd22` |

Trained TP2 remains unvalidated. Two starts failed during memory admission before weight loading, with 9.72 and 12.50 GiB free on the occupied rank; another task retains 87152 MiB on physical GPU 0. That same process/memory snapshot appears before and after every TP1 workload on the other physical GPU. It is retained background evidence, not proof of contention stability or unseen-machine accuracy. No process was terminated to obtain a window. The collector now records explicit KV bytes and maps the CUDA-visible UUID correctly; GPU admission fraction 0.99 and KV bytes 536870912 describe the successful TP1 run. vLLM skips memory profiling when explicit KV bytes are supplied; the fraction still gates admission and does not cap this allocation.

The duplicate download files were removed after their writers exited naturally, releasing 43.8 GiB. After native/CPU completion and Mac evidence backup, the cleanup worker removed all 20 trained checkpoint shards (98,248,224,120 bytes). No model weights, incomplete downloads or assembly parts remain in the task's model directories; configuration/tokenizer files remain for CPU reproduction.
