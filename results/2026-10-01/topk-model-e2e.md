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

The private environment uses vLLM 0.30.0, PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.18.0 and FlashInfer 0.6.18.post1 on two H20s connected by NV18. FlashInfer's NVCC JIT uses the separate NVCC 13.0.88 / runtime-header 13.0.96 toolkit; Triton uses its own compiler and selected PTX assembler, whose binary hash is recorded separately. Runtime 13.0 headers paired with the initially supplied NVCC 13.4 failed CCCL's version check. Matching toolkit files, runtime library aliases and cuRAND headers pass the complete architecture preflight; its random-weight results are excluded from checkpoint validation.

Native execution uses BF16 TP2, eager synchronous scheduling, prefix/chunked prefill disabled, maximum eight sequences, model length 512 and a 1 GiB KV budget per rank. The native engine selects Triton unquantized MoE, FlashAttention MLA and FlashInfer MNNVL all-reduce fusion. Loaded library/source/toolchain hashes and GPU UUIDs are retained per rank. Whole-step measurements already include host work and communication and cannot be installed as Driver kernel-service costs.

On this SM90 target, Triton's selected assembler reports CUDA 12.8.93 despite the CUDA 13 PyTorch runtime and FlashInfer toolkit. This is an independently bundled code-generation path, not the earlier NVCC/header mismatch. System `nvcc` alone does not identify the compiler for every executed kernel.

Calibration records batches 1/4/8 × prompts 32/64/128 × 64 output tokens, twice: 18 workloads and 1152 measured engine steps. The resulting table is written before warming held-out shapes. Validation uses batches 2/3/6 × prompts 48/96/112 × 32 output tokens, three times: 27 workloads, 864 steps and 3168 generated tokens. Those durations never enter fitting. Both native ranks compare exact batch membership and token IDs. Two additional chat questions check real generation; fixed-length timing workloads ignore EOS.

CPU replay verifies the trusted scheduler configuration and exact frozen table before opening validation. Two independent original-Scheduler/KV processes replay all 27 workloads and assert exact batch membership and declared arrivals. CPU token values come from the explicitly labelled fixed-length oracle; only the native engine produces numerical checkpoint outputs. The native and CPU processes finish naturally. Model downloads, logs and raw JSON remain outside the submitted source tree.

## Checkpoint execution status

All 20 checkpoint files have been downloaded and matched against HF SHA-256. Full-architecture random-weight TP2 prefill/decode and the hybrid original-Scheduler CPU rank check pass. Real-checkpoint execution has not completed: two starts failed during memory admission, before weight loading, with 9.72 and 12.50 GiB free on the occupied rank. The shared machine's other vLLM-Omni task holds about 85 GiB per card. No model timing or accuracy result is inferred from those starts.

The collector exposes the memory admission fraction separately from its explicit 1 GiB KV budget; the default fraction is now 0.55, sufficient for the measured approximately 49 GiB full-architecture rank footprint. It still requires actual free memory and cannot resolve the current 85 GiB occupancy. Checkpoint weights remain retained until the campaign completes and every reader/writer exits naturally.
