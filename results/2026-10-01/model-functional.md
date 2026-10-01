# Complete-checkpoint functional serving matrix

Status: Granite TP1/TP2 functional matrices verified; TP2+EP completes all native/replay cases but fails independent CPU token equality. Kimi's remaining current-memory matrices are pending their coordinated windows. This report extends the [trained TP1 prediction results](topk-model-e2e.md) and [serving/EP contracts](serving-ep.md). No pending native row is a passing E2E result. Native numerical inference is supplied by the original vLLM backend; CPU replay supplies explicit recorded control metadata, not logits or predictions of model quality.

## Checkpoints and execution scope

The original Kimi-Linear-48B-A3B-Instruct BF16 architecture and checkpoint remain the target: revision `e1df551a447157d4658b573f9a695d57658590e9`, 20 immutable shards, 98,248,224,120 bytes, 27 complete layers. Its previously qualified TP1 campaign is retained. The additional original Granite 3.0-1b-a400m-instruct checkpoint exercises generic serving and distributed paths; it does not replace Kimi validation.

Granite revision `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445` has 24 complete layers, 32 experts and top-8 routing. Its single BF16 weight file has 2,669,283,096 bytes and SHA-256 `f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc`. The original configuration, tokenizer and complete weights are used; no layer reduction, dummy initialization, quantization or CPU weight offload is used.

The private serving environment pins vLLM 0.30.0, Torch 2.13.0+cu130, Triton 3.7.1 and Transformers 5.18.0. Native runs record the actual model, runner and Scheduler source hashes, loaded vLLM/NCCL libraries, CUDA UUID, parameter bytes/dtypes and expert shard shapes. Graph evidence counts actual `torch.cuda.CUDAGraph.replay` calls. Request length is bounded by the functional workload, rather than changing the model architecture.

## Finite cases

| Case | Input | Required contract |
| --- | --- | --- |
| Single output | One request, prompt 48 / output 1 | Native token and natural drain |
| Uniform | Eight requests, prompt 128 / output 16 | Continuous batch, exact token conservation |
| Heterogeneous | Prompts 32/96/112/192, output 8 | Native schedule metadata and token parity |
| Two waves | Two prompt-64 requests; two prompt-128 requests after step 2 | Explicit release ordinal, mixed prefill/decode |
| Decode cancellation | Two prompt-48 requests; cancel first after step 2 | Remaining request finishes; matched-policy cancelled tokens match exactly |
| Prefill cancellation | Two prompt-192 requests; cancel first after step 1 | Chunk mode cancels before first output; exact original Scheduler replay |

Cancellation occurs after a synchronized native step. Concurrent in-flight cancellation and last-use resource release are separately covered by the serving contracts. Releases use completed-step ordinals; these cases do not measure production arrival timing or SLA.

Eager, full-decode Graph, chunk prefill and Graph+chunk each use their own naturally exiting process. Explicit KV capacity is 64 MiB for Granite, divided by TP ranks; chunk budget is 64 tokens, maximum running requests eight. Sampling is greedy, with EOS ignored for the bounded synthetic-token cases. Two natural-language prompts additionally compare the independent CPU reference with original native generation. Prefix caching, speculative decoding and asynchronous scheduling remain disabled.

## Current-memory eligibility and verified results

| Complete checkpoint / topology | Fits current residency | Native variants | Original Scheduler CPU replay | Independent numerical reference |
| --- | --- | --- | --- | --- |
| Original Kimi TP1, 512 MiB KV | Yes, replacement H20 idle capacity | Previous homogeneous E2E PASS; new four-variant matrix pending coordinated window | Previous 27 workloads PASS; new matrix pending | Previous native answers No / Paris |
| Original Kimi TP2 / EP | Replacement two-card capacity admits the SHA-verified full BF16 weights; native workspace still requires validation | Pending coordinated finite queue | Pending native controls | Pending native comparison |
| Original Granite TP1 | Yes | All four variants × six cases PASS | 268 scheduled steps, 906 recorded-control reads PASS | Two natural-language prompts × four variants exactly match complete BF16 Transformers CPU reference; no CUDA initialized |
| Original Granite TP2 | Yes, replacement two-card capacity | All four variants × six cases PASS | Both ranks: 536 scheduled steps, 1,812 recorded-control reads PASS | All eight natural-language outputs per rank exactly match the fresh complete BF16 CPU reference |
| Original Granite TP2+EP | Yes, replacement two-card capacity | All four variants × six cases complete with native status zero; strict independent-reference gate FAIL | Both ranks: 536 scheduled steps, 1,812 recorded-control reads PASS | Paris exactly matches; prime answer diverges at token index 28, zero based |

The full Granite CPU reference naturally exits with answers “No, 123 is not a prime number…” and “Paris.” Default calibrated vLLM adapter behavior remains unchanged: continued chunks still fail the homogeneous cost domain unless an explicit recorded-control replay is requested. Serving and EP CPU contracts pass 7/7 each.

The fixed TP1 queue naturally exits with native and CPU-replay status zero for every process. All 24 layers and every `[32,1024,1024]` expert shard are present, with 2,669,381,632 parameter bytes. The effective KV pool has 1,360 tokens; no reduced architecture or dummy weights are used.

| Granite TP1 mode | Cases | Scheduled steps | Recorded-control reads | Actual Graph replays | Peak Torch reserved GiB | Whole process seconds, including startup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager | 6 | 53 | 212 | 0 | 2.6230 | 96.759 |
| Graph | 6 | 53 | 212 | 44 | 2.6563 | 17.597 |
| Chunk | 6 | 81 | 241 | 0 | 2.6035 | 20.185 |
| Graph+chunk | 6 | 81 | 241 | 43 | 2.6367 | 18.366 |

These startup times are observations, **not a speed comparison**: the first process spends about 70 seconds in cold JIT preparation, later processes reuse private compiled caches, and native workloads are not a balanced fresh-process A/B campaign. No throughput ratio is computed from this table.

Within each prefill policy, Graph and eager output tokens match exactly. All eight natural-language outputs, including prompt and generated token IDs, exactly match the independent full-model CPU reference. Across full and chunked prefill, three synthetic repeated-token requests diverge; exact cross-policy greedy-token equality does **not** pass. Independent BF16 CPU teacher forcing evaluates the identical prefix at each first divergence:

| Case / request | First different output index, zero based | Full-prefill token | Chunk token | CPU top-1 / top-2 | CPU top-1 minus full-prefill selected logit |
| --- | ---: | ---: | ---: | --- | ---: |
| Uniform / 3 | 12 | 74 | 35 | 35 / 74 | 0.09375 |
| Uniform / 6 | 10 | 5370 | 10506 | 10506 / 5370 | 0.06250 |
| Two waves / 2 | 3 | 34 | 36 | 36 / 34 | 0.00000 |

Both native selections are among the CPU reference's top two at the first divergence; one is an exact BF16 tie. This localizes the difference, but does not measure native logit error or establish its cause. The simulator replays each variant's actual sampler controls and never replaces one trajectory with fabricated common tokens. Each variant preserves exact request token counts, original Scheduler batch membership, mixed/continued prefill metadata, cancellation and natural drain.

The original TP1 native collector SHA-256 is `9f4231add7efe35332d6010b5920ec08e0a0e1eacedcad6e7ae2b2bee643c582`. Its exact source snapshot is retained outside Git. Later comparison and CPU teacher-forcing additions did not change that campaign's collected native workload. No speed improvement or new prediction-accuracy claim follows from recorded native sampler replay. Existing independent whole-model timing accuracy and equal-target CPU simulator speed tables retain their original boundaries.

The replacement-host Granite TP2 matrix uses collector SHA-256 `d847f7eef9c729e2ad1e739b1f874498a2f80fa17e63fb4599aa36b8ff64b243` from committed source `4b3cd57`. Every native process and both rank replays exit zero; cross-rank target hashes agree. All 24 layers are present on both ranks, with expert shards `[32,512,1024]` and 1,335,527,424 parameter bytes per rank. Graph matches eager within each prefill policy, including cancelled tokens; natural-language examples also agree between ranks. Four synthetic requests differ across full/chunk prefill, and those differences remain in the evidence.

| Granite TP2 mode | Cases per rank | Steps per rank | Control reads per rank | Actual Graph replays per rank | Peak Torch reserved GiB per rank | Whole native process seconds, including startup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager | 6 | 53 | 212 | 0 | 1.3555 | 158.775 |
| Graph | 6 | 53 | 212 | 44 | 1.3730 | 33.638 |
| Chunk | 6 | 81 | 241 | 0 | 1.3340 | 36.400 |
| Graph+chunk | 6 | 81 | 241 | 43 | 1.3691 | 34.439 |

These are functional observations on shared GPUs, with cold JIT in the first process and overlapping finite FA/microbenchmark jobs. The startup numbers do not qualify a speedup. The old collector emits nonfatal TCPStore warnings when rank zero exits before other distributed handles are destroyed; statuses remain zero. Subsequent collection explicitly synchronizes rank teardown and destroys the original distributed groups, without terminating a process or changing sampling.

Granite TP2+EP uses the same frozen collector and full checkpoint. Every native process and both rank replays exit zero, with `[16,1024,1024]` expert shards and 1,335,527,424 parameter bytes per rank. All four modes preserve matched-policy Graph tokens, exact cancellation, scheduling, cross-rank replay targets and identical natural-language outputs between modes/ranks. Three synthetic requests differ between full and chunk prefill. The strict independent-reference gate exits **one**, because the prime answer chooses a colon (token 44) instead of the CPU reference's period (token 32) at output index 28; the native answer continues with `: 1,` until the 32-token limit. The Paris answer matches exactly. This is not an independent numerical-reference pass.

| Granite TP2+EP mode | Cases per rank | Steps per rank | Control reads per rank | Actual Graph replays per rank | Peak Torch reserved GiB per rank | Whole native process seconds, including startup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager | 6 | 53 | 212 | 0 | 1.3555 | 35.968 |
| Graph | 6 | 53 | 212 | 44 | 1.3730 | 34.190 |
| Chunk | 6 | 81 | 241 | 0 | 1.3340 | 37.904 |
| Graph+chunk | 6 | 81 | 241 | 43 | 1.3691 | 35.190 |

Independent CPU teacher forcing at the identical first-divergence prefix gives period 32 as top-1, colon 44 as top-2, with a 1.25-logit gap. This measures the CPU reference alone; native logit error and the cause of the difference remain unmeasured. The verifier now saves this exact mismatch before retaining its failing assertion, so diagnostics do not erase the failed gate. No common token trajectory is fabricated, and no metric in this table qualifies speed or prediction accuracy.

Logs, JSON, configuration pickles, native profiles and weights remain outside Git. No process is terminated; queued performance windows, live checkpoint writers and the existing GPU0 lease are respected. Completed task-owned model files are cleaned only after their readers and writers exit naturally.

## Reproduce the finite matrices

`model_functional.py matrix` runs exactly four independent native variants, then the original CPU Scheduler for every rank, followed by matched-policy Graph and cross-rank replay checks. A failed native/replay process stops the remaining queue after natural exit. It does not impose process timeouts or send termination signals. Each native process keeps the existing model/runner/Scheduler source audit; the orchestration and verifier are separate from the retained measured collector.

Use the pinned private serving Python and CUDA 13.0 toolchain/cache environment from the protocol above. These commands reproduce complete-checkpoint matrices in separate evidence directories, each within a confirmed GPU window. Granite has completed the commands below; Kimi's additional matrices are still pending:

```sh
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" tests/integration/tooling/model_functional.py matrix \
  --model "$GRANITE_CHECKPOINT" --evidence "$RAW/granite-tp2" \
  --tp 2 --attention-backend TRITON_ATTN
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" tests/integration/tooling/model_functional.py matrix \
  --model "$GRANITE_CHECKPOINT" --evidence "$RAW/granite-ep2" \
  --tp 2 --ep --attention-backend TRITON_ATTN
CUDA_VISIBLE_DEVICES=0 "$PYTHON" tests/integration/tooling/model_functional.py matrix \
  --model "$KIMI_CHECKPOINT" --evidence "$RAW/kimi-functional" \
  --kv-bytes 536870912 --memory-fraction .97 --trust-model-code
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" tests/integration/tooling/model_functional.py matrix \
  --model "$KIMI_CHECKPOINT" --evidence "$RAW/kimi-tp2" \
  --tp 2 --kv-bytes 536870912 --memory-fraction .55 --trust-model-code
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" tests/integration/tooling/model_functional.py matrix \
  --model "$KIMI_CHECKPOINT" --evidence "$RAW/kimi-ep2" \
  --tp 2 --ep --kv-bytes 536870912 --memory-fraction .55 --trust-model-code
```

Granite's already verified complete CPU reference can be copied into each new evidence root before verification. The `independent_reference_verified` field records whether that input was actually present; missing reference evidence never becomes an implicit pass. Kimi's full published checkpoint and SHA verification remain mandatory. Its earlier native configuration had a 4096-token scheduling budget; the new finite functional workload uses 512 (64 for chunks), with the same complete 27-layer architecture and 512 MiB KV budget.

The old endpoint `region-42.seetacloud.com:38416` became unavailable. The user supplied port `39817`; its ED25519 host key matches the previously recorded key, and the existing SSH identity connects. The replacement container has two idle H20 devices, UUIDs `GPU-12812ea6-65db-7709-44f5-d1a18756e037` and `GPU-a603caaa-2262-235a-315b-3152c90bfe66`, each reporting 97,871 MiB. The former 85 GiB GPU0 lease is absent. Consequently, trained Kimi TP2/EP must now be tested rather than retaining the old capacity exclusion.

Migration retained direct-download Kimi byte-range segments but cleared the private serving environment, CUDA toolchain, complete checkpoints and remote raw files. Previously collected evidence remains backed up locally. The old download writers are absent. The immutable HF metadata and matching mirror weight hashes have been revalidated; incomplete ranges resume by appending only the requested remaining bytes, followed by complete-shard SHA-256 verification. Checkpoint metadata and the pinned runtime are restored in task-private XFS storage. Recovery/download work does not establish a new native matrix result or reuse an old GPU handoff.

External restoration subsequently repopulated old paths and replaced an active writer's log inode. Collection uses a separate committed source directory, runtime snapshot and evidence namespace. The runtime passes dependency checks and an integrity audit of 53,146 files: one shared build script matches its FlashInfer owner; two video-library files exactly match the SHA-256-verified published wheel despite stale hashes inside that wheel's `RECORD`. These packaging differences are retained in private evidence. The restored model, Scheduler and runner source hashes match the prior measured vLLM wheel. A fresh complete Granite BF16 CPU reference on the replacement host reproduces both original answers without initializing CUDA.

The resumed Kimi writer naturally exits with status zero after verifying all 20 original shards, totaling 98,248,224,120 bytes. Its complete verification manifest SHA-256 is `1ae12bc7a2d00103ed93ec989221bbeb108f2bf147efcb413d1c7a1fb27638a5`; all temporary range segments are removed. The checkpoint's ten configuration/tokenizer/model-code files also match the immutable revision. Replacement-host Driver 580.65.06 differs from the earlier 580.105.08; prior timing qualification remains attached to its original hardware/conditions. These recovery checks establish readiness, not new native E2E passes.

The local continuation audit strengthens matched-policy verification: all six named cases must exist, every scheduled item must match, and cancelled-request token sequences must also match exactly. The complete backed-up TP1 dataset passes. A copied evidence fixture with the cancelled request output removed is correctly rejected; an empty prefix cannot bypass Graph parity. This verifier check supplies no new native GPU result.
