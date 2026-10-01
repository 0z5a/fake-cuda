# Complete-checkpoint functional serving matrix

Status: full Granite TP1 functional matrix verified; remaining current-memory matrices are in progress. This report extends the [trained TP1 prediction results](topk-model-e2e.md) and [serving/EP contracts](serving-ep.md). No pending native row is a passing E2E result. Native numerical inference is supplied by the original vLLM backend; CPU replay supplies explicit recorded control metadata, not logits or predictions of model quality.

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
| Decode cancellation | Two prompt-48 requests; cancel first after step 2 | Remaining request finishes; cancelled tokens are a parity prefix |
| Prefill cancellation | Two prompt-192 requests; cancel first after step 1 | Chunk mode cancels before first output; exact original Scheduler replay |

Cancellation occurs after a synchronized native step. Concurrent in-flight cancellation and last-use resource release are separately covered by the serving contracts. Releases use completed-step ordinals; these cases do not measure production arrival timing or SLA.

Eager, full-decode Graph, chunk prefill and Graph+chunk each use their own naturally exiting process. Explicit KV capacity is 64 MiB for Granite, divided by TP ranks; chunk budget is 64 tokens, maximum running requests eight. Sampling is greedy, with EOS ignored for the bounded synthetic-token cases. Two natural-language prompts additionally compare the independent CPU reference with original native generation. Prefix caching, speculative decoding and asynchronous scheduling remain disabled.

## Current-memory eligibility and verified results

| Complete checkpoint / topology | Fits current residency | Native variants | Original Scheduler CPU replay | Independent numerical reference |
| --- | --- | --- | --- | --- |
| Original Kimi TP1, 512 MiB KV | Yes, GPU1 idle capacity | Previous homogeneous E2E PASS; new four-variant matrix pending download/window | Previous 27 workloads PASS; new matrix pending | Previous native answers No / Paris |
| Original Kimi TP2 / EP | No while GPU0 retains its 85 GiB lease | Not run | Not claimed | Not claimed |
| Original Granite TP1 | Yes | All four variants × six cases PASS | 268 scheduled steps, 906 recorded-control reads PASS | Two natural-language prompts × four variants exactly match complete BF16 Transformers CPU reference; no CUDA initialized |
| Original Granite TP2 | Yes with current GPU0 free capacity | Pending coordinated finite queue | Pending native controls | Pending native comparison |
| Original Granite TP2+EP | Yes with current GPU0 free capacity | Pending coordinated finite queue | Pending native controls | Pending native comparison |

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

The actual native collector SHA-256 is `9f4231add7efe35332d6010b5920ec08e0a0e1eacedcad6e7ae2b2bee643c582`. Its exact source snapshot is retained outside Git. Subsequent edits add the cross-policy comparison and independent CPU teacher-forcing diagnostic; the native collection function is unchanged. No speed improvement or new prediction-accuracy claim follows from recorded native sampler replay. Existing independent whole-model timing accuracy and equal-target CPU simulator speed tables retain their original boundaries.

Logs, JSON, configuration pickles, native profiles and weights remain outside Git. No process is terminated; queued performance windows, live checkpoint writers and the existing GPU0 lease are respected. Completed task-owned model files are cleaned only after their readers and writers exit naturally.

## Reproduce the remaining finite matrices

`model_functional.py matrix` runs exactly four independent native variants, then the original CPU Scheduler for every rank, followed by matched-policy Graph and cross-rank replay checks. A failed native/replay process stops the remaining queue after natural exit. It does not impose process timeouts or send termination signals. Each native process keeps the existing model/runner/Scheduler source audit; the orchestration and verifier are separate from the retained measured collector.

Use the pinned private serving Python and CUDA 13.0 toolchain/cache environment from the protocol above. The following commands specify complete checkpoint directories and evidence destinations; they require a newly confirmed idle GPU window. They are **remaining commands**, not completed runs:

```sh
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" tests/integration/tooling/model_functional.py matrix \
  --model "$GRANITE_CHECKPOINT" --evidence "$RAW/granite-tp2" \
  --tp 2 --attention-backend TRITON_ATTN
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" tests/integration/tooling/model_functional.py matrix \
  --model "$GRANITE_CHECKPOINT" --evidence "$RAW/granite-ep2" \
  --tp 2 --ep --attention-backend TRITON_ATTN
CUDA_VISIBLE_DEVICES=1 "$PYTHON" tests/integration/tooling/model_functional.py matrix \
  --model "$KIMI_CHECKPOINT" --evidence "$RAW/kimi-functional" \
  --kv-bytes 536870912 --memory-fraction .97 --trust-model-code
```

Granite's already verified complete CPU reference can be copied into each new evidence root before verification. The `independent_reference_verified` field records whether that input was actually present; missing reference evidence never becomes an implicit pass. Kimi's full published checkpoint and SHA verification remain mandatory. Its earlier native configuration had a 4096-token scheduling budget; the new finite functional workload uses 512 (64 for chunks), with the same complete 27-layer architecture and 512 MiB KV budget.

On continuation, `region-42.seetacloud.com:38416` returns **Connection refused**. Old SSH handles exiting 255 do not establish the remote download processes' state. Weight completeness, current per-card capacity, foreign GPU processes and any previous queue handoff must be revalidated after reconnecting; no replacement download or GPU worker was started in response to the lost observation channel.
