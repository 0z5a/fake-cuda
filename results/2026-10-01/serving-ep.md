# Request admission, chunk prefill and expert dispatch

Incremental baseline: `d93a3dadd7b979f12ac399d56792b701bbe51d1a` on `0z5a:moe-cost-ledger`, stacked on `device-launch-metadata`. This iteration extends P5/M2 request replay and P6/M3 explicit resource scheduling. Native numerical results, calibrated estimates and analytical topology scenarios retain separate scopes.

## Request and memory contracts

`sim.serving` implements FCFS admission, decode priority, continuous admission and a finite prefill token budget. First output becomes visible after the last prefill/sampling step. Each decode step contributes one token per request; single-output ITL is undefined. Cancellation suppresses later token visibility, but already submitted work retains pages and state slots until completion. Computed/discarded tokens and processed prompt tokens remain accounted for.

Every physical card has an explicit usable capacity, all resident weights, fixed state/conv pools, page geometry and workspace/graph/communication/allocator budgets. Fixed KV pools are charged once; otherwise final-context pages are reserved before admission. Requests that can never fit are rejected; FCFS preserves queued head order. This policy is conservative and has no prefix sharing, offload or preemption.

Seven CPU serving contracts cover capacity, chunking, cancellation, actual slot reuse, fixed-pool accounting, token/request conservation, undefined percentiles and 100 deterministic replays. Request P95 needs 20 samples and P99 needs 100; insufficient populations remain null. Observation starts at the first arrival and includes drain, with zero warmup.

The original vLLM 0.30.0 Scheduler remains a separate U1 adapter. Its single/dual-rank, delayed IPC, paced/coordinated, fresh-epoch and natural-exit contracts pass using an explicit eight-sequence fixture. The retained full-model configuration actually supports eight sequences; the previous 32-sequence test expectation was invalid for that configuration. No production memory budget was enlarged to satisfy the fixture.

## Frozen whole-model replay and capacity counterfactual

`adapters.vllm.explicit.UniformStepCost` connects the frozen whole-step table to M2. Its scope includes host, communication and sampling. It accepts homogeneous decode or fresh unchunked prefill only; continued full-model chunks and heterogeneous contexts fail. An unknown prefill cost cannot be replaced with repeated decode costs.

The 27 independently collected trained Kimi-Linear-48B-A3B-Instruct BF16 TP1 workloads retain exact native batch membership. Median absolute whole-workload duration error is **1.615%**, using the original frozen calibration. This reuses the [trained checkpoint E2E evidence](topk-model-e2e.md), without downloading weights again.

The following **predicted capacity scenario** submits 32 requests at time zero, prompt 96/output 32. Per-card input is 95 GiB usable memory, a conservative all-weight budget equal to the complete checkpoint's 98,248,224,120 bytes, seven BF16 MLA latent page layouts (16 tokens, 576 values/token), 20 FP32 KDA state layers, declared conv pools, 1 GiB workspace and 512 MiB allocator overhead. Those page/buffer inputs are declared scenario geometry, not a measured native cache allocator layout. Costs stay within the frozen whole-model domain.

| State slots/card | Completed / visible tokens | Predicted drain s | Predicted tokens/s | TTFT P50 / P95 s | ITL P50 / P95 ms | Peak GiB/card | Predicted throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 32 / 1024 | 9.791811 | 104.577 | 4.338706 / 8.622624 | 37.671966 / 37.843966 | 93.166 | 1.000x |
| 8 | 32 / 1024 | 4.970648 | 206.009 | 1.929303 / 3.793296 | 37.977988 / 38.009738 | 93.332 | 1.970x |

The ratio is a model counterfactual, not a new measured hardware improvement. Request TTFT P99 is undefined with 32 requests. Both scenarios complete all requests and conserve 1024 visible tokens.

## Native chunk prefill

The new `KDAPrefill` descriptor uses explicit chunk lengths, chunk size and convolution width, with independent chunk stage families. Unknown executed recurrence FLOPs remain null. Grouped-value heads remain unsupported. Native collection uses FLA 0.5.2 `chunk_kda_fwd`, chunk size 64, BF16 QKV/O projections (D=2304), H=32, dk=dv=128, FP32 fixed-slot state and four-tap depthwise convolution. Fixed alpha/beta and output gate retain the existing surrogate scope; this is not the entire production Kimi attention module.

GPU1 is H20 UUID `8ee84e7d-143f-dd29-1097-85943783e027`, Driver 580.105.08, Torch 2.7.0+cu128 and Triton 3.3.0. Native outputs/state pass the packed, one-token recurrent reference. Calibration batches 1/4/8/16 freeze before independent batches 2/3/6. Each uses five fresh processes, 30 warmups and 200 CUDA-event measurements per stage, graph mode and fixed state reuse. Numerical/JIT setup is outside event windows. No foreign GPU1 process is accepted at campaign endpoints.

| Prefill validation | Result |
| --- | ---: |
| Held-out configurations / process observations | 3 / 15 |
| Strict unknown stage count | 0 |
| Long-stage relative error P50 / P95 | 4.14% / 7.62% |
| Whole-operator relative error P50 | 0.65% |
| Deterministic replay hashes | 100 / 100 identical |
| CPU replay median, one operator/imports excluded | 73.11 us |

Q/K and recurrent-reference slices must be contiguous: the low-level kernels use a packed layout rather than the dense-but-strided convolution output. The reference evaluates one token at a time, preserving explicit state-slot indexing. These fixes precede formal calibration. Ragged chunks, rotating slots, cold state and full-model continued chunks need separate native timing paths.

## EP ledgers and shared resource clock

`sim.ep` records source-token assignments A and target-rank unique tokens U independently. Dispatch deduplication, metadata, dtype, placement and combine reduction are explicit protocol inputs. Local transfers stay in the ledger but never enter remote links. The golden case sends one token to two experts on rank 1: A[0,1]=2 and U[0,1]=1. FP32 assignment combine and target-local reduction produce different return payloads.

Each topology supplies global device identity, DRAM/SM/buffer capacity, physical links, paths, startup and duplex policy. Shared physical link IDs prevent opposite flows from each receiving full shared bandwidth. The link byte/bandwidth bound is reported separately from scheduled completion.

The existing C++ `ResourceEngine` remains the execution clock. It recomputes remaining service and completion generations when competitors change. New held reservations survive dispatch, blocked compute phases and combine; buffers become available only after final use. Chunk pipeline gain comes from executing the dependency/resource schedule again.

| Synthetic two-chunk fixture | Scheduled ns | Ratio vs serial | Input distinction |
| --- | ---: | ---: | --- |
| Serial chunks | 480 | 1.000x | Two 100-byte transfers/chunk; 40 ns compute; one shared 1 byte/ns link |
| Pipeline, shared link | 440 | 1.091x | Both chunks admitted; link capacity still shared |
| Pipeline, separate directed links | 240 | 2.000x | Each direction has its own declared 1 byte/ns capacity |
| Pipeline, one chunk buffer/card | 480 | 1.000x | 16-byte capacity prevents early second-chunk admission |

These are analytical fixtures, not H20 or DeepEP performance predictions. Seven EP contracts also check slow-rank barriers, startup latency, saturated bandwidth, invalid paths/placement, repeatability and discrete combined costs charged once.

## Native P2P and compute pairing

Two peer-accessible H20s use native mapped-peer `cuMemcpyDtoDAsync_v2`: eight 128 MiB copies (1 GiB/group) from GPU0 to GPU1. Compute is either BF16 2048-square GEMM or the FP32-state KDA recurrent core (B=64, H=32, dk=dv=128). Copy and compute use independent streams/graphs with a common event release and completion barrier. A resident GPU0 lease at PID 171961 occupies 87,152 MiB; its retained identity is part of calibration. Any additional observed GPU process or changed background rejects a case. This is P2P memcpy, not NCCL/DeepEP communication.

Calibration repetitions 1/4/8/16/32 freeze before held-out 2/12/24. Five fresh process sessions use 30 warmups and 200 event measurements per phase. All 6 held-out configurations pass: 90 standalone/paired phase observations, **P50/P95 error 0.08%/6.17%**. No NCU DRAM/SM demand is inferred from logical bytes or observed latency.

The table uses medians across five sessions. The last column divides the **sum of separately measured standalone windows** by the paired window. There was no serialized-arm measurement, so it is not a complete-process speedup.

| Compute family | Repetitions | Standalone copy us | Standalone compute us | Paired us | Sum of standalone / paired |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMM | 2 | 2746.000 | 287.392 | 2746.640 | 1.104x |
| GEMM | 12 | 2745.488 | 1692.928 | 2746.416 | 1.616x |
| GEMM | 24 | 2747.504 | 3381.728 | 3459.792 | 1.772x |
| KDA | 2 | 2746.560 | 276.000 | 2747.792 | 1.100x |
| KDA | 12 | 2747.024 | 1616.480 | 2747.568 | 1.588x |
| KDA | 24 | 2746.656 | 3245.520 | 3521.056 | 1.702x |

`sim.resources.profile_phase` admits only a qualified, identity-matched, in-domain discrete combination and reserves an explicit combination lane. Its measured duration is charged once with zero additional rate demand; standalone costs are not added or divided by bandwidth again. This qualifies the stated paired configurations, not arbitrary concurrent resource allocations.

All 30 held-out paired observations run through this C++ clock consumer with exact predicted start/end sums: 91,615,710 ns for the serialized combination trace. Pair-only P50/P95 prediction error is **0.05%/9.03%**. One hundred trace replays have the same target SHA-256 `f79972af38aef9faa556bc0bc1cad34698f35ba2f9d2ec1be8ff655d688903bb`. This validates timing delivery and single charging, not a native serialized-arm measurement.

## Verification, provenance and remaining domains

C++23 build and **36/36 registered contracts** pass through the direct no-kill runner. The semantic/serving/EP Python suites cover 14/7/7 cases. The original-vLLM integration also passes with CUDA hidden. The coordinated small numerical window uses Torch 2.13.0+cu130 and NCCL 2.29.7, independently of the event-calibration environment; it makes no new timing claims.

| Native correctness contract | Result | Boundary |
| --- | --- | --- |
| Short-row selection | 36 configurations PASS | FP32/BF16; widths 64/128/256, k=1/2/4/8; 257-wide ties and k=E; NaN/Inf conventions |
| Captured mapped-peer copy | 6 cases PASS | Both directions, three changed source values; destination zeroed before replay |
| Two-rank MoE dispatch/expert/combine | PASS, maximum absolute error 6.263e-7 | BF16 assignment dispatch, rank-owned experts, FP32 assignment combine, CPU reference |

Native EP assignments are `[[2,2],[3,1]]`, unique-token counts `[[1,1],[2,1]]`, dispatch bytes `[[16,16],[24,8]]`, and independently directed combine bytes `[[32,48],[32,16]]`. Local entries bypass NCCL. Frozen host routing metadata is explicitly outside the payload; this small assignment protocol does not claim native deduplicated DeepEP or full-model EP validation. Native reference tolerance is atol=2e-5/rtol=3e-2.

The first NCCL attempt failed because ctypes prototypes were set on bracket-loaded function objects while calls used different attribute-cached objects. Reusing the configured callable preserves 64-bit pointer arguments and passes the numerical contract. The failed process exited naturally; no process was killed or device reset.

| Frozen artifact | SHA-256 |
| --- | --- |
| Whole-model calibration | `617cb6d6320549f980f26d7f2baf71cdd7fc2241adb53ed1fbdf6e9434e321ab` |
| Whole-model qualified cost table | `c10b436a1548fec558433933cef024724065869f42198c32a3ed4d81aaff7295` |
| Chunk-prefill profiles | `44bb9735246168a3ad9be6cded5ed5a574a7a170e5816d260baa36365bcdb762` |
| Standalone/paired profiles | `37f0c5a62867c091e42b52d24fc05941af7749e0f181a89bcb186bbadb2e3ef9` |
| Qualified paired profiles | `5440458d84092d4bcfc890e6b26eeb3ebe3aeaac71f3f46e2148a2b636e309d4` |
| Native NCCL DSO | `aa957cdfb91b516eae0d54a28e9ee5db52730d02e0ab45580efc3c19a68327a4` |

Raw sessions, frozen/qualified profiles, actual collector source snapshots and diagnostics remain outside git with a Mac backup. A wording-only overlap report correction follows collection; the retained collector snapshot identifies the measured revision. No logs, JSON, binaries or model weights are submitted. Trained checkpoint shards and duplicate downloads remain cleaned after their readers/writers exited naturally. No process was killed. Shared environments and lcpu NFS remain untouched.

| Plan area | Current boundary |
| --- | --- |
| P0–P4 / M1 | Existing qualified MoE/top-k/KDA decode plus new 64-token surrogate chunk prefill |
| P5 / M2 | Request/memory/cancellation contracts; homogeneous trained whole-step subset independently replayed |
| P6.1/2/4/5 | Explicit EP ledger, declared topology, capacity sharing, remaining-work updates and chunk buffer lifetime |
| P6.3 | Independently qualified native P2P+GEMM/KDA discrete combinations on these two H20s |
| Further native validation | Full-model heterogeneous/continued chunks, trained TP2/EP, real DeepEP/topology demands and broader cold/rotating/backend timing |
| Later extensions | Historical-token selector timing and complete production attention-module attribution |

The partial module trace cannot supply a full-model attention fraction. The complete trained TP1 E2E is established; trained TP2/EP remains unqualified while the large GPU0 lease prevents the required residency. These domains stay unsupported rather than inheriting surrogate costs.
