# Resource execution, original scheduling and native routing

This iteration validates the controlled **U1** serving boundary in vLLM **0.30.0**: the original CPU Scheduler/KV manager consumes an explicit fixed-length token oracle, while worker service and causal completion delivery are adapted. Native calibration runs numerical inference on Qwen2.5-0.5B; the simulator does not compute tensors. Single-rank timing accuracy and synthetic dual-rank protocol correctness are separate results.

The new ResourceEngine owns ready admission, resident capacities, weighted shared throughput, phase transitions and mutable completion tokens. Completed work remains queryable but no longer participates in active scans. `replay_resource_trace` shares predictor coverage and atomic commit with the existing fixed-trace oracle. A blocked earlier submission cannot reserve a future slot ahead of independent ready work. The live paced CUDA Driver keeps its existing queue policy; this is an explicit execution engine used by the replay/serving harness.

| Structural contract | Result | Verified scope |
|---|---|---|
| H01/H02/H03 | Pass | Driver counts 1/2/4/8/16/24/32/257, invalid-input rejection, unknown attributes fail; topology layouts 1/2/4 and 8N with explicit local/global reorder/subset mapping |
| K01/K02 | Pass | Existing eager/capture/replay metadata and owned packed-parameter contracts |
| K03/K04 | Pass | Allocated-resource threshold/waves, zero resource demands, invalid/cluster rejection; prior two native occupancy comparisons on RTX 5090 matched 2 CTA/SM |
| Q01 | Pass | Independent ready operation finishes at 20 ms rather than 130 ms; dependency-bound replay makespan 130 → 110 ms with identical service sum |
| Q02/Q03 | Pass | Capacity-bound weighted rates, phase release, restored rates, generation invalidation after contention; engine completion tokens only |
| G01/G02/G03 | Existing supported scope passes | Same-exec ordering, independent-exec boundary and metadata lifetime; no new claim that shared-rate tokens replace live Driver graph events |
| T01/T02/T03/T04 | Pass | Arrival during service, query-delay invariance, created/delivered/observed timestamps, blocked certificates/in-flight messages and explicit protocol failure |
| N01/N02/N03 | Pass in declared boundary | Collective rank/generation/sequence/replay identity; physical shared-link flow limit; whole-step scope prevents extra communication billing. No NCCL ABI interception |
| M01/M02/M03 | Pass | Unknown control-read producer chain fails; unique block/refcount/transfer readiness; explicit Graph/KV budget changes original scheduler admission |
| R01/R05 | Pass | Native llm-d disabled-endpoint filter/queue scorer/picker, delayed queue metrics, original scheduler/KV feedback |
| R02 | Ledger passes | Generation/sequence/visibility rejects stale cache snapshots; native queue-only routing does not consume prefix cache events |
| R03/R04/R05 | Pass | Native Dynamo partial P/D compensation, cancel/late/duplicate/virtual-lease cleanup and original scheduler completion feedback; final native loads zero |
| A01/A02 | Pass | Native AIS regression units, identity/scope rejection and support boundary; no extra host/NCCL charge |

The new final run passed **31/31 registered contracts**, plus original-vLLM, KV-budget, native EPP, native Dynamo and AIS integration contracts. It ran in the provided GPU-enabled Vast container; CPU integration asserts that Torch never initializes CUDA. That is distinct from the earlier 27/27 contracts and PyTorch probes in no-GPU `runc --network=none` containers. The new machine has no nested Docker runtime, and no new no-GPU-container result is claimed.

## Native routing feedback

All endpoint-loop cases use 80 arrivals spaced 1000 ns apart, P64/O8, two original vLLM schedulers with max-seq 1, a 5000 ns synthetic whole-step cost and 500 ns metric delay. Each run completes 640 oracle tokens, observes exact declared arrivals and restores original physical KV block counts. The table records one native choice realization. Native picker ties are retained and can change first-arrival timing by arrival intervals.

| Native policy | Endpoint service scales | Virtual finish ns | Requests to endpoint 0 | Requests to endpoint 1 |
|---|---|---:|---:|---:|
| llm-d queue profile | 1 / 1 | 1601000 | 40 | 40 |
| llm-d queue profile | 0.2 / 1 | 1442000 | 44 | 36 |
| Dynamo selection/load tracker | 1 / 1 | 1601000 | 40 | 40 |
| Dynamo selection/load tracker | 0.2 / 1 | 1440000 | 44 | 36 |

These are causal-sensitivity fixtures, not measured GPU speedups. Arrival selects the endpoint using native policy. The selected original Scheduler allocates KV and emits the next batch; service completion updates tokens/cache; metrics and native prefill/completion acknowledgments become visible only at their registered delivery. Subsequent native selections see that feedback. Snapshot/shadow Dynamo calls use native selection without reservations.

The Dynamo P/D fixture first reserves P with no eligible D worker, receives the native failure, and compensates P. A subsequent plan independently holds P and D. Prefill completion releases P while retaining D; completion, cancel or virtual lease expiry closes the rest once. Native `free_reservation` itself rejects a second release: adapter-owned identities deduplicate callbacks before invoking it.

## KV budget feedback

The original Qwen layout has a 16-token block and 8192 page bytes per layer; block budget uses the actual cache groups/layer membership. An explicit shared Graph/KV pool is reduced before constructing the original KVCacheManager. This models caller-declared admission and does not assert vLLM's automatic GPU profiling semantics or measure a CUDA Graph private pool.

| Graph reservation in block-equivalent bytes | Original KV blocks | First batch requests | Virtual finish ns |
|---:|---:|---:|---:|
| 0 | 7 | 2 | 4000000 |
| 3 | 4 | 1 | 8000000 |

Separate cache contracts verify one physical allocation with multiple request references, live-reference eviction rejection, transfer readiness and graph-pool lifetime through pending replays. They also reject future visibility, stale sequences and older endpoint generations.

## Stage boundaries

| Plan stage | Delivered in this iteration | Further scope |
|---|---|---|
| P0/P1/P2 | Existing graph/profile/launch/provider/probe contracts retained; ready replay added | Automatic live code/argument binding and unseen kernel prediction |
| P3 | Original vLLM 0.30 single-rank CPU scheduling, every oracle control read, native independent calibration/validation | Unmodified U0 framework execution, arbitrary sampling semantics |
| P4 | Registered actor safe horizon, paced/coordinated pair, dual-rank delay/arrival parity, collective matcher | Cross-process runtime/OS timer interception and native TP accuracy campaign |
| P5 | Ready ResourceEngine, ordinary allocated-resource occupancy features, shared rates, mutable completion tokens | Per-SM CTA placement and integration with live Driver graph/event completion |
| P6 | Global node/device identity, explicit process mapping, 1/2/4/8N topology fixtures, ring and shared-path flow baseline | Hardware catalog/placement generator with measured NIC/NUMA/topology discovery; arbitrary transports |
| P7 | Unique physical/cache ledgers, transfer gate, delayed visibility and original KV budget feedback | Live prefix events and measured native Graph-pool allocation tracking |
| P8 | Separate native EPP/Dynamo policies, booking-free snapshots/shadow, closed loops, reservation compensation/cancel | Full deployment policies, prefix-aware EPP and virtualized native DEP timers |
| P9 | Pinned AIS native regression, complete homogeneous FPM mapping, frozen independent sweep and measured-candidate regret | Qualified op-level database for vLLM 0.30/A10040, cross-device/shape performance generalization |

The supported U1 path meets the initial median-error and measured-candidate regret targets; this does not mark every broader P0–P9 capability complete. [Serving accuracy and measured coverage](serving-validation.md) and [fresh-process speed comparison](serving-speed.md) keep target time, predictor wall time and complete simulator execution cost separate.

Versions: vLLM 0.30.0, Torch 2.13.0+cu130, CUDA toolkit 13.0, native driver 595.91.07, AISimulate 0.12.0, Dynamo runtime 1.5.0, llm-d Router v0.10.0. The llm-d source archive SHA-256 is `9524db96c27fb21544c0ab0cf68b0b5f4092a0a43f2ef0121583daa639181946`. The Qwen model revision is `060db6499f32faf8b98477b0a26969ef7d8b9987`. Input hashes are in the accuracy report; raw artifacts remain outside git.

Primary API references: [Dynamo standalone native selection](https://docs.nvidia.com/dynamo/v1.5.0/knowledge-base/modular-components/router/standalone-selection), [AISimulate 0.12 core API](https://github.com/ai-dynamo/aisimulate/blob/v0.12.0/docs/core-api.md), [llm-d Router source](https://github.com/llm-d/llm-d-router/tree/v0.10.0).

Tested source snapshot SHA-256: `8b489436a12d25457f9e47f41e79fb7da9c5bbafb1c86355e1146058cba284a5`. The final topology-helper build passed all contracts and preserves resource semantics; the speed report identifies the earlier compared binaries explicitly.
