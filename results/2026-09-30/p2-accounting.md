# P2 accounting and fixed-trace replay

Baseline: `d05e060`. Tested implementation: `ebac6c9`. Release builds used GCC 11.4 and CUDA 13 headers. All checks ran in the existing `0z5a` environment inside offline, no-GPU `runc` containers with Python 3.12 and PyTorch `2.13.0+cu130`. No packages or running services were changed; no model files were downloaded.

Each eager launch and graph replay receives fresh kernel invocation IDs. A prediction must cover exactly its corresponding invocation. The shared validator rejects duplicate, missing, reordered or foreign coverage before scheduling or charging service time. Only kernel device-service costs are accepted.

`replay_kernel_trace` uses a fixed, topologically ordered kernel trace, explicit arrivals and host costs, one host submission lane and one serialized compute queue per device. Target timestamps come only from these inputs and the predicted service durations. Query wall time is part of replay-call wall time, not an additional target cost.

## Query-delay invariance

Four synthetic oracle samples span two devices. Service durations are 3, 6, 2 and 1 ms; host costs are 2, 1, 3 and 2 ms. The third invocation waits for the second on the other device, and the fourth arrives at 12 ms. Expected device intervals are `[2,5]`, `[3,9]`, `[9,11]` and `[14,15]` ms.

| Injected query delay | Target makespan | Kernel service sum | Host service sum | Query wall | Replay-call wall | Target time / replay wall |
|---:|---:|---:|---:|---:|---:|---:|
| 100 µs | 15.000 ms | 12.000 ms | 8.000 ms | 0.175 ms | 0.200 ms | 75.00× |
| 10,000 µs | 15.000 ms | 12.000 ms | 8.000 ms | 10.152 ms | 10.168 ms | 1.48× |

Increasing the injected delay 100× leaves every host/submission/ready/start/completion timestamp identical. The last column describes this tiny synthetic replay's execution rate, not GPU or model acceleration. Replay-call wall includes validation, prediction and scheduling; it excludes caller file I/O and output. These are single diagnostic observations, not a throughput benchmark or real cache cold/hot measurements.

## Contract results

| Check | Result |
|---|---|
| Registered ABI, graph, profile, provider, file and offline replay contracts | 25/25 passed |
| Fresh coverage IDs in eager execution and successive graph replays | Passed |
| Invalid coverage in eager/graph submission | Rejected; service ledger and compute queue unchanged |
| Duplicate/missing/foreign/reordered coverage and whole-forward result | Five offline cases rejected; no partial timeline or sample consumption |
| Duplicate/zero IDs, forward/self dependencies, negative host/arrival time, inconsistent context/device/profile | Eight traces rejected before querying the predictor |
| Target timeline overflow | Rejected without consuming samples or returning partial intervals |
| Empty trace | Zero target/host time; no predictor call or commit |
| Timing file → explicit binding → provider → offline replay | 31 ms service + 3 ms host, 33 ms makespan; provenance retained |
| Saved four-card 5060 Ti and A100 profiles/peer matrices | Both passed; 588 attributes per system |
| PyTorch copy/add on every device, four graph replays per process | 16/16 processes passed across both saved systems |

## Default-path process-time comparison

Both revisions used the same saved A100 system manifest and device-3 graph fixture with the default 10 ms provider. ABBA then BAAB ordering supplied four samples per revision. Medians include container startup, Python/PyTorch import, capture and four synchronized replays. Builds and other test batches had finished before this comparison.

| Fixture | Before | After | Before / after | Speed change |
|---|---:|---:|---:|---:|
| Copy + cross-stream fork/join | 2.5361 s | 2.4622 s | 1.0300× | +3.00% |
| Add + cross-stream fork/join | 2.5746 s | 2.6985 s | 0.9541× | −4.59% |

The directions differ and include startup noise; these observations do not establish a general speed improvement. Speed change is `(before / after − 1) × 100%`.

## Reproduction and scope

Run `build/kernel_replay_contract tests/fixtures/kernel.timing` and `build/perf_model_contract` in the no-GPU container. The former covers explicit trace inputs, injected predictor delays and the file-backed path; the latter exercises the normal paced Driver paths. Run all registered contracts and the saved-system profile/graph checks described in the [A100 report](../2026-09-29/a100-cross-validation.md).

This is a kernel-only fixed-trace diagnostic. The caller supplies dependencies and normalized stream labels; it does not extract CUDA synchronization, model transfers or host blocking, or coordinate live framework actors. Compute queues retain the submission-order heuristic. The paced Driver still has unknown target-host time. Automatic framework binding, calibrated performance prediction, ready-driven scheduling and the P4 coordinator remain separate work. This iteration adds no native-GPU or model E2E result. Raw logs remain outside the repository.
