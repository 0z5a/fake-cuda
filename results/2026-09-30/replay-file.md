# Ordered timing-file validation

Baseline: `8a7783e`. Tested implementation: `d82eeb0`. Release builds used GCC 11.4 and CUDA 13 headers. Tests ran in the existing `0z5a` environment inside offline no-GPU `runc` containers, with Python 3.12 and PyTorch `2.13.0+cu130`. No packages or running services were changed, and no model files were downloaded.

The loader imports finite kernel service-time samples through explicit current-process bindings. Code/hardware/condition identities, symbol, dimensions, dynamic shared memory and packed parameters must match. The embedding harness is responsible for verifying those external identities; matching strings does not independently prove binary identity or calibration accuracy. The committed fixture uses synthetic contract values.

| Contract | Result |
|---|---|
| Registered ABI, graph, profile, provider and file-loader contracts | 24/24 passed |
| File-backed eager kernel | 7 ms service |
| File-backed two-kernel graph after source-graph destruction | 11 + 13 ms service |
| Ledger including initial default-provider launch | Exactly 41 ms = 10 + 7 + 11 + 13 |
| Source and binding/code/hardware/condition identities | Retained on scheduled predictions |
| Sample index | Zero-based indices retained; exhaustion rejects another replay |
| Invalid headers, scope/costs, identities, launch metadata, durations and bindings | 36 cases rejected |
| Duplicate binding IDs and input read failure | Rejected |
| Valid prefix followed by an invalid row | Entire file rejected; no partial provider |
| CRLF, zero duration and binary packed bytes `00ff` | Accepted and checked |
| Saved four-card 5060 Ti and A100 attributes/peer matrices | Both passed; 588 attributes per system |
| PyTorch copy/add on each device with four graph replays | 16/16 processes passed across both systems |

## Process-time comparison

Both versions used the same saved A100 system manifest and device-3 graph fixture. ABBA then BAAB ordering supplied four samples per version. Times are medians including container startup, Python/PyTorch import, capture and four synchronized replays. These runs retain the default provider; they check existing paths after the provenance-record change, not file-import throughput or hardware/model acceleration.

| Fixture | Before | After | Before / after |
|---|---:|---:|---:|
| Copy + cross-stream fork/join | 2.493 s | 2.442 s | 1.021× |
| Add + cross-stream fork/join | 2.538 s | 2.621 s | 0.968× |

The two directions differ and include startup noise; this sample does not establish a general speed improvement.

## Reproduction

Run `build/replay_file_contract tests/fixtures/kernel.timing` in the no-GPU container. The fixture contains three explicitly synthetic samples, loaded through `load_measured_replay` and executed through the normal eager/graph paths. Re-run the registered contracts and the profile/graph commands in the [cross-validation report](../2026-09-29/a100-cross-validation.md) for both saved system manifests.

The input schema and binding responsibilities are documented in [README](../../README.md#kernel-service-time-providers). There is still no automatic binding of arbitrary framework kernels, environment selector or coordinated offline clock. This iteration adds no new real-GPU calibration or model E2E claim.
