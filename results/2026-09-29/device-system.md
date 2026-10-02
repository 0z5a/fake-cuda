# Per-device profiles and directed peer access

Baseline: `98ed658`. Tested implementation: `18f8e7a`. Both versions were built in Release mode with GCC 11.4 and CUDA 13 headers. All simulator/PyTorch checks ran in the existing `0z5a` environment inside offline no-GPU `runc` containers. The hardware collector used the unchanged four-RTX-5060-Ti host and its real Driver; no environment or running service was modified.

`FAKE_CUDA_SYSTEM` loads all four measured profiles in one process and supplies the directed access matrix. The previous mode replicated card 0's profile across all ordinals and assumed peer access for every pair.

| Check against the four-card capture | Before | After |
|---|---:|---:|
| Attribute mismatches across 4 × 147 queries | 7 | 0 |
| Unsupported directed pairs incorrectly reported accessible | 12/12 | 0/12 |
| Enable calls on those pairs | 12 × success | 12 × `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` (217) |
| Correct device slots, capacities and distinct virtual UUIDs | Profile 0 replicated | All four profiles matched |

The peer-access matrix is distinct from the copy transport model. `cuMemcpyPeer` scheduling still uses the existing virtual outgoing queue and bandwidth heuristic; host-staged routes and NUMA/link contention are not calibrated by this change.

## Validation

| Test | Result |
|---|---|
| Registered contracts, including the previous ABI/graph/thread probes | 22/22 passed |
| Invalid profile/count/system input, matrix completeness and conflicting selectors | 32/32 rejected as expected |
| Relative paths containing spaces and a 16-device absolute-path system | Passed |
| Synthetic mixed devices: capacity, allocation admission, device-specific launch limits, shared context accounting | Passed |
| Asymmetric peer graph, enable/duplicate-enable/disable semantics | Passed |
| Real four-card profile + matrix round trip in one fake process | Passed |
| PyTorch cross-stream copy/add capture and four replays on devices 0–3 | 8/8 processes passed |

## Process-time comparison

The same device-3 graph fixture ran ABBA then BAAB, four samples per version. Before used the previous supported single-profile configuration; after used the measured four-profile system. Median wall time includes container startup, Python/PyTorch import, capture and four synchronized replays. This is an observed no-GPU regression comparison, not a kernel/model speedup or calibrated prediction.

| Fixture | Before | After | Before / after |
|---|---:|---:|---:|
| Copy + cross-stream fork/join | 2.495 s | 2.420 s | 1.031× |
| Add + cross-stream fork/join | 2.599 s | 2.513 s | 1.034× |

## Reproduction

Capture each device with `capture_device_profile REAL_DRIVER N > device-N.profile`, then capture `capture_device_profile REAL_DRIVER --system > devices.system` in the same directory. Clear the legacy profile/count selectors and set `FAKE_CUDA_SYSTEM` to the manifest before loading the shim. Inside the no-GPU container, run `profile_contract.py --library SHIM --system devices.system --devices 4` followed by `graph_contract.py --mode fake --device N --kernel copy` and `add` for N = 0–3. These scripts are under `tests/integration/tooling` and `tests/integration/pytorch`, respectively. The native numerical checks and live model-service smoke test remain documented in [four-device validation](four-device-validation.md).
