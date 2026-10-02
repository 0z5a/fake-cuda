# Kernel service-time provider validation

Baseline: `cf371a7`. Tested implementation: `d5df7d6`. Both libraries were built in Release mode with GCC 11.4 and CUDA 13 headers. Tests used the existing `0z5a` environment, Python 3.12 and PyTorch `2.13.0+cu130`, in offline `runc` containers with no GPU devices. No environment updates or process termination were performed.

This is the first P2 slice: an internal C++ provider contract, finite ordered replay, accounting scope and separate time records. Default behavior remains `synthetic_constant_v1`. Replay values in the unit test are synthetic oracle inputs, not measured hardware calibration. An embedding harness must explicitly bind real observations to invocations before using `MeasuredReplay`; no file loader or framework adapter is included.

## Contract results

| Check | Result |
|---|---|
| Registered ABI, graph, device, thread and provider contracts | 23/23 passed |
| Default synthetic kernel duration | Exactly 10 ms |
| Explicit eager replay sample | Exactly 7 ms; consumed once |
| Two kernels per graph replay | 11 + 13 ms, then 17 + 19 ms; 60 ms charged in total |
| Capture and source-graph destruction | No capture charge; executable retains launch metadata |
| Wrong packed parameters or invocation identity/shape/mode | Rejected without queue mutation or cursor advance |
| Unknown parameter layout or exhausted sequence | Rejected; no constant-time fallback |
| Graph batch mismatch and later memory preflight failure | No work submitted, samples consumed or service charged |
| Whole-forward / end-to-end cost, negative/overflow duration, missing source | Rejected |
| Predictor delay injection | Extra 20 ms query wall time; kernel service remains 7 ms |
| Target-host time | Explicitly unknown |
| Saved 5060 Ti and A100 system profiles and peer states | Both passed; 588 attributes per system |
| PyTorch copy/add across four devices, four replays per process | 8/8 per system; 16/16 processes passed |

The kernel service ledger is a sum, not makespan, and excludes transfer costs. The simulator ledger currently records predictor queries only. The scheduler still uses paced wall time: delaying a query can delay submission, so the 20 ms test does not claim strict offline completion-time invariance. Coordinated clocks and target-host CPU modeling remain separate work.

## Before/after process-time comparison

The same saved four-A100 manifest was used on both versions. Device 3 ran the copy and add graph fixtures in ABBA then BAAB order, with four samples per version. Median wall time includes container startup, Python/PyTorch import, capture and four synchronized replays. Only the default provider was selected; these are regression timings, not a model or GPU speedup.

| Fixture | Before | After | Before / after |
|---|---:|---:|---:|
| Copy + cross-stream fork/join | 2.480 s | 2.573 s | 0.964× |
| Add + cross-stream fork/join | 2.561 s | 2.566 s | 0.998× |

No speed improvement was observed in these samples. Per-process startup noise is included; this small sample does not isolate predictor-interface overhead.

## Reproduction and remaining boundary

Build and run `perf_model_contract` inside the no-GPU container alongside the registered contracts. Repeat `profile_contract.py` and `graph_contract.py` with each saved system manifest, as described in the [A100 report](../2026-09-29/a100-cross-validation.md). The provider test binds samples directly to the current process's loaded kernel records and immutable device profiles; portable code hashes, parameter provenance and calibration-domain validation are not implemented.

No fresh native measurement or model E2E result is claimed in this iteration. The supplied A100 SSH endpoint closed the connection when checked. Existing native hardware results remain in the earlier reports; no model weights were downloaded for this work.
