# Device profile and launch metadata validation

Baseline: `a566dcc` (draft PR #1, now the fork's main). Tested implementation: `0d6c785`.
Both libraries were built from source with GCC 11.4, CUDA 13 headers and `CMAKE_BUILD_TYPE=Release`. Tests used the existing Python 3.12 / PyTorch `2.13.0+cu130` environment, mounted read-only in `python:3.12-slim` with `--runtime=runc --network=none --pull=never`, without GPU devices or environment changes. Test executables ran directly, without a timeout runner.

| Contract | Result |
|---|---|
| Existing Driver ABI, capture, graph lifecycle, multistream, multidevice and thread probes | 10/10 passed |
| Counts 1, 2, 4, 8, 16, 24, 32, 257; ordinal UUID uniqueness; independent memory accounting | 8/8 passed |
| Synthetic profile: name, attributes, capacity, allocation rejection and missing attributes | Passed |
| Invalid counts and profiles, including overflow, duplicate attributes and missing files | 17/17 cases passed |
| Launch records: eager/captured geometry, dynamic SMEM, packed buffer lifetime, repeated replay, module/library identity, unknown ABI and invalid inputs | Passed |
| Collector → profile loader round trip against baseline simulator | 147 collected attributes, name and capacity matched across 32 devices |
| PyTorch cross-stream capture and four replays, copy/add | 16/16 processes passed across baseline and implementation |

The first five rows are 21 registered test invocations. Default-profile checks also compare all 148 original GH200 attributes, including the attribute beyond this build header's sentinel. The collector queries only attributes known to its headers; its round trip is synthetic, not a hardware measurement.

## Speed comparison

Each fixture ran in alternating baseline/implementation order: ABBA then BAAB, four samples per version. Median wall time includes Docker startup, Python/PyTorch import, capture, four replays and synchronization. Ratio is baseline / implementation. These observations do not establish a GPU or model-inference speedup.

| Generic graph fixture | PR #1 baseline | Implementation | Observed ratio |
|---|---:|---:|---:|
| Copy + cross-stream fork/join | 2.605 s | 2.504 s | 1.040× |
| Add + cross-stream fork/join | 2.645 s | 2.636 s | 1.003× |

## Reproduction and remaining work

Build both revisions with the same Release options. `ctest --test-dir build --show-only` lists the individual contract commands; run them inside the offline no-GPU container. `tests/unit/device_profile_inputs.py` takes the built `device_profile_contract` and `tests/fixtures/synthetic.profile` paths. For the process-time comparison, run `tests/integration/pytorch/graph_contract.py --mode fake --kernel copy` (then `add`) with each library selected through `LD_PRELOAD`, using the order above.

Real four-GPU tests and hardware profile/topology capture await access to the new test host. No new model files were downloaded. Kernel images and `kernelParams` layouts remain opaque; static resources, typed pointer/allocation-generation snapshots, measured timing and multi-node topology remain follow-up work. The simulator still uses the fixed 10 ms kernel heuristic and does not execute numerical kernels.
