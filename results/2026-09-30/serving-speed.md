# Complete CPU serving process comparison

Three repeats per version, alternating process order, same vLLM 0.30.0 original Scheduler/KV code, frozen calibration and 18 complete workloads. Wall time covers interpreter startup/imports, calibration fit, configuration loading, every scheduler/oracle/predictor/IPC operation and natural subprocess/interpreter teardown. Prior model download, native calibration and build are excluded from both versions.

| Complete workloads per process | Previous s | Optimized s | Simulator speedup | Wall-time reduction |
|---:|---:|---:|---:|---:|
| 18 | 19.060 | 17.001 | 1.121× | 10.80% |

All six processes produced identical virtual token/batch/arrival timelines: `860b206438e601d421f622b685a62985e6a12ca208aaa5d60e0d1498b7e512c5`. These are complete timing simulation runs; the simulator emits an explicit fixed-length token oracle and does not execute numerical inference.

| Binary | SHA-256 |
|---|---|
| Before | `ffae997bf52becba601a85d5ae4422d453c1134e56fbff1ed18e0537e70bccd7` |
| After | `ec51f5789af7c6e6d85d2e006474ba3615a3ecdd90aa70db0937b99de79c5f74` |

The compared engine source differs only in history traversal: completed entries remain queryable, while the optimized engine indexes unfinished work for settle/rate/advance/next-event scans. The benchmarked binaries above precede the unrelated local/global topology helper. Both used GCC 13.3, C++23 and CMake Release on the same machine.

| Operations | Before s | After s | Simulator speedup | Timeline parity |
|---:|---:|---:|---:|---|
| 10000 | 13.433356 | 0.094579 | 142.03× | Every reply identical |
Three repeats, median subprocess wall time including protocol and teardown. This measures simulator execution, not GPU performance.

Reproducible engine baseline: `b5cc5c0fe6169ee17a21db6fe44b184135c8db3c`; unfinished-index change: `74bd9d77ff34487275bf8e959a1ea3182cbf3e8d`. Build the same final harness against the two resource-engine source/header revisions to isolate the optimization.

| Source | SHA-256 |
|---|---|
| src/impl/resource_engine.cpp | `ced63d96387b12177d666195aea858d876b7b1fab564e432e1988c4b9a9c8c2c` |
| include/impl/resource_engine.h | `47050ff10fe7ff1b556a0503afbc2d77776d1e559ed8b90f8fa7b062120f6a58` |
| tests/integration/tooling/resource_engine_bridge.cpp | `d3d4c0a723db9e36e79a28b3ff5d5c8e94a1d79f30815bcfac3c983d49f9e4cf` |
