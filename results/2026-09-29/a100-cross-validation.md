# A100 and RTX 5060 Ti cross-validation

Simulator revision: `519ff75`. The same collector and native graph fixture were used on a second host with four NVIDIA A100-SXM4-40GB devices, driver 595.91.07 (Driver API 13.2), CUDA 13.0 build headers and GCC 13.3. Tools compiled with `-O2 -Wall -Wextra -Werror`. Each card reports compute capability 8.0, 108 SMs, 40 MiB L2 and 42,405,855,232 bytes of Driver memory.

The host had no active GPU compute processes at the initial check. Its preconfigured model service started loading during validation and was left untouched. No packages, services, clock settings or environments were changed by this work. Native C++ probes ran on the A100 host; Python checks used the existing `0z5a` environment on the original host, inside offline containers with no GPU devices. This task downloaded no model weights; the service's own files were preserved.

## Capability and peer contracts

| Check | Four RTX 5060 Ti | Four A100 SXM4 |
|---|---|---|
| Captured attributes reproduced by FakeCUDA | 588/588 | 588/588 |
| Device names, capacities and distinct virtual UUIDs | Passed | Passed |
| Directed peer-access matrix | 12/12 unavailable | 12/12 available |
| FakeCUDA peer enable | 12/12 return unsupported (217) | 12/12 succeed |
| FakeCUDA duplicate enable / disable / duplicate disable | Unavailable pairs remain disabled | 12/12 match expected 704 / success / 705 |
| Native peer lifecycle on this run | Host no longer reachable; saved capture reused | 12/12 passed |
| A100-profile PyTorch copy/add graph capture and four replays | — | 8/8 processes passed |

Both saved systems were rechecked against the same unchanged simulator binary. The native Driver fixture now also checks enable, duplicate enable, disable and duplicate disable, and the no-GPU profile fixture checks the corresponding state transitions. No hardware-specific simulator changes were needed.

All A100 pairs report `NODE`; all four cards report NVLink links inactive. Both read and write P2P topology queries report `OK`. Direct peer access works in this configuration, but these observations do not establish NVLink transport or measured peer-copy bandwidth. The [earlier 5060 Ti capture](four-device-validation.md) reports no direct peer access.

## Native graph correctness and speed

The unchanged timing workload uses 16 KiB pinned buffers, integer doubling and two-stream fork/join dependencies. Three independent runs on each A100 passed eager numerical checks and four changed-input graph replays after source-graph destruction: 12/12 device-runs. A subsequent run adding peer-lifecycle checks also passed on all four devices; its timings are excluded from the comparison below.

Each run uses five warmups and four alternating ABBA/BAAB rounds, 100 submissions per timing sample. Values below are medians of the three per-run medians; speed ratio is median eager time divided by median graph time.

| GPU | A100 eager µs | A100 graph µs | A100 eager / graph | Previous 5060 Ti eager / graph |
|---:|---:|---:|---:|---:|
| 0 | 20.625 | 20.581 | 1.002× | 0.931× |
| 1 | 20.475 | 19.211 | 1.066× | 0.842× |
| 2 | 19.723 | 19.877 | 0.992× | 0.837× |
| 3 | 20.948 | 19.019 | 1.101× | 0.886× |

These are native host-wall measurements of a small Driver fixture. CPU, driver and resident workloads differ between hosts. The 5060 Ti host had an existing model service. The A100 host was initially idle; the three timing runs completed at 11:40:44–46 UTC, before its service logged model loading at 11:41:53. Background activity was not controlled. The table compares observed graph overhead, not GPU compute throughput, model speedup or FakeCUDA timing accuracy. A100 per-run ratios ranged from 0.924× to 1.163×, so graph execution is not uniformly faster.

No model E2E result is claimed on this host: its preconfigured inference endpoint was not listening. Existing earlier model-service results remain in the 5060 Ti report. FakeCUDA graph checks validate control flow only; numerical results above come from the real Driver.

## Reproduction

Use `capture_device_profile REAL_DRIVER N > device-N.profile` for each device, followed by `capture_device_profile REAL_DRIVER --system > devices.system`. Run `driver_graph_contract REAL_DRIVER` on the real host. In an offline no-GPU container, load the simulator with `FAKE_CUDA_SYSTEM` set to that manifest and run `profile_contract.py --library SHIM --system devices.system --devices 4`, then `graph_contract.py --mode fake --device N --kernel copy` and `add` for ordinals 0–3. Repeat the profile contract with the saved 5060 Ti manifest to cover unavailable peers.
