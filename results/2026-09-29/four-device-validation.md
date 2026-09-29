# Four-device hardware validation

Hardware: 4 × NVIDIA GeForce RTX 5060 Ti 16GB, driver 580.126.09, CUDA 13.0 headers, GCC 13.3. Each captured Driver profile reports compute capability 12.0, 36 SMs, 32 MiB L2 and 16,617,766,912 memory bytes. The existing vLLM service remained running, using approximately 13.4 GiB on each GPU. No processes were stopped, packages changed, clocks configured or model weights downloaded.

## Topology and capability profiles

| Pair | Reported connection | Driver direct peer access, both directions |
|---|---|---:|
| 0 ↔ 1 | NODE | 0 |
| 0 ↔ 2 | NODE | 0 |
| 1 ↔ 2 | PHB | 0 |
| 0/1/2 ↔ 3 | SYS, across NUMA nodes | 0 |

All 12 directed `cuDeviceCanAccessPeer` queries returned zero; `nvidia-smi topo -p2p r/w` reported CNS. This checks direct peer access, not the behavior or throughput of host-staged copies. At `98ed658`, FakeCUDA simulated all-to-all peer availability. The subsequent [system configuration change](device-system.md) imports the direct-access matrix; transfer routes and bandwidth remain separate work.

The four captured profiles contain 147 attributes each. Their differences are source identity, PCI bus ID, multi-GPU board group ID and host NUMA ID. Each profile was injected separately into a four-device FakeCUDA process on the original `0z5a` environment, inside an offline no-GPU container. All advertised attributes, names and memory capacities matched, and virtual UUIDs remained distinct. This run selected one profile at a time. The subsequent system configuration change validates all four profiles together in their original device slots.

## Graph correctness and speed

The standalone Driver fixture uses 16 KiB pinned input/output buffers, an integer-doubling PTX kernel, two streams and fork/join events. It checks eager values, destroys the source graph, then verifies four changed-input replays on alternating launch streams, including saved outputs after later replays. All four GPUs passed in three independent runs (12 device-runs).

Each run times four alternating ABBA/BAAB rounds, 100 submissions per sample, with five warmups. The table uses the median of the three per-run medians. Host-wall timing includes submission and final synchronization. No CPU affinity or GPU clock pinning was applied; the resident service was preserved. These results describe this small native Driver fixture, not model speed or FakeCUDA prediction accuracy.

| GPU | Eager µs | Graph µs | Eager / graph | Result |
|---:|---:|---:|---:|---|
| 0 | 7.934 | 8.526 | 0.931× | Numerical checks passed; graph slower |
| 1 | 7.746 | 9.197 | 0.842× | Numerical checks passed; graph slower |
| 2 | 7.708 | 9.211 | 0.837× | Numerical checks passed; graph slower |
| 3 | 7.603 | 8.582 | 0.886× | Numerical checks passed; graph slower |

On the original `0z5a` Python 3.12 / PyTorch `2.13.0+cu130` environment, eight no-GPU PyTorch processes (copy/add × devices 0–3) passed cross-stream capture and four replays using the captured profiles. Those checks validate control flow only; numerical results above come from the real Driver fixture.

## Existing model service E2E smoke test

The existing four-worker Qwen/Qwen3.5-9B service accepted three chat-completion requests with temperature zero, thinking disabled and a request to return `READY`. All returned HTTP 200, exact output `READY`, 19 prompt tokens and 2 completion tokens, with finish reason `stop`. Wall times were 159.890, 66.949 and 63.409 ms. This is a live-service E2E smoke test, not an eager/graph model benchmark or inference through FakeCUDA. Existing in-use model files were preserved.

## Reproduce

```sh
cmake --build build --target capture_device_profile driver_graph_contract
build/capture_device_profile /path/to/real/libcuda.so.1 0 > device.profile
build/driver_graph_contract /path/to/real/libcuda.so.1
```

For the no-GPU half, mount the captured file and environment read-only in an offline `runc` container, select it with `FAKE_CUDA_PROFILE`, set `FAKE_CUDA_DEVICE_COUNT=4`, and preload the built shim. Run `tests/integration/tooling/profile_contract.py --library /path/to/shim --profile /path/to/device.profile --devices 4`, then `tests/integration/pytorch/graph_contract.py --mode fake --device N --kernel copy` and `add` for N = 0–3. The new machine's Python environment was not used or modified.
