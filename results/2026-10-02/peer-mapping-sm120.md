# Peer mappings and SM120 validation, 2026-10-02

FakeCUDA now supports directed peer device-pointer access and mapped ordinary D2D in eager submission and Graph replay. The implementation at `810cb39828ae301180f8cedef4a2efb376078db8` passes **38/38 registered contracts inside a real no-GPU Docker container**. Native A100 mapping metadata passes, while peer reads remain numerically incorrect on that host. Native RTX 5090 direct mapping is unavailable and stays unavailable in its captured simulator configuration.

## Simulator behavior and no-GPU evidence

Peer access belongs to an observing context and a particular allocation-owning context. Enabling access includes existing and future allocations, without granting access to another context on the same GPU. UVA context/device/range/backing metadata remains visible across live contexts before and after mapping changes; `DEVICE_POINTER` requires access. `MAPPED=1` identifies backing allocation, not peer permission. These observations agree with the native controls below and the [Driver pointer contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__UNIFIED.html).

Mapped D2D retains both allocation owners. Cross-device copies use the actual source→destination queue; copying between two allocations on the same remote GPU uses that GPU's compute queue. Bounds/readiness/mapping are checked before submission and again on replay. Revocation, free and owner retirement reject stale Graph uses without committing resource timelines, service charges or model predictions. Executables remain usable after their source graph handle is destroyed. Only the owner can free an allocation; synchronous free and teardown include remotely submitted uses, and async D2D/peer free rejects an earlier unordered release.

The new `peer_mapping_contract` uses the synthetic three-device directed fixture. It checks these permissions, ownership, future allocations, separate contexts, bounds, routing, owner drain and failed-replay atomicity, including temporal failure after prediction. The CUDA 13 ABI multidevice probe also submits a mapped D2D through the exported Driver entry point; existing CUDA 12/13 ABI contracts pass. Registry validates context/capability state before acquiring the scheduler mutex; mapping state never requires a reverse scheduler→Registry lock.

| No-GPU check | Completed result |
| --- | --- |
| Registered Driver/CPU contracts | **38/38 PASS**, including peer mapping and CUDA 12/13 ABI |
| PyTorch discovery/tensor/memory/streams/Graph/multidevice | **12/12 PASS**: six probes × cu128/cu130 |
| Captured SM120 profiles and directed access matrix | Both cards match all captured fields; both directions remain 0 |
| SM120-profile copy/add Graph control flow | **8/8 configurations**, 32 replays: two cards × two wheels × two operations |

Docker inspection records `Runtime=runc`, `NetworkMode=none`, no device mappings/requests and an empty `/dev/nvidia*` list. The registered commands run directly through `run_contracts.py` without timeout termination. A container-only tmpfs supplies CTest's `build/Testing` logs while source/interpreter mounts remain read-only. The shell wrapper's original hardcoded aarch64 expat path failed on x86_64; it now resolves the host architecture and mounts the actual system or managed Python prefix. The fresh framework/container checks close the earlier Docker verification gap.

These simulator checks cover timing/control flow. No device bytes or kernels are executed. Host transfers/memset still require local ownership; opaque kernel pointer arguments are not decoded. Explicit `cuMemcpyPeer[Async]` during capture remains unsupported; this addition covers ordinary mapped D2D capture.

## Native SM120 controls

The host has two RTX 5090 GPUs, compute capability 12.0, Driver 580.105.08 and a KVM guest kernel. Native calls load the absolute, hash-frozen NVIDIA Driver, with FakeCUDA/preload/device-visibility variables removed. The existing shared GPU lock covers each complete finite queue and is inherited by children. Boot, assigned UUIDs, native-library hash and empty compute admission are rechecked before/after each native child. Desktop graphics consumers remain present; other compute resumes after our queues finish, so no sustained host isolation is claimed.

Private environments use Python 3.12.11, PyTorch **2.11.0+cu128** and **2.11.0+cu130**, NumPy 2.2.6, CMake 4.1.3, GCC 12.3 and NVRTC 13.0. Environment installs/freezes/checks are retained separately.

| Native check | Completed result |
| --- | --- |
| Selector values/edges | **144/144 configurations**: 36 × two GPUs × two wheels |
| PyTorch copy/add Graph with changed inputs/saved outputs | **8/8 configurations**, 32 replays PASS |
| Small two-device BF16 dispatch/expert/FP32 combine fixture | **2/2 PASS**, one per wheel |
| Driver Graph oracle using embedded SM80 PTX | Both GPUs PASS, four changed-input replays each |
| Driver Graph oracle using genuine SM120 cubin | Both GPUs PASS; loaded binary version **120** on each |
| Raw Driver copy/source/staging matrix | **26 PASS, 0 FAIL, 8 SKIP_UNSUPPORTED**; all 26 destination/reset checks pass |
| Peer pointer metadata/state fixture | **42/42 checks PASS**; all 16 mapped-copy cases explicitly skipped |

The 34-observation raw matrix contains ten source/same-device checks, four explicit `cuMemcpyPeer[Async]` transfers and twelve pinned-host D2H/H2D Graph observations. All 26 executed observations pass. Eight ordinary mapped-peer observations are skipped because `cuDeviceCanAccessPeer=0` in both directions; enable returns 217 and disable returns 705. The four successful explicit Peer API transfers have an **unmeasured route** and do not establish direct mapping. NCCL logs show `SHM/direct/direct` channels; actual runtime versions are `2.28.9+cuda13.0` and `2.28.9+cuda12.9` for the cu130/cu128 environments respectively. Their numerical success does not turn the captured direct-access zeros into ones.

The optional oracle image implements `double_u32(input, output)` with 128 threads per block. CPU-only NVRTC compilation uses `--gpu-architecture=sm_120 --std=c++17`; the retained 5,336-byte image loads on both cards with `CU_FUNC_ATTRIBUTE_BINARY_VERSION=120`. The default SM80 PTX run exercises Driver JIT instead; [Blackwell compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html) and [NVRTC image generation](https://docs.nvidia.com/cuda/nvrtc/index.html) distinguish these paths.

| Host-wall oracle timing | Eager µs | Graph µs | Eager/Graph |
| --- | ---: | ---: | ---: |
| PTX, device 0 | 7.457 | 8.623 | 0.865 |
| PTX, device 1 | 6.751 | 7.845 | 0.860 |
| SM120 cubin, device 0 | 7.314 | 8.040 | 0.910 |
| SM120 cubin, device 1 | 6.544 | 8.142 | 0.804 |

These desktop-host observations use alternating eager/Graph timing order and show no Graph speedup. They are not isolated kernel-service calibration or whole-model measurements.

## Native A100 mapping control without host root

The existing two-A100-SXM4-40GB container exposes the normal userspace Driver API while lacking host-kernel administration. The previous native queues have exited; vLLM remains stopped. The same finite guard uses its existing shared lock, boot, two UUIDs and unchanged Driver 595.71.05 library. No host driver, module or IOMMU setting is changed.

Each direction creates legacy contexts with `cuCtxCreate_v2`, enables only the observer→owner mapping, and queries seven attributes before enable, after enable and after disable. **All 42 checks pass.** Context/device/range/backing metadata remains valid; `DEVICE_POINTER` changes from error 1 to the unchanged UVA address after enable, then back to error 1 after disable. Both directions advertise access 1, enable returns 0, and disable returns 0.

The numerical fixture tests 4,096 uint32 elements, two changed inputs per eager/Graph mode, independently verifying the source upload and the zeroed destination before each observation. All **16 source readbacks and 16 resets pass**. Each Graph executable replays twice after its source graph is destroyed; observer stream completion precedes owner-context readback.

| Observer operation | Eager pass/fail | Graph pass/fail | Total |
| --- | ---: | ---: | ---: |
| Read peer into local allocation, both observers | 0/4 | 0/4 | **0 PASS, 8 FAIL** |
| Write local allocation into peer, both observers | 4/0 | 4/0 | **8 PASS, 0 FAIL** |

Every failed read has **4096/4096 mismatches**; API submission itself succeeds. Both Driver ordinals are tied to actual UUID/PCI identities in the retained native result, rather than inferred from NVML order. This legacy-context/read-write fixture is separate from the [previous 34-observation directional control](a100-peer-control.md); their counts and conditions are not pooled. Mapping availability alone does not qualify peer data transfer. The earlier `DMA-FQ` host condition remains a candidate cause pending a host-operator before/after control.

## Retained identities and boundaries

| Identity | SHA-256 / value |
| --- | --- |
| Tested implementation | `810cb39828ae301180f8cedef4a2efb376078db8` |
| Tested candidate source archive | `61a6337d1e0351e1d1a9954a1859d3f8b6f14d512c20729dda7707cd0908554c` |
| Built FakeCUDA library | `9ba90f80629088997430f5b5760742b342c9b598222766faf97cc9de3ed23748` |
| Native mapping probe source | `87040fc133c123b8d7a9cb7588537ad05d2daf1cba74126f1bab31fabd1eca07` |
| SM120 CUDA source | `7c7c0cce14c209ff62969ebeea97bc5816bf46c5ba6a658cc83b96092becc401` |
| Genuine SM120 cubin | `93af3115882b163f50c1f66d735d21382ef2dfb197a6b7dfb7b7c76fec51e6c5` |
| Native 5090 Driver | `df9183549feb062f4195e6cf130e0ef372de4a59e59dbe51554ad8c3c5b167db` |
| Native A100 Driver | `76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6` |
| SM120 evidence archive | `5cfc80b2ba334f0636db219a2d6eb3d07d08cb81e883574e7bec7606b58b8831` |
| A100 mapping evidence archive | `3b83ece04a03c7a54aaa93faaa946aabaac05760d4a3fae628a9003685d09f2e` |

Raw logs/JSON/profiles, images, frozen environments, source snapshots, admission receipts and manifests remain outside Git. The Mac backups verify all **200 SM120 and 16 A100 files** by size and SHA-256. Failed setup/container attempts and lock deferrals are retained separately from completed runs. Processes exit naturally and managed jobs have restart disabled; the resource window is released.

This campaign does not rerun the complete Granite/Kimi numerical matrices or change their strict failures in [model-functional.md](../2026-10-01/model-functional.md). The original Kimi TP2 campaign required over 46 GiB per rank and cannot be reproduced unchanged on these approximately 32 GiB 5090s. Small Driver/framework/EP contracts do not qualify full inference or new performance predictions; PR #2 remains draft.
