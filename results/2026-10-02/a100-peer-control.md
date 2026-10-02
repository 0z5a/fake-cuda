# Native A100 peer-copy control, 2026-10-02

The completed synchronized control passes **28/34 observations** and fails six direct peer observations. Explicit pinned-host staging passes in both directions. Direct peer data correctness remains unqualified on this host; FakeCUDA and Torch are absent from the copy process.

## Completed control

Every observation compares all 4096 FP32 elements (16 KiB) exactly. Changed-input Graph replay uses `0.125`, `-0.75` and `2.0`. Before each upload or device copy, a zero reset, context completion and full host readback verify the relevant buffer's starting contents. All **34 reset checks pass**. The native process completes cleanup and naturally exits 1 because six numerical checks fail.

| Observation | Cases | Pass | Fail |
| --- | ---: | ---: | ---: |
| Source uploads and same-device copies | 10 | 10 | 0 |
| Synchronous `cuMemcpyPeer` | 2 | 1 | 1 |
| Mapped-peer eager `cuMemcpyDtoDAsync_v2` | 2 | 1 | 1 |
| Explicit eager `cuMemcpyPeerAsync` | 2 | 1 | 1 |
| Mapped-peer Graph with changed input | 6 | 3 | 3 |
| Pinned-host D2H Graph and host readback | 6 | 6 | 0 |
| Pinned-host H2D Graph and device readback | 6 | 6 | 0 |
| **Total** | **34** | **28** | **6** |

All six failed observations transfer from **Driver ordinal 0 to 1**, each with **4096/4096 mismatches** and eight retained sample values equal to zero. All six direct transfers from ordinal 1 to 0 pass. Direction labels refer to native Driver ordinals; the UUID inventory below comes from NVML.

The final control tests 1→0 before 0→1. Its preceding synchronized trial tested the opposite order and recorded the same six failures. That preceding trial did not reset the destination between synchronous Peer and mapped-peer eager copies, so its successful eager observation could have retained the previous value. The final control fixes this gap and independently verifies resets for every observation. Neither stale destination contents nor the tested direction order explain the remaining 0→1 failures.

Source uploads complete and are read back before peer copies. Destination initialization completes before synchronous/asynchronous copies and Graph launches; non-default stream work completes before numerical readback. Host staging uses portable pinned memory, a source D2H Graph and destination H2D Graph, with a CPU stream-completion barrier between them. Its success covers this explicit 16 KiB protocol and these values, rather than general model inference or direct P2P.

## Admission and independent NCCL evidence

The original peer waiter naturally retired with exit 75 before CUDA initialization when NCCL attempt 040 failed lock admission. Attempt 041 subsequently completes with runner and both rank exits 0. Each rank contains all **144 unique** operation/dtype/count/round combinations, finite and zero mismatch: four operations, three dtypes, four element counts and three rounds. This is **288 successful data checks**, rather than a prepared plan or process-start claim.

The queue owner returns its finite window before these native controls. Each control rechecks the successful NCCL artifacts, current boot/UUID inventory and empty NVML compute list under the shared whole-queue lock. The native child inherits the locked descriptor until its natural exit. Both control parents/children naturally terminate; their Supervisor programs have automatic restart disabled. Compute activity appears in later observations, so this provides no sustained host-isolation or timing qualification.

NCCL logs record bidirectional Channel 00/01 `via SHM/direct`, with only debug environment settings in the retained rank metadata. Rank JSON reports NCCL `[2, 29, 7]`, while both runtime logs announce `2.30.7+cuda13.3`; these version fields are retained separately. NCCL numerical success and its chosen transport do not prove native Driver peer-copy correctness.

## Identities and remaining host condition

| Identity | Recorded value |
| --- | --- |
| Hardware | Two NVIDIA A100-SXM4-40GB; Driver 595.71.05 |
| Boot | `5014bee8-18f5-4eb1-b67d-df7dfb9d9f90` |
| NVML UUID inventory | `GPU-03d6de3a-ca5b-03ca-727f-698cddfe0c46`, `GPU-14d8e70c-93be-2085-471f-28f7fed2ec91` |
| Absolute native library | `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05` |
| Native library SHA-256 | `76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6` |
| Final control source SHA-256 | `e1ff452f5553062fa8f8326fe391e2a3ad957762c9571e9cffa11d5f03cf2406` |
| NCCL probe source SHA-256 | `58cf6315546b8d197c074b2a00ba8b668f09df8f767de40acef7f800df9e6962` |
| NCCL plan SHA-256 | `d428b6a1949ef6ed7cf6007a8e6f0dc7f07218b8a26f9844921bd3a4e899fced` |
| Final 12-file evidence archive SHA-256 | `d97d5fafe8b1006a5c29d2a1715638a0d145dba2013cecdcc86943586eb90c7f` |
| Preceding 26-file evidence archive SHA-256 | `6f9bb836d797242c894fae8929b8909c20fd9866b5b028079dad54507724f1dc` |

Raw source, JSON, logs and manifests remain outside Git and have SHA-verified Mac backups. These native controls do not rerun the complete Granite/Kimi models or replace the 37 CPU contracts validated at implementation commit `fddafc7`.

GPU IOMMU groups 48/49 remain `DMA-FQ`, translated DMA according to the [Linux ABI](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-kernel-iommu_groups). Read-only board/CPU/VM observations support a bare-metal host. [NVIDIA documents](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#host-iommu-hardware-pci-access-control-services-and-vms) that this host configuration is unsupported for PCIe P2P and may corrupt data. It is a strong candidate cause, with causal confirmation still requiring a host-operator configuration correction and an unchanged before/after control. The current container has read-only sysfs and lacks host administration capability.
