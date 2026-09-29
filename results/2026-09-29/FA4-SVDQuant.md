# FA4 and SVDQuant kernel follow-up

Real RTX 5090 (device 7), existing PyTorch 2.13.0+cu130 environment, driver 580.82.07. No installed packages, clocks, power settings, or other processes were changed. Source checkouts and Nunchaku build outputs are task-local. No model checkpoints are required by these operator tests.

## FA4

Official [FlashAttention source](https://github.com/Dao-AILab/flash-attention/tree/e9cf2c1651d2303191eb40a739a3c135fda00999), commit `e9cf2c1651d2303191eb40a739a3c135fda00999`; `flash_attn.cute.flash_attn_func`, SM120 forward, FP16, batch 1, 8 heads, head dimension 64. Existing dependencies: Cutlass DSL 4.6.2, quack-kernels 0.6.4, apache-tvm-ffi 0.1.11, torch-c-dlpack-ext 0.1.5. The source declares tvm-ffi >=0.1.12; the installed 0.1.11 passes these cases, without implying general version compatibility.

Each case checks eager output and three Graph replays against independently calculated FP32 softmax attention, including changes at stable input addresses; atol=rtol=0.003. The maximum observed absolute error is below 0.0011. Initial harness execution revealed that the current public API returns a tuple; the final benchmark selects its output tensor.

Three fresh processes, four alternating APPA/PAAP blocks per case, 100 invocations per arm, five untimed warmups. Entries below are medians of each process's median wall time; speedups are medians of per-process ratios. Timings include host submission and final completion, exclude compilation/capture, and do not represent isolated device kernel time or full-model speedup. CUDA-event measurements and every arm are in `fa4-run{1,2,3}.json`. Nunchaku CPU compilation overlapped these FA4 measurements; no second benchmark shared device 7.

| Tokens | Causal | Eager µs | Graph µs | Speedup | Observed process range |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 512 | False | 64.400 | 16.593 | 3.882× | 3.788–3.984× |
| 2048 | False | 65.697 | 53.507 | 1.228× | 1.176–2.090× |
| 2048 | True | 64.917 | 57.683 | 1.126× | 1.100–1.160× |

GPU-free FakeCUDA: import and virtual allocations pass; warmup fails in CuTe/TVM `cuda_dialect_init_library_once` with `cudaErrorUnknown`. No capture/replay or numerical success is claimed for this path. See `fa4-fake.log` and `fa4-fake-lookup-excerpt.log`: library load and kernel lookup return success, then `cuKernelGetAttribute` is unresolved immediately before the error. No success stub was added. The simulator exposes GH200/SM90, so this exercises the SM90 source path, not the real GPU's SM120 binary.

## SVDQuant W4A4 plus low-rank branch

Official [Nunchaku source](https://github.com/nunchux-ai/nunchaku/tree/302e0e97024ebd68688fe890e5df83731edf7b54), commit `302e0e97024ebd68688fe890e5df83731edf7b54`. The installed environment has no Nunchaku package and upstream wheels do not match its PyTorch 2.13. We compiled official sources with the existing toolchain. While the full extension continued compiling its unrelated attention units (still running at report finalization), `build_svdquant_binding.py` linked the 23 completed official core objects into a test-only extension exposing the two unchanged functions from `nunchaku/csrc/ops.h`. No substitute numerical kernel is used. The optional full-package build was left to exit naturally; its completion is not required for the tested operator binding. Results here refer to that test binding, not the complete Nunchaku model package.

The measured operation includes activation quantization fused with the rank-32 down projection, followed by official W4A4 GEMM fused with the up projection. K=1024, N=2048, group size 64; FP16 activation/output, signed INT4 weights, nonzero FP16 low-rank weights. Random inputs are placed on an exactly representable quantization grid (activation steps 1/8, weight steps 1/32, each activation group reaches magnitude 7/8); this tests packing and execution independently of quantizer rounding ambiguity. The reference is FP32 `A @ W.T + (A @ down.T) @ up.T`. Low-rank pack/unpack is exactly reversible. Eager and three changed-input Graph replays pass atol=rtol=0.02 for all shapes in all six fresh processes. Maximum absolute errors: 0.01468, 0.01653 and 0.02111 respectively; the separate low-rank contribution reaches 0.51–0.55, and is included explicitly in the numerical reference.

**Initial shared-GPU measurements were unstable.** The same four APPA/PAAP blocks and 100 invocations per arm were used. Device 7 was initially free, but another SGLang scheduler appeared and held approximately 18 GB while these measurements were collected; see `svdquant-shared-gpu.txt`. The second process shows Graph regressions and absolute timings vary strongly between processes. We retain each process separately rather than combining unrelated medians into a misleading improvement claim. Concurrent activity is a confounder; these observations do not prove its exact contribution to the variance.

| Process | M | Eager µs | Graph µs | Eager / Graph |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 296.925 | 293.818 | 1.011× |
| 1 | 1024 | 299.106 | 295.494 | 1.012× |
| 1 | 4096 | 569.097 | 544.855 | 1.044× |
| 2 | 256 | 156.245 | 203.709 | 0.767× |
| 2 | 1024 | 180.850 | 209.673 | 0.863× |
| 2 | 4096 | 327.291 | 333.788 | 0.981× |
| 3 | 256 | 160.332 | 152.791 | 1.049× |
| 3 | 1024 | 142.411 | 135.361 | 1.052× |
| 3 | 4096 | 281.101 | 272.797 | 1.030× |

Once the device was observed idle with the new scheduler and its model still resident, three additional fresh processes were run to resolve the timing variance. These showed consistent modest improvements below. This is a conditional shared-GPU observation, not an exclusive-device performance guarantee; all earlier runs above remain part of the record. The new process's startup timing and observed occupancy motivate the rerun but do not establish the exact cause of every outlier.

| M | Follow-up eager µs | Follow-up Graph µs | Median speedup | Process range |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 134.951 | 131.326 | 1.028× | 1.026–1.028× |
| 1024 | 135.431 | 131.409 | 1.031× | 1.030–1.031× |
| 4096 | 262.850 | 260.434 | 1.009× | 1.009–1.009× |

Extension SHA256: `c341dd753891a4334a29b90f941c748387c317f2ddf3e8020397302a11f91121`. Raw records: `svdquant-binding-run{1,2,3,4,5,6}.json`. A preliminary five-repetition check was used to validate the harness and is excluded from this performance table. These tests cover a synthetic linear pipeline, not SVD decomposition quality, NVFP4, pretrained model quality, or full-model E2E.

GPU-free FakeCUDA: imports and virtual allocations pass. The first fused activation-quantization warmup fails at official `gemm_w4a4_launch_impl.cuh:498`, `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)`, with unsupported Driver API. The trace shows unresolved `cuFuncSetAttribute` / `cuKernelSetAttribute`. Capture is not reached. See `svdquant-binding-fake.log`; no success stub was added.

## Source-only reproduction

Use the existing interpreter and matching source commits; do not run pip install or build isolation. FA4 uses a namespace containing only `flash_attn/cute` (as its own packaging does). The Nunchaku namespace links the compiled extension, official operators and weight packer, skipping unrelated model imports. Neither kernel source is modified. The tested SVDQuant binding is explicitly linked below; the full model extension is not required once all 23 core objects exist.

```sh
# Source checkouts under this task's kernels directory:
# flash-attention: e9cf2c1651d2303191eb40a739a3c135fda00999
# nunchaku: 302e0e97024ebd68688fe890e5df83731edf7b54
# Initialize Nunchaku's recorded submodule revisions before copying sources.
cd /home/gongji/0z5a/work/fake-cuda-pcie-test/kernels/nunchaku
CUDA_VISIBLE_DEVICES=7 MAX_JOBS=2 NUNCHAKU_BUILD_WHEELS=1 \
  /home/gongji/0z5a/bin/python setup.py build_ext --inplace
cd /home/gongji/0z5a/work/fake-cuda-pcie-test
CUDA_VISIBLE_DEVICES=7 MAX_JOBS=1 /home/gongji/0z5a/bin/python \
  patched/tests/integration/tooling/build_svdquant_binding.py \
  --source kernels/nunchaku --build kernels/nunchaku-kernel-binding
/home/gongji/0z5a/bin/python patched/tests/integration/tooling/prepare_kernel_namespaces.py \
  --fa4 kernels/flash-attention --nunchaku kernels/nunchaku --output kernels \
  --svdquant-extension kernels/nunchaku-kernel-binding/_C.so
CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$PWD/kernels/fa4" \
  /home/gongji/0z5a/bin/python patched/tests/integration/tooling/benchmark_fa4_graph.py \
  --output evidence/fa4-run1.json
CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$PWD/kernels/nunchaku-ops" \
  /home/gongji/0z5a/bin/python patched/tests/integration/tooling/benchmark_svdquant_graph.py \
  --output evidence/svdquant-binding-run1.json
```

Repeat each benchmark in three fresh processes with distinct output paths. For the no-GPU probes, use the parent report's read-only Docker mounts and `LD_PRELOAD`, add the appropriate `PYTHONPATH` above, and run `tests/integration/pytorch/{fa4,svdquant}_control_flow.py`. Keep `--runtime=runc --network=none --pull=never` and no device mounts. Lookup diagnostics use `FAKE_CUDA_TRACE=1`; never use tracing in the real performance runs.
