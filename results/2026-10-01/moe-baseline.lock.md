# M1 baseline lock

Baseline ID: `pr2-63444ef-m1`. Verified 2026-10-01 before implementation.

| Item | Locked value |
| --- | --- |
| Repository / reference PR | `monsoon235/fake-cuda`, draft [PR #2](https://github.com/monsoon235/fake-cuda/pull/2) |
| PR base | `75fd2cd27713373baef17007547a5dda03219223` |
| PR head / implementation base | `63444efab60ac5e78dae4d5d89afc5f22536a895` |
| Exported PR diff SHA256 | `74fe077606305edf1a97edfb4c25f85272a0874de690346e85bad9635ee54fc2` |
| New branch / submission repository | `moe-cost-ledger`, `0z5a/fake-cuda` |
| Stack base | `0z5a/fake-cuda:device-launch-metadata` at the implementation base above |
| Execution-plan SHA256 | `687e383c92bdff1252e4b44f88a50c88d5d4e3704a8e00138d50d7c02ad6a18e` |
| Kimi configuration revision | `e1df551a447157d4658b573f9a695d57658590e9` |
| Configuration SHA256 | `a6ac3c2c4b5aa72370f9727f49ffa4432715d20061889acdb37c688be853096e` |

PR #2 was open/draft, with 127 changed files, 7,437 additions and 119 deletions. Its actual diff and the repository guide were inspected; origin/main was not substituted as the implementation base. The submission extends the owned launch/Graph path rather than rebasing or modifying main.

| Existing component | M1 treatment |
| --- | --- |
| Immutable launch records, Graph ownership and atomic prediction preflight | Reuse; append owned semantic templates/bindings and coverage checks |
| Kernel-only PerformanceModel / MeasuredReplay | Retain accounting boundary; no operator event spans installed here |
| Device-owned paced queues | Retain; no replacement by operator replay |
| ResourceEngine / kernel replay | Retain; not used to disguise whole-operator costs as kernel service |
| Registered actor Coordinator / PacedCoordinator | Reuse for the equal-target CPU speed comparison |
| vLLM 0.30.0 / Dynamo / AIS adapters | Retain; complete serving is outside M1 |
| Operator algebra, explicit routes/slots, profile qualification | New `sim/` layer |
| Numerical device storage | Absent in the Driver; native collection runs separately |

Actual baseline commands, in the private environment:

```sh
cmake -S /root/autodl-tmp/0z5a/fakecuda-moe/source \
  -B /root/autodl-tmp/0z5a/fakecuda-moe/baseline-build \
  -DCUDA_DRIVER_INCLUDE_DIR=/root/autodl-tmp/0z5a/fakecuda-moe/env/lib/python3.12/site-packages/nvidia/cu13/include \
  -DCUDA12_DRIVER_INCLUDE_DIR=/usr/local/cuda/include \
  -DPython3_EXECUTABLE=/root/autodl-tmp/0z5a/fakecuda-moe/env/bin/python
cmake --build /root/autodl-tmp/0z5a/fakecuda-moe/baseline-build -j4
/root/autodl-tmp/0z5a/fakecuda-moe/env/bin/python \
  /root/autodl-tmp/0z5a/fakecuda-moe/source/tests/integration/tooling/run_contracts.py \
  /root/autodl-tmp/0z5a/fakecuda-moe/baseline-build
```

Baseline: **32/32** registered contracts. Candidate uses `candidate` / `candidate-build` with the same headers/interpreter: **34/34**, including 12 operator CPU cases and the semantic Graph contract. Final contracts ran with CUDA hidden and natural process completion. Docker is absent on this rental machine, so these results are not a new no-GPU-container framework validation.

| Native dependency / artifact | Revision or SHA256 |
| --- | --- |
| Python / GCC / Driver | `3.12.3` / `11.4.0` / `580.105.08` |
| CUDA 13 build headers | private `nvidia-cuda-runtime==13.0.96` |
| PyTorch | `2.7.0+cu128`, Git `134179474539648ba7dee1317959529fbd0e7f89` |
| Triton / FLA core / einops | `3.3.0` / `0.5.2` / `0.8.1` |
| Native harness | `bf58f121c90c2e05bc2be9528fb4fb9621fe41715eff63c0e80e2b7848d497c0` |
| Installed KDA forward source | `d2d44359655433bf80cd5068191359ae8be85952c01b47d76097d254cd3bf151` |
| Loaded `libtorch_cuda.so` | `86a2c7e7ec748f87c3d1fb935d138b05b8f37a7b4d0c133b8364411516699fd9` |
| Loaded `libcublas.so.12` | `f6c022dc5fd4583164c5a964f3b470c5170eecd647820831dffe008d8441de99` |
| Loaded `libcublasLt.so.12` | `4ef47ce60325811dfce57acbc2ced3000dbb6c50ce65f4ec34340b1897d6911a` |
| Loaded real Driver | `df9183549feb062f4195e6cf130e0ef372de4a59e59dbe51554ad8c3c5b167db` |
| Backend identity digest | `c4ea0cf4179f51b71efc5a94b11489cf76b109d5ffd0e2a5c9ddf6ce40e62edd` |

The model [configuration](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/config.json) is metadata only: 27 layers, 20 KDA / 7 full-attention layers and a first dense FFN. No checkpoint was downloaded. The measured MoE is a separate declared small-width backend, not a full Kimi model. Raw manifests retain worker PID/interpreter, loaded binary hashes, routes, measurements and input digests outside git.
