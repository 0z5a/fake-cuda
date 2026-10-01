# Python integration probes

Run from the repository root. The documented entry points remain at `tests/`;
`run_no_gpu.sh` runs `pytorch/` probes in an offline, GPU-free Docker container:

```sh
bash tests/setup_uv.sh
bash tests/run_no_gpu.sh all
bash tests/run_no_gpu.sh all tensor
bash tests/run_no_gpu.sh all multidevice
bash tests/run_no_gpu.sh all memory    # allocations, H2D/D2D/D2H call paths, sync + async
bash tests/run_no_gpu.sh all streams   # multiple streams, virtual copies, event wait/query/sync
bash tests/run_no_gpu.sh all graph     # CUDAGraph virtual D2D/event capture + two replays
```

`ctest` also runs `graph_lifecycle_probe`, which checks cross-stream event dependencies, node enumeration during and after capture, and two independent executable handles replayed after the source graph is destroyed. It verifies control flow only, not device data.

The `memory` probe checks allocation identity/capacity and that PyTorch's
synchronous and asynchronous copy APIs can be submitted and synchronized; it
**cannot** check bytes, which this driver does not copy. The `streams` probe
records and waits on events around virtual copies on distinct streams, without
assuming that a Python-level timing measurement is reliable. Both probes pass
for CUDA 12.8 and 13.0 wheels in the no-GPU container.

`graph` exercises PyTorch `CUDAGraph` capture of virtual D2D and event
record/wait operations, two replays, and a cross-stream completion wait. It
passes with both CUDA 12.8 and 13.0 wheels without suppressing exceptions.
The original `cudaErrorCallRequiresNewerDriver` occurred in the Runtime's
pre-capture lazy kernel load: it needed `cuLibraryLoadData`. The Runtime then
used `cuStreamGetCaptureInfo`, `cuGraphGetNodes`, and
`cuGraphInstantiateWithFlags` along the actual capture/replay path; these
Driver entry points now have targeted ABI/semantics tests. Graphs with
allocation/free during capture or per-node kernel priorities remain outside
the supported simulator subset; no device data is copied or computed.

`tests/compare_real_gh200.py` and `tests/benchmark_real_bandwidth.py` remain
compatible command paths for the optional **real-GPU-only** scripts in `tooling/`.
They must not be used as fake-driver numerical tests.

## General graph validation

`pytorch/graph_contract.py --mode real|fake --kernel copy|add` checks a cross-stream capture and four replays on alternating launch streams. Real mode additionally verifies updated inputs and saved output ownership. Fake mode requires the no-GPU container and checks only control flow.

`tooling/benchmark_graph.py --operation copy|add` runs on real CUDA, validates values and changed-input replay, then prints an eager/Graph Markdown speed table. Its `benchmark_graph(run, reference, update_input, *, atol, rtol, repetitions=100)` function accepts typed callbacks for other operators. Adapters own their tensors and update input values in place; the runner owns warmup, capture, validation and alternating timing. Operator-specific dependencies and raw outputs stay outside the repository. See the [validation report](../../results/2026-09-29/README.md).

## Framework startup stages (not inference)

Install vLLM and SGLang in **separate** uv venvs (see `AGENTS.md` for tested
versions and environment notes). Each environment must be mountable in Docker
at the same absolute path. Run the individual probes through the existing runner:

```sh
bash tests/run_framework_no_gpu.sh vllm /absolute/path/to/vllm-venv
bash tests/run_framework_no_gpu.sh sglang /absolute/path/to/sglang-venv
# Optional: attempt engine startup with an existing LOCAL model directory:
bash tests/run_framework_no_gpu.sh vllm /absolute/path/to/vllm-venv /absolute/path/to/model
bash tests/run_framework_no_gpu.sh sglang /absolute/path/to/sglang-venv /absolute/path/to/model
```

The runner uses `--runtime=runc --network=none --pull=never`, mounts the driver,
venv and optional model read-only, disables Hugging Face hub downloads, and
limits the Docker probe to **300 seconds**. A timeout is a failure (exit 124),
not a successful startup. With no model, engine startup is **skipped**: only
Torch CUDA device discovery, framework import, and vLLM platform detection or
SGLang Engine import are checked. With a model, each probe additionally attempts
engine construction; it does **not** call generate or validate any output.
Each stage prints `STAGE`, `PASS` or `FAIL`; failures retain their traceback.

Observed previously with vLLM 0.26.0: CUDA platform detection depends on NVML,
which is outside this Driver-only project's scope. The installed Python NVML
binding cannot replace native `libnvidia-ml.so.1` or supply a virtual NVML
device count; installing `pynvml` via uv does not solve platform detection.
For **diagnosis only**, `FAKE_CUDA_VLLM_DIAGNOSE_WITHOUT_NVML=1 bash
tests/run_framework_no_gpu.sh vllm /absolute/path/to/vllm-venv` overrides
vLLM's platform detector inside the probe process to reveal the next blocker.
It does not exercise normal vLLM platform selection and must never be reported
as a successful end-to-end run. In this environment the diagnostic progresses
to importing `vllm.platforms.cuda`, which currently fails because the native
vLLM extension requires the missing Driver symbol `cuTensorMapEncodeTiled`.
This API encodes a real hardware TMA descriptor, not a virtual memcpy; exporting
a dummy that returns unconditional success would misrepresent kernel behavior. The vLLM probe reports a
non-CUDA platform as **blocked** rather than claiming an engine started; check
its logs before attributing a failure with another installation to NVML.
Observed previously with SGLang 0.5.17: `Engine` imports, but local-model
startup has not been verified. Missing dependencies, incompatible wheels, or
framework/runtime-library failures should be reported at the failing stage,
not papered over by fake Driver success. `FAKE_CUDA_TRACE=1` and
`FAKE_CUDA_TRACE_CALLS=1` can be passed to the runner for symbol and call logs.

**Numerical inference is impossible with this driver**: device allocations are
virtual addresses, no tensor bytes are stored or copied, and kernels do not
execute. Passing any import, discovery, or startup stage is not evidence that
model outputs or computations work.

## Native whole-model and CPU scheduler replay

`tooling/calibrate_serving.py --suite model --model /path/to/checkpoint --tp 2 --attention-backend auto --evidence /path/to/raw` loads the real BF16 checkpoint under vLLM 0.30.0, captures complete engine steps and generates text examples. Calibration uses batches 1/4/8, prompt lengths 32/64/128 and 64 output tokens. It freezes the step table before warming and capturing held-out batches 2/3/6, prompts 48/96/112 and 32 output tokens, with three repeats. TP ranks agree on batch membership and exact generated token IDs. The `smoke` suite is a shorter architecture check; `--load-format dummy` labels its evidence as random-weight and cannot qualify checkpoint inference.

The collector supports TP1 as well as TP2. `--kv-cache-memory-bytes` sets an explicit KV budget (default 1 GiB), independently of `--gpu-memory-utilization`'s admission fraction. vLLM skips memory profiling under this explicit budget. Actual visible-device UUIDs are recorded even when `CUDA_VISIBLE_DEVICES` remaps physical ordinals. The completed trained-checkpoint campaign uses TP1 on one H20 with 512 MiB KV; the separate TP2 architecture preflight uses random weights.

`tooling/evaluate_model_replay.py --evidence /path/to/raw --config-sha256 TRUSTED_DIGEST --bridge /path/to/resource_engine_bridge --output /path/to/raw/result.json` verifies trusted scheduler metadata and the frozen calibration table before reading validation. One independent CPU process per TP rank runs the original Scheduler/KV manager, checking exact batch membership and declared arrivals. This U1 path emits fixed-length oracle tokens; numerical token values remain native evidence. Whole-step costs already contain host and any communication work. Each child receives EOF and exits naturally; these tools do not terminate processes on a timeout.

`tooling/benchmark_semantic_process.py` compares immutable before/after source trees using complete offline processes and checks target digests. [Short-trace speed and model results](../../results/2026-10-01/topk-model-e2e.md) record the actual scopes and retained input identities.
