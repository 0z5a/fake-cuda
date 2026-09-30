# Controlled serving adapters

The qualified path is **U1**: original vLLM 0.30.0 `Scheduler` and `KVCacheManager`, an explicit fixed-length token oracle, and registered actors on a common clock. Worker execution and completion delivery are adapted. It is not an unmodified inference engine, a numerical simulator, or a replacement for arbitrary Python/Go/OS clocks.

`common/` contains the coordinator, control provenance, collective matcher and separate physical/cache-visibility ledgers. `vllm/` owns original scheduling and the common endpoint event loop. `llmd_epp/` embeds llm-d's native queue profile; `dynamo/` embeds its native Rust selection/reservation/load core. `aisimulate/` embeds native whole-step regression in a separate process. The routing policies remain independent.

| Component | Pinned version | Boundary |
|---|---|---|
| vLLM | 0.30.0 | CPU original scheduler/KV manager; synchronous eager, no prefix/chunked prefill/speculation |
| llm-d Router | v0.10.0, `71f4f0999f95b96c49a9d0c4afbd18dfdb943c26` | Native enabled-label filter, queue scorer weight 1, max-score picker; native tie behavior |
| Dynamo runtime | 1.5.0 | Native eligibility, selection, reservation and potential-load tracker; KV events/reuse disabled explicitly |
| AISimulate | 0.12.0 | Native regression over independent observed whole-step samples; bounded recorded shape domain |

Each synchronous step cost includes host/communication inside `LLMEngine.step()`. Do not add a kernel or collective estimate to it. The Driver's kernel-only `PerformanceModel` remains a separate API and rejects whole-forward costs. Request admission/loop overhead outside the measured step remains unmodeled. Predictor and IPC wall time affect simulator speed only.

The AIS identity contains model-config and hardware-profile hashes, backend/version, dtype, attention backend, graph mode, TP/PP/attention-DP/CP/MoE-TP/MoE-EP, KV block size and accounting scope. Regression fitting consumes seconds; the native estimate returns milliseconds; the adapter returns nanoseconds. Homogeneous prefill/decode metrics use complete scheduled/queued FPM fields. Heterogeneous or mixed steps, extrapolation, unsupported layouts and absent estimates fail explicitly. The installed native op-level database rejects vLLM 0.30.0 on `a100_sxm`; it must not be relabeled as 0.14.0. Regression is an explicitly chosen mode.

## Reproduce the controlled campaign

Keep the original vLLM environment unchanged. Install AIS/Dynamo only into an isolated task environment when authorized. For the native EPP executable, place `main.go` under `cmd/fakecuda-epp/` in the pinned llm-d Router checkout and build that package with its upstream `go.mod`/`go.sum`. This uses the real upstream policies rather than rewriting their scores.

Use task-owned model weights and a separate evidence directory. The collector needs a real GPU; it leaves the original scheduler and model worker in use and records actual numerical inference steps before later CPU replay. It pins single-process vLLM execution so owned processes exit naturally. The scheduler pickle is trusted configuration only: loaded attention modules are removed before serialization.

```sh
CUDA_VISIBLE_DEVICES=0 /path/to/vllm-python tests/integration/tooling/calibrate_serving.py \
  --model /path/to/Qwen2.5-0.5B --evidence /path/to/evidence
```

Retain the matching native `device-0.profile` in that directory and verify the configuration digest before unpickling it. The collector finishes the 24-run calibration file before starting the independent 18-run validation file; evaluators freeze predictors before reading validation durations.

```sh
/path/to/vllm-python tests/integration/tooling/sweep_serving.py \
  --evidence /path/to/evidence --config-sha256 TRUSTED_SHA256 \
  --bridge /path/to/resource_engine_bridge \
  --baseline-bridge /path/to/previous_resource_engine_bridge \
  --ais-python /path/to/ais-python --condition 'Observed native GPU/load condition'

/path/to/vllm-python tests/integration/tooling/benchmark_serving.py \
  --evidence /path/to/evidence --config-sha256 TRUSTED_SHA256 \
  --baseline /path/to/previous_resource_engine_bridge \
  --candidate /path/to/resource_engine_bridge
```

The first comparison scores duration, throughput, median/P95 TTFT and ITL, unsupported coverage and regret only among measured max-sequence settings. The second alternates fresh-process order across three repeats and checks the digest of every target token/batch/arrival timeline, including imports, fit and natural teardown in wall time.

## Contracts

The ordinary CMake contracts require no framework packages. On machines where processes must never be killed, run their registered commands without CTest's timeout machinery:

```sh
python tests/integration/tooling/run_contracts.py /path/to/build
```

The following integration tools use the same `--config`, `--config-sha256` and `--bridge` arguments:

| Tool | Additional argument | Checks |
|---|---|---|
| `vllm_scheduler_contract.py` | None | Single/dual rank, fixed-token producer reads, exact arrivals, slow-query invariance, paced/coordinated parity, wrong-scope rejection |
| `kv_budget_contract.py` | None | Explicit Graph reservation changes layout-derived original KV block admission |
| `epp_closed_loop_contract.py` | `--epp /path/to/native-epp` | Delayed queue snapshots, native filter/scores/picker and service-sensitive routes |
| `dynamo_closed_loop_contract.py` | `--dynamo-python /path/to/ais-python` | Native bookings, delayed prefill/completion feedback, service-sensitive routes, zero final native loads |

Run `ais_contract.py` and `dynamo_contract.py` with the isolated AIS/Dynamo Python. The latter covers failed partial P/D selection, separate P/D release, cancel, late/duplicate callbacks, virtual lease expiry and booking-free snapshot/shadow decisions. Virtual leases belong to this adapter; they are not a claim that Dynamo DEP reservation timers have been virtualized. Releasing a reservation does not stop physical work or evict physical KV.

Graph/KV budget tests use an explicit shared budget; they do not infer native graph-pool size or reproduce automatic GPU memory profiling. Cache snapshots enforce generation/sequence/visibility separately, but the qualified native EPP profile uses queue depth and does not claim live prefix-event integration. Arbitrary NCCL/NIXL/RDMA paths, multi-node discovery and whole-framework U0 execution remain outside this path.

[Prediction/configuration results](../results/2026-09-30/serving-validation.md), [complete-process speed comparison](../results/2026-09-30/serving-speed.md), [implementation and validation matrix](../results/2026-09-30/execution-validation.md).
