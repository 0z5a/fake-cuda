# Explicit operator replay

The operator layer covers gated three-matrix MoE decode, short-row top-k, recurrent KDA decode and explicit chunk prefill. Request replay and EP use separate cost/resource boundaries. Python 3.10+ runs the core without CUDA, PyTorch or NumPy. Native collection requires the pinned GPU environment in the [validation reports](../results/2026-10-01/serving-ep.md); only profile fitting needs NumPy.

```text
caller-supplied routes / slots / launch metadata
             |
       Work + Registry                KernelLaunch
             |                             |
          lowering                   owned template
             |                             |
    work / traffic / residency         existing Graph
             |                             |
    qualified stage profiles         fresh binding per invocation
             |
    serial integer clock
             |
    timeline / ledger / coverage
```

`Work` has a versioned descriptor, explicit backend revision, execution identity, dependencies and optional compiled-stage variants. `Registry` rejects duplicate IDs/keys and dangling/cyclic dependencies. `Graph` owns immutable templates and binds fresh routes/slots within declared token/state capacities. It rejects static shape, precision and layout changes. Native benchmark graphs are captured separately for each fixed route/shape; the binding tests do not imply that an arbitrary numerical framework graph can change expert-loop topology.

`lower()` is pure. MoE retains actual counts, active experts and per-expert row padding; shared-expert work is explicit. KDA separates SIMT state updates from TC projections, counts live sequences and allocated state capacity, and does not scale recurrent work with history length. `KDAPrefill` retains chunk lengths/size and convolution width in distinct stage families; its executed recurrence FLOPs are unknown, so an analytic decode multiplier cannot supply chunk latency. Grouped-value heads remain unsupported. `permutation()`, `combine()` and `kda_reference()` are bounded CPU references for validation, separate from fake device execution.

Logical work, executed padding, logical access bytes and known resident weights/state occupy separate ledgers. DRAM traffic, workspace and graph-buffer capacity remain null until measured/supplied; known-weight totals are not a serving memory budget. The analytic provider requires declared rates and an explicit cold-traffic assumption before using logical accesses as DRAM demand. Selection and unknown auxiliary algorithms still need calibration.

Profiles match hardware/driver/backend revision, mode, cache policy, stage, precision/layout and declared compiled variants. Exact feature samples and interpolation have separate qualification. One-dimensional interpolation stays between measured anchors; multidimensional fits use nonnegative coefficients within frozen feature bounds. Unqualified, mismatched and out-of-domain queries remain unsupported. Fixed-slot KDA calibration does not cover rotating-slot traces.

The default clock serializes all device stages while preserving stream sequence, dependencies and host releases. Fusion requires one explicitly supplied physical-group cost, charged once to `fused_shared`; logical components retain their own work ledgers without receiving invented duration shares. `counterfactual()` reruns the dependency/release schedule, so host gaps can absorb saved work.

Create a JSON array from `Work.dumps()` **outside git**, then replay:

```sh
python -m sim /absolute/path/trace.json \
  --profiles /absolute/path/qualified.profiles \
  --output /absolute/path/replay.md
```

`--releases` supplies an external `op_id -> integer ns` mapping. Strict mode fails unknown costs. Analytic mode needs `--rates TC_FLOP_S SIMT_FLOP_S BYTE_S`; `--logical-cold-traffic` declares an unvalidated traffic assumption. Compat mode labels every 10 ms fallback; those reports are not calibrated performance evidence. Incomplete costs never become zero or a complete makespan.

For native collection, use separate output directories for calibration and validation. The benchmark checks numerical references, warms each case, captures fixed-address graphs, records device-event intervals and host submission separately, and fingerprints loaded Driver/Torch/cuBLAS binaries. Routing launch diagnostics run before warmup; their timestamps are not calibration samples.

```sh
python tests/integration/tooling/benchmark_semantic.py \
  --family moe --split calibration --batches 1,4,8,16,17,32,64,128 \
  --output /absolute/path/calibration
python tests/integration/tooling/evaluate_semantic.py freeze \
  --input /absolute/path/calibration/*.json --output /absolute/path/frozen.profiles
# Collect whole shapes/routing families not used for fitting, after freezing.
python tests/integration/tooling/benchmark_semantic.py \
  --family moe --split validation --batches 3,10,28,56,112 \
  --output /absolute/path/validation
python tests/integration/tooling/evaluate_semantic.py evaluate \
  --profiles /absolute/path/frozen.profiles --input /absolute/path/validation/*.json \
  --output /absolute/path/accuracy.md --qualified-output /absolute/path/qualified.profiles
```

Defaults are 30 warmups, 200 measurements and five independent process sessions. `--family prefill --sequence-length 64` selects the independently calibrated chunk path. Keep raw sessions, profiles, logs and compiled artifacts outside git; commit Markdown evidence. Requalification changes the accepted-domain flag only, preserving the frozen parameters and input hashes.

## Request admission and expert dispatch

`serving.run(requests, policy, devices, cost)` implements FCFS admission, decode priority, bounded prefill chunks and continuous admission. `DeviceMemory` lists every per-card resident weight/state/conv pool and page/buffer budget. A fixed KV pool is charged once; dynamic pages otherwise reserve the final context before admission. Cancellation retains resources until submitted work finishes. Token visibility, physical prompt progress and discarded tokens remain separate. ITL for one output is null; request P95/P99 need 20/100 samples. Metrics include waiting, capacity, rejected/completed/cancelled counts and a full-drain observation window.

Costs must declare `whole_step_including_host_and_communication`. The frozen vLLM `UniformStepCost` adapter supports homogeneous decode and fresh unchunked prefill, and rejects continued chunks or heterogeneous contexts. Synthetic chunk policies have explicit contract costs. Kernel-stage profiles cannot silently become full-model serving costs.

`ep.ledger()` retains assignment/unique-token matrices, explicit placement and independently declared dispatch/combine protocols. `ep.run()` lowers chunk dependencies, per-card DRAM/SM/buffer inputs and physical links into the existing C++ `ResourceEngine` through `resources.Engine`. Held chunk buffers survive blocked intermediate phases until combine finishes. Shared-link capacity and generation-checked remaining work constrain pipeline progress. Declared topology results remain analytical unless all capacities and phase inputs have their own qualification.

`resources.profile_phase()` embeds an independently qualified discrete combination with matching identity and shape bounds. Reserve an exclusive combination lane and charge the measured duration once, with no further bandwidth division or standalone addition. `benchmark_overlap.py` collects standalone copy, standalone compute and paired event windows; it freezes before independent holdouts. The ratio of summed standalone windows to paired time is not a measured serialized-arm speedup.

[P5/P6 contracts and H20 validation](../results/2026-10-01/serving-ep.md) distinguish declared memory/topology scenarios from native calibration. Existing vLLM/Dynamo/AIS and live Driver scheduling remain separate. Broader full-model chunk/heterogeneous costs, real DeepEP topology demands, historical-token selectors and complete production attention fractions need their own evidence.
