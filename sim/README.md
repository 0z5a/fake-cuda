# Explicit operator replay

M1 covers a single-card operator trace: gated three-matrix MoE decode, short-row top-k and recurrent KDA decode. Python 3.10+ runs the core without CUDA, PyTorch or NumPy. Native collection requires the pinned GPU environment in the [validation report](../results/2026-10-01/moe-serving.md); only profile fitting needs NumPy.

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

`lower()` is pure. MoE retains actual counts, active experts and per-expert row padding; shared-expert work is explicit. KDA separates SIMT state updates from TC projections, counts live sequences and allocated state capacity, and does not scale recurrent work with history length. Grouped-value heads and prefill are rejected. `permutation()`, `combine()` and `kda_reference()` are bounded CPU references for validation, separate from fake device execution.

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

Defaults are 30 warmups, 200 measurements and five independent process sessions. Keep raw sessions, profiles, logs and compiled artifacts outside git; commit Markdown evidence. Requalification changes the accepted-domain flag only, preserving the frozen parameters and input hashes. Complete serving, EP, shared-bandwidth overlap, historical-token selectors and full-model attention fractions are later milestones. Existing vLLM/Dynamo/AIS and Driver scheduling remain separate.
