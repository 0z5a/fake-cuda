"""Score frozen whole-step predictors through the original vLLM 0.30 scheduler."""
import time
PROCESS_START = time.perf_counter_ns()

import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import pickle
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.aisimulate.client import AisStepCost
from adapters.aisimulate.identity import Identity
from adapters.vllm.costs import Shape, StepModel, StepSample, UnsupportedStep
from adapters.vllm.runner import Workload, run


def percentile(values: list[int], quantile: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * quantile
    left = math.floor(position)
    right = math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def tails(tokens: dict[str, list[int]], workload: Workload) -> tuple[float, float]:
    first = [tokens[str(i)][0] - arrival for i, arrival in enumerate(workload.arrivals_ns)]
    intervals = [b - a for times in tokens.values() for a, b in zip(times, times[1:])]
    return percentile(first, .95), percentile(intervals, .95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--baseline-bridge", type=Path, required=True)
    parser.add_argument("--ais-python", type=Path, required=True)
    parser.add_argument("--condition", required=True, help="native collection's observed hardware/load condition")
    args = parser.parse_args()
    if version("vllm") != "0.30.0":
        parser.error("qualified backend is vLLM 0.30.0")
    root = args.evidence
    data = (root / "scheduler-config.pkl").read_bytes()
    if hashlib.sha256(data).hexdigest() != args.config_sha256:
        parser.error("trusted scheduler configuration digest mismatch")
    config, kv, block, hashed = pickle.loads(data)
    samples = [StepSample(Shape(**step["shape"]), step["duration_ns"])
               for case in json.loads((root / "calibration.json").read_text()) for step in case["steps"]]
    identity = Identity(
        hashlib.sha256(json.dumps(config.model_config.hf_config.to_dict(), sort_keys=True).encode()).hexdigest(),
        hashlib.sha256((root / "device-0.profile").read_bytes()).hexdigest(),
        "vllm", "0.30.0", "bfloat16", "TRITON_ATTN", "eager", 1, 1, 1, 1, 1, 1, block,
        "whole_step_including_host_and_communication")
    measured = StepModel(samples)
    ais = AisStepCost(args.ais_python, root / "calibration.json", identity)
    # Complete both fits before reading independent validation observations.
    ais.predict(Shape("prefill", 8, 64))
    models = {"measured-step": measured, "AIS regression": ais}
    trials: dict[tuple[str, str, int], list[tuple[float, ...]]] = defaultdict(list)
    coverage = {name: [0, 0, 0., 0., 0.] for name in models}
    chosen: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    actuals: dict[tuple[str, int], list[float]] = defaultdict(list)
    hits = {name: [0] * 6 for name in models}
    complete = {name: 0 for name in models}
    parity = 0
    baseline_total = 0
    optimized_total = 0
    try:
        validation = json.loads((root / "validation.json").read_text())
        for case in validation:
            workload = Workload(**{**case["workload"], "arrivals_ns": tuple(case["workload"]["arrivals_ns"])})
            sequences = case["max_sequences"]
            actual = case["metrics"]
            actuals[(workload.name, sequences)].append(actual["tokens_per_second"])
            configurations = []
            for _ in range(3):
                cfg = pickle.loads(data)
                cfg[0].scheduler_config.max_num_seqs = sequences
                configurations.append(cfg)
            started = time.perf_counter_ns()
            baseline = run(workload, [configurations[0]], [measured], args.baseline_bridge)
            baseline_wall = time.perf_counter_ns() - started
            baseline_total += baseline_wall
            real_tails = tails(case["tokens"], workload)
            for index, (name, model) in enumerate(models.items(), 1):
                counts = coverage[name]
                for step in case["steps"]:
                    counts[1] += 1
                    try:
                        prediction = model.predict(Shape(**step["shape"]))
                    except UnsupportedStep:
                        continue
                    observation = step["duration_ns"]
                    counts[0] += 1
                    counts[2] += abs(prediction - observation)
                    counts[3] += observation
                    counts[4] += abs(prediction / observation - 1)
                started = time.perf_counter_ns()
                try:
                    result = run(workload, [configurations[index]], [model], args.bridge)
                except UnsupportedStep as error:
                    print(f"Unsupported workload: {name}, {workload.name}, sequences={sequences}: {error}", flush=True)
                    continue
                wall = time.perf_counter_ns() - started
                score = result.metrics(workload)
                if result.arrivals_observed_ns != list(workload.arrivals_ns):
                    raise ValueError("coordinator skipped a declared arrival")
                if name == "measured-step":
                    if (result.finish_ns, result.tokens, result.batch_ids) != (baseline.finish_ns, baseline.tokens, baseline.batch_ids):
                        raise ValueError("resource optimization changed the target timeline")
                    parity += 1
                    optimized_total += wall
                predicted_tails = tails(result.tokens, workload)
                pairs = [(score[key], actual[key]) for key in
                         ("finish_ns", "tokens_per_second", "ttft_median_ms", "itl_median_ms")]
                pairs.extend(zip(predicted_tails, real_tails))
                errors = tuple(100 * (a / b - 1) for a, b in pairs)
                trials[(name, workload.name, sequences)].append(
                    (actual["finish_ns"] / 1e9, baseline_wall / 1e9, wall / 1e9,
                     actual["finish_ns"] / wall, baseline_wall / wall, *errors))
                chosen[(name, workload.name, sequences)].append(score["tokens_per_second"])
                complete[name] += 1
                for i, error in enumerate(errors):
                    hits[name][i] += abs(error) <= 10
            print(f"scored {workload.name} sequences={sequences}", file=sys.stderr, flush=True)

        print("# A100 serving prediction and configuration sweep\n")
        print("vLLM 0.30.0; Qwen2.5-0.5B bf16, TP1, eager TRITON_ATTN; P64/O64; fixed-length token oracle. " + args.condition + " CPU replay initializes no CUDA. This is U1: original Scheduler/KV manager with adapted worker completion and an explicit coordinator. Whole-step costs include host and communication; neither is billed again.\n")
        print("Calibration: 24 separate runs, batch 1/8/16/32, prompt 32/64/128, output 96. Both predictors were frozen before reading 18 validation runs: three repeats of each workload and max-sequence setting. AISimulate 0.12.0 uses its native regression explicitly, not an op-level database for a different backend version.\n")
        print("| Predictor | Workload | Max seq | Native s | CPU s | Reuse speedup | Duration error | Throughput error | TTFT median error | ITL median error | TTFT P95 error | ITL P95 error |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for (name, workload, sequences), rows in trials.items():
            med = [statistics.median(column) for column in zip(*rows)]
            print(f"| {name} | {workload} | {sequences} | {med[0]:.3f} | {med[2]:.3f} | {med[3]:.2f}× | " + " | ".join(f"{error:+.2f}%" for error in med[5:]) + " |")
        print("\nCPU workload wall time includes original scheduling, oracle, predictor and bridge IPC/startup/teardown. Reuse speedup excludes prior downloads/calibration and process imports; it is a timing simulation speedup, not numerical inference acceleration. Reported rows are medians of three runs. Tail percentiles use linear interpolation of all request TTFTs and token intervals in each run; three repeats under shared host load do not establish confidence bounds.\n")
        print("| Predictor | Held-out step coverage | Conditional WAPE | Conditional MAPE | Completed workloads | ≤10% duration / throughput / TTFT / ITL / P95 TTFT / P95 ITL |\n|---|---:|---:|---:|---:|---|")
        for name, (covered, total, absolute, observed, percentage) in coverage.items():
            print(f"| {name} | {int(covered)}/{int(total)} | {100*absolute/observed:.2f}% | {100*percentage/covered:.2f}% | {complete[name]}/{len(validation)} | " + " / ".join(f"{hit}/{complete[name]}" for hit in hits[name]) + " |")
        print("\n| Workload | Previous core CPU s | Optimized core CPU s | End-to-end simulator speedup |\n|---|---:|---:|---:|")
        for (name, workload, sequences), rows in trials.items():
            if name == "measured-step":
                med = [statistics.median(column) for column in zip(*rows)]
                print(f"| {workload}, max seq {sequences} | {med[1]:.3f} | {med[2]:.3f} | {med[4]:.2f}× |")
        print(f"\nAll {parity}/{len(validation)} baseline/optimized runs have identical virtual token timestamps and batch membership. Core-only history scaling is reported separately.\n")
        print(f"Sum of all {len(validation)} complete serving-run wall times: previous core {baseline_total/1e9:.3f} s, optimized {optimized_total/1e9:.3f} s, {baseline_total/optimized_total:.2f}× ({100*(1-optimized_total/baseline_total):.2f}% less wall time). Imports, fitting and independent scoring are excluded from this identical-work comparison.\n")
        print("| Predictor | Workload | Chosen max seq | Measured best max seq | Throughput regret |\n|---|---|---:|---:|---:|")
        for name in models:
            for workload in sorted({w for w, seq in actuals}):
                candidates = [seq for w, seq in actuals if w == workload]
                if any((name, workload, seq) not in chosen for seq in candidates):
                    print(f"| {name} | {workload} | unsupported | — | — |")
                    continue
                selection = max(candidates, key=lambda seq: statistics.median(chosen[(name, workload, seq)]))
                best = max(candidates, key=lambda seq: statistics.median(actuals[(workload, seq)]))
                regret = 100 * (1 - statistics.median(actuals[(workload, selection)]) / statistics.median(actuals[(workload, best)]))
                print(f"| {name} | {workload} | {selection} | {best} | {regret:.2f}% |")
        print("\nRegret is restricted to the three independently measured settings, 8/16/32; no optimum is inferred for unmeasured configurations. Native request admission runs at step boundaries while the simulator delivers declared arrivals exactly; native add-request overhead is outside the step-cost scope. Neither discrepancy is fitted on validation data.\n")
        print("| Retained input | SHA-256 |\n|---|---|")
        for name in ("scheduler-config.pkl", "calibration.json", "validation.json", "device-0.profile"):
            print(f"| {name} | `{hashlib.sha256((root/name).read_bytes()).hexdigest()}` |")
        print(f"| Canonical model configuration | `{identity.model_config_sha256}` |")
    finally:
        ais.close()
    print(f"\nComplete comparison process wall: {(time.perf_counter_ns()-PROCESS_START)/1e9:.3f} s, including imports, both fits, all baseline/comparator runs and scoring.")


if __name__ == "__main__":
    main()
