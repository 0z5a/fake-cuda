"""Replay frozen native costs with independent original Scheduler processes."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.vllm.costs import Shape, StepModel, StepSample, UnsupportedStep
from adapters.vllm.ranks import RankGroup
from adapters.vllm.runner import Workload, run
from sweep_serving import tails


def condition(case: dict) -> tuple[str, ...] | None:
    keys = ("background_jobs_before", "background_jobs_after")
    if all(key not in case for key in keys):
        return None  # Legacy evidence has no observed load qualification.
    if any(key not in case for key in keys):
        raise UnsupportedStep("incomplete background observations")
    before, after = (tuple(sorted(case[key])) for key in keys)
    if before != after:
        raise UnsupportedStep("background changed during workload")
    return before


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--fresh-ranks", action="store_true")
    parser.add_argument("--limit", type=int, help="Bound speed experiments to the first N retained cases")
    args = parser.parse_args()
    if version("vllm") != "0.30.0":
        parser.error("qualified scheduler is vLLM 0.30.0")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    started = time.perf_counter_ns()
    root = args.evidence
    data = (root / "scheduler-config.pkl").read_bytes()
    if hashlib.sha256(data).hexdigest() != args.config_sha256:
        parser.error("trusted configuration digest mismatch")
    config, kv, block, hashed = pickle.loads(data)
    tp = config.parallel_config.tensor_parallel_size
    calibration = json.loads((root / "calibration.json").read_text())
    samples: dict[tuple[str, ...] | None, list[StepSample]] = {}
    rejected_calibration = 0
    for case in calibration:
        try:
            key = condition(case)
        except UnsupportedStep:
            rejected_calibration += 1
            continue
        samples.setdefault(key, []).extend(StepSample(Shape(**step["shape"]), step["duration_ns"])
                                            for step in case["steps"])
    models = {key: StepModel(values) for key, values in samples.items()}
    # Fit before opening the held-out measurements; no validation refitting.
    validation = json.loads((root / "validation.json").read_text())
    if args.limit is not None:
        validation = validation[:args.limit]
    configurations = [pickle.loads(data) for _ in range(tp)]
    group = None if args.fresh_ranks else RankGroup(configurations, processes=True)
    rows = []
    digest = hashlib.sha256()
    coverage = [0, 0., 0., 0.]
    unsupported = []
    total_steps = 0
    try:
        for case in validation:
            total_steps += len(case["steps"])
            try:
                model = models.get(condition(case))
                if model is None:
                    raise UnsupportedStep("uncalibrated background condition")
                predictions = [model.predict(Shape(**step["shape"])) for step in case["steps"]]
            except UnsupportedStep as error:
                unsupported.append({"workload": case["workload"]["name"], "max_sequences": case["max_sequences"],
                                    "reason": str(error), "steps": len(case["steps"])})
                continue
            workload = Workload(**{**case["workload"], "arrivals_ns": tuple(case["workload"]["arrivals_ns"])})
            for configuration, kv, block, hashed in configurations:
                configuration.scheduler_config.max_num_seqs = case["max_sequences"]
            for step, predicted in zip(case["steps"], predictions):
                observed = step["duration_ns"]
                coverage[0] += 1
                coverage[1] += abs(predicted - observed)
                coverage[2] += observed
                coverage[3] += abs(predicted / observed - 1)
            begin = time.perf_counter_ns()
            workers = group or RankGroup(configurations, processes=True)
            try:
                result = run(workload, configurations, [model] * tp, args.bridge, ranks=workers)
            finally:
                if group is None:
                    workers.close()
            wall = (time.perf_counter_ns() - begin) / 1e9
            local = run(workload, [pickle.loads(pickle.dumps(c)) for c in configurations], [model] * tp, args.bridge)
            semantic = (result.finish_ns, result.tokens, result.batch_ids, result.arrivals_observed_ns, result.control_reads)
            assert semantic == (local.finish_ns, local.tokens, local.batch_ids, local.arrivals_observed_ns, local.control_reads)
            assert len(result.rank_pids) == tp
            digest.update(json.dumps(semantic, sort_keys=True).encode())
            score = result.metrics(workload)
            predicted_metrics = [score[key] for key in ("finish_ns", "tokens_per_second", "ttft_median_ms", "itl_median_ms")]
            actual = [case["metrics"][key] for key in ("finish_ns", "tokens_per_second", "ttft_median_ms", "itl_median_ms")]
            errors = [100 * (p / a - 1) for p, a in zip(predicted_metrics, actual)]
            errors.extend(100 * (p / a - 1) for p, a in zip(tails(result.tokens, workload), tails(case["tokens"], workload)))
            rows.append({"workload": workload.name, "max_sequences": case["max_sequences"],
                         "native_s": case["metrics"]["finish_ns"] / 1e9, "cpu_s": wall,
                         "errors_percent": errors, "metrics": score, "ipc_messages": result.ipc_messages})
            print(f"scored {workload.name}, max seq={case['max_sequences']}", file=sys.stderr, flush=True)
    finally:
        if group is not None:
            group.close()
    print(json.dumps({"tp": tp, "fresh_ranks": args.fresh_ranks, "rows": rows,
        "covered_steps": int(coverage[0]), "total_steps": total_steps, "unsupported": unsupported,
        "rejected_calibration_runs": rejected_calibration, "calibrated_conditions": len(models),
        "legacy_calibration": None in models,
        "step_wape_percent": 100 * coverage[1] / coverage[2] if coverage[2] else None,
        "step_mape_percent": 100 * coverage[3] / coverage[0] if coverage[0] else None, "semantic_sha256": digest.hexdigest(),
        "campaign_s": (time.perf_counter_ns() - started) / 1e9}))


if __name__ == "__main__":
    main()
