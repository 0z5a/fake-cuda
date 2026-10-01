"""Replay a native checkpoint campaign with the original vLLM scheduler."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.vllm.costs import Shape, StepModel, StepSample
from adapters.vllm.ranks import RankGroup
from adapters.vllm.runner import Workload, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True, help="digest of trusted native scheduler metadata")
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    native = json.loads((root / "native-run.json").read_text())
    if not native["checkpoint"] or native["vllm"] != "0.30.0":
        parser.error("requires a real checkpoint campaign from the qualified vLLM version")
    data = (root / "scheduler-config.pkl").read_bytes()
    if hashlib.sha256(data).hexdigest() != args.config_sha256:
        parser.error("trusted scheduler configuration digest mismatch")
    calibration_data = (root / "calibration.json").read_bytes()
    samples = [StepSample(Shape(**step["shape"]), step["duration_ns"])
               for case in json.loads(calibration_data) for step in case["steps"]]
    model = StepModel(samples)
    frozen = json.dumps(model.table, sort_keys=True)
    if frozen != (root / "frozen-step-table.json").read_text():
        raise ValueError("calibration differs from the table frozen before native holdout capture")
    # Shape medians and interpolation are fixed before opening holdout data.
    validation = json.loads((root / "validation.json").read_text())
    records = []
    step_errors = []
    absolute_step_error_ns = 0
    native_step_ns = 0
    target = hashlib.sha256()
    configurations = [pickle.loads(data) for _ in range(native["tp"])]
    ranks = RankGroup(configurations, processes=True)
    try:
        for case in validation:
            workload = Workload(**{**case["workload"], "arrivals_ns": tuple(case["workload"]["arrivals_ns"])})
            for config in configurations:
                config[0].scheduler_config.max_num_seqs = case["max_sequences"]
            for step in case["steps"]:
                prediction = model.predict(Shape(**step["shape"]))
                step_errors.append(abs(prediction / step["duration_ns"] - 1))
                absolute_step_error_ns += abs(prediction - step["duration_ns"])
                native_step_ns += step["duration_ns"]
            started = time.perf_counter_ns()
            result = run(workload, configurations, [model] * native["tp"], args.bridge, ranks=ranks)
            wall_ns = time.perf_counter_ns() - started
            if result.batch_ids != [step["ids"] for step in case["steps"]]:
                raise ValueError("original scheduler replay changed native batch membership")
            if result.arrivals_observed_ns != list(workload.arrivals_ns):
                raise ValueError("declared request arrival was skipped")
            if len(result.rank_pids) != native["tp"] or os.getpid() in result.rank_pids:
                raise ValueError("replay requires independent original-scheduler rank processes")
            actual, predicted = case["metrics"], result.metrics(workload)
            target.update(json.dumps((result.finish_ns, result.tokens, result.batch_ids), sort_keys=True).encode())
            records.append({"workload": workload.name, "native_ns": actual["finish_ns"], "cpu_ns": wall_ns,
                            "predicted_ns": result.finish_ns,
                            "duration_error_percent": 100 * (predicted["finish_ns"] / actual["finish_ns"] - 1),
                            "throughput_error_percent": 100 * (predicted["tokens_per_second"] / actual["tokens_per_second"] - 1),
                            "native_tokens_s": actual["tokens_per_second"], "predicted_tokens_s": predicted["tokens_per_second"],
                            "ipc_messages": result.ipc_messages,
                            "rank_pids": result.rank_pids,
                            "background_jobs": case["background_jobs_before"] + case["background_jobs_after"]})
            print(workload.name, f"native={actual['finish_ns']/1e9:.3f}s CPU={wall_ns/1e9:.3f}s", flush=True)
    finally:
        ranks.close()
    assert json.dumps(model.table, sort_keys=True) == frozen
    output = {"records": records, "target_sha256": target.hexdigest(),
              "calibration_sha256": hashlib.sha256(calibration_data).hexdigest(),
              "frozen_table_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
              "config_sha256": args.config_sha256, "covered_steps": len(step_errors), "unknown_steps": 0,
              "step_p50_error_percent": 100 * statistics.median(step_errors),
              "step_p95_error_percent": 100 * sorted(step_errors)[math.ceil(.95 * len(step_errors)) - 1],
              "step_mape_percent": 100 * statistics.mean(step_errors),
              "step_wape_percent": 100 * absolute_step_error_ns / native_step_ns,
              "scope": "whole_model_serving_fixed_length_token_oracle",
              "numerical_tokens": "retained_in_native_campaign_only"}
    args.output.write_text(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
