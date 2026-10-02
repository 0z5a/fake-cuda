"""Validate explicit requests against retained native steps and declared capacities."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.vllm.costs import Shape, StepModel, StepSample
from adapters.vllm.explicit import UniformStepCost
from sim.serving import DeviceMemory, PageLayout, Policy, Request, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    calibration, validation = args.calibration.read_bytes(), args.validation.read_bytes()
    source = hashlib.sha256(calibration).hexdigest()
    model = StepModel([StepSample(Shape(**step["shape"]), step["duration_ns"])
                       for case in json.loads(calibration) for step in case["steps"]])
    frozen = json.dumps(model.table, sort_keys=True)
    costs = UniformStepCost(model, source)
    devices = []
    for entry in json.loads(args.memory.read_text()):
        devices.append(DeviceMemory(**{**entry, "pages": tuple(PageLayout(**page) for page in entry.get("pages", []))}))
    records = []
    for case in json.loads(validation):
        w = case["workload"]
        requests = tuple(Request(str(i), arrival, w["prompt"], w["output"]) for i, arrival in enumerate(w["arrivals_ns"]))
        report = run(requests, tuple(devices), Policy(case["max_sequences"], 4096), costs)
        if [[item.request_id for item in step.batch.items] for step in report.batches] != [step["ids"] for step in case["steps"]]:
            raise ValueError("explicit request batches differ from held-out native scheduling")
        records.append({"workload": w["name"], "native_ns": case["metrics"]["finish_ns"],
                        "predicted_ns": report.finish_ns, "metrics": report.metrics()})
    scenarios = []
    for slots in (4, 8):
        cards = tuple(replace(device, state_slots=slots) for device in devices)
        requests = tuple(Request(str(i), 0, 96, 32) for i in range(32))
        started = time.perf_counter_ns()
        report = run(requests, cards, Policy(8, 4096), costs)
        elapsed = time.perf_counter_ns() - started
        scenarios.append({"slots": slots, "metrics": report.metrics(), "cpu_replay_ns": elapsed})
    assert json.dumps(model.table, sort_keys=True) == frozen
    errors = [abs(record["predicted_ns"] / record["native_ns"] - 1) for record in records]
    raw = {"calibration_sha256": source, "validation_sha256": hashlib.sha256(validation).hexdigest(),
           "frozen_table_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
           "memory_sha256": hashlib.sha256(args.memory.read_bytes()).hexdigest(),
           "memory": [asdict(device) for device in devices], "records": records, "scenarios": scenarios,
           "scope": "explicit_request_replay_with_declared_memory; native_costs_fixed_length_oracle"}
    args.raw.write_text(json.dumps(raw, sort_keys=True))
    lines = ["# Explicit request and capacity replay", "",
             f"Frozen calibration SHA256: `{source}`. {len(records)} retained held-out native cases have identical batch membership; median absolute whole-workload error **{statistics.median(errors)*100:.3f}%**.", "",
             "The following is a synthetic capacity counterfactual with 32 simultaneous requests, prompt 96 and output 32. Each physical device uses explicitly declared resident weights, state/conv pool, page geometry and buffer budgets. These memory budgets are scenario inputs, not the native cache allocator's measured layout. Costs come from the frozen whole-model table; these are predictions, not a new native throughput measurement. Observation includes drain, with no warmup. Request-level P99 is undefined with 32 samples.", "",
             "| State slots/card | Completed / tokens | Predicted drain s | Predicted tokens/s | TTFT P50 / P95 s | ITL P50 / P95 ms | Peak GiB/card | Throughput ratio |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    baseline = scenarios[0]["metrics"]["tokens_per_second"]
    for scenario in scenarios:
        m = scenario["metrics"]
        ttft, itl = m["ttft"], m["itl"]
        peaks = ", ".join(f"{n/1024**3:.3f}" for n in m["peak_bytes"].values())
        lines.append(f"| {scenario['slots']} | {m['completed']} / {m['emitted_tokens']} | {m['finish_ns']/1e9:.6f} | {m['tokens_per_second']:.3f} | {ttft['p50_ns']/1e9:.6f} / {ttft['p95_ns']/1e9:.6f} | {itl['p50_ns']/1e6:.6f} / {itl['p95_ns']/1e6:.6f} | {peaks} | {m['tokens_per_second']/baseline:.3f}x |")
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
