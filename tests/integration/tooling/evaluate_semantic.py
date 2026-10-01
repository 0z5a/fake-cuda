"""Freeze calibration before collecting disjoint whole-shape validation sessions."""
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.cost import CalibrationStore, Identity, Profile
from sim.models import lower
from sim.replay import replay
from sim.serde import read_profiles
from sim.work import KDA, Registry, Work


def nonnegative_fit(features: list[tuple[int, ...]], durations: list[int]) -> tuple[float, ...]:
    import numpy as np
    design = np.array([(1., *row) for row in features])
    values = np.array(durations)
    best = float("inf"), np.zeros(design.shape[1])
    for count in range(1, design.shape[1] + 1):
        for columns in itertools.combinations(range(design.shape[1]), count):
            selected = design[:, columns]
            coefficients = np.linalg.lstsq(selected, values, rcond=None)[0]
            if (coefficients < 0).any():
                continue
            error = float(np.square(selected @ coefficients - values).sum())
            if error < best[0]:
                vector = np.zeros(design.shape[1]); vector[list(columns)] = coefficients
                best = error, vector
    return tuple(float(value) for value in best[1])


def freeze(inputs: list[Path], output: Path) -> None:
    groups: dict[tuple[str, tuple[str, ...]], dict[tuple[int, ...], list[int]]] = {}
    identity = None
    hashes = {}
    shapes = set()
    for path in inputs:
        document = json.loads(path.read_text())
        if document["split"] != "calibration" or document["correctness"] != "passed":
            raise ValueError("only checked calibration sessions may fit profiles")
        current = Identity(**document["identity"])
        if identity is not None and identity != current:
            raise ValueError("mixed backend/hardware/condition identities")
        identity = current
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in document["rows"]:
            op = Work.loads(row["descriptor"])
            shapes.add((type(op.spec).__name__, op.spec.b if isinstance(op.spec, KDA) else op.spec.n, document["routing"]))
            for stage in lower(op).stages:
                groups.setdefault((stage.name, stage.family), {}).setdefault(stage.features, []).append(row["measurements"][stage.name]["median_ns"])
    if identity is None:
        raise ValueError("no calibration input")
    profiles = []
    for (name, family), samples in sorted(groups.items()):
        keys = sorted(samples)
        durations = [round(statistics.median(samples[key])) for key in keys]
        coefficients = nonnegative_fit(keys, durations)
        profile_id = hashlib.sha256(json.dumps([asdict(identity), name, family, keys, durations, coefficients]).encode()).hexdigest()
        profiles.append(Profile(profile_id, identity, name, family, tuple(zip(keys, durations)), coefficients,
                                tuple(min(row[i] for row in keys) for i in range(len(keys[0]))),
                                tuple(max(row[i] for row in keys) for i in range(len(keys[0])))))
    output.write_text(json.dumps({"schema": 1, "identity": asdict(identity), "training_hashes": hashes, "training_shapes": sorted(shapes),
                                  "profiles": [asdict(profile) for profile in profiles]}, sort_keys=True))
    print("frozen", len(profiles), hashlib.sha256(output.read_bytes()).hexdigest())


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else 0.


def evaluate(profile_path: Path, inputs: list[Path], output: Path, qualified_output: Path | None = None) -> None:
    identity, profiles = read_profiles(profile_path)
    store = CalibrationStore(identity, profiles, candidate=True)
    frozen = json.loads(profile_path.read_text())
    training_hashes = set(frozen["training_hashes"].values())
    stage_errors: dict[str, list[tuple[int, int]]] = {}
    layer_errors = []
    records = []
    cpu_times, digests = [], []
    seen_shapes = set()
    seen_inputs: set[str] = set()
    sessions: set[int] = set()
    publication_samples = True
    for path in inputs:
        document = json.loads(path.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in training_hashes or digest in seen_inputs or document["split"] != "validation":
            raise ValueError("calibration/validation leakage")
        seen_inputs.add(digest)
        sessions.add(document["session"])
        publication_samples &= document["warmup"] >= 30 and document["iterations"] >= 200
        if Identity(**document["identity"]) != identity or document["correctness"] != "passed":
            raise ValueError("validation provenance mismatch")
        for row in document["rows"]:
            op = Work.loads(row["descriptor"])
            registry = Registry((op,))
            started = time.perf_counter_ns()
            report = replay(registry, store)
            cpu_times.append(time.perf_counter_ns() - started)
            for item in report.timeline:
                measured = row["measurements"][item.stage]["median_ns"]
                stage_errors.setdefault(item.stage, []).append((measured, item.end_ns - item.start_ns))
            measured_whole = row["measurements"]["whole"]["median_ns"]
            predicted_whole = report.makespan_ns
            assert predicted_whole is not None and report.unknown_count == 0
            error = abs(predicted_whole - measured_whole) / measured_whole
            layer_errors.append(error)
            shape = (type(op.spec).__name__, op.spec.b if isinstance(op.spec, KDA) else op.spec.n, document["routing"])
            if shape in {tuple(item) for item in frozen.get("training_shapes", ())}:
                raise ValueError("validation must hold out whole shape/routing configurations")
            seen_shapes.add(shape)
            records.append((op.op_id, document["session"], measured_whole, predicted_whole, error))
            if not digests:
                digests = [replay(registry, store).digest() for _ in range(100)]
    if not records:
        raise ValueError("no validation observations")
    long_relative = [abs(p - m) / m for values in stage_errors.values() for m, p in values if m >= 10000]
    short_absolute = [abs(p - m) for values in stage_errors.values() for m, p in values if m < 10000]
    stage_gate = all(percentile([abs(p-m)/m for m,p in values if m >= 10000], .5) <= .15 and
                     percentile([abs(p-m)/m for m,p in values if m >= 10000], .95) <= .30 and
                     percentile([abs(p-m) for m,p in values if m < 10000], .95) <= 2000 for values in stage_errors.values())
    gate = publication_samples and len(sessions) >= 5 and len(seen_shapes) >= 3 and stage_gate and percentile(layer_errors, .5) <= .20
    lines = ["# M1 independent shape validation", "", f"Frozen profile SHA256: `{hashlib.sha256(profile_path.read_bytes()).hexdigest()}`.", "",
             f"Gate: **{'PASS' if gate else 'FAIL'}**; {len(seen_shapes)} shape/routing configurations, {len(records)} session observations, strict unknown count **0**.",
             f"Long stage p50/p95 error: {percentile(long_relative, .5)*100:.2f}% / {percentile(long_relative, .95)*100:.2f}%; short stage p95 absolute error: {percentile(short_absolute,.95)/1000:.3f} us; whole operator p50 error: {percentile(layer_errors,.5)*100:.2f}%.",
             f"100 replay hashes identical: **{len(set(digests)) == 1}**. CPU replay median: {statistics.median(cpu_times)/1000:.2f} us (one operator, imports excluded).", "",
             "| Operator | Session | Native GPU us | Predicted us | Error |", "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {name} | {session} | {measured/1000:.3f} | {predicted/1000:.3f} | {error*100:.2f}% |" for name, session, measured, predicted, error in records]
    lines += ["", "| Stage | p50 error | p95 error | p95 absolute us |", "| --- | ---: | ---: | ---: |"]
    for name, values in sorted(stage_errors.items()):
        relative = [abs(p-m)/m for m,p in values]
        absolute = [abs(p-m)/1000 for m,p in values]
        lines.append(f"| {name} | {percentile(relative,.5)*100:.2f}% | {percentile(relative,.95)*100:.2f}% | {percentile(absolute,.95):.3f} |")
    output.write_text("\n".join(lines) + "\n")
    print("PASS" if gate else "FAIL", output)
    if not gate:
        raise SystemExit(1)
    if qualified_output is not None:
        frozen["profiles"] = [asdict(replace(profile, qualified=True)) for profile in profiles]
        frozen["frozen_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        frozen["validation_hashes"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        qualified_output.write_text(json.dumps(frozen, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "evaluate"))
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--qualified-output", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.input, args.output)
    else:
        if args.profiles is None:
            parser.error("evaluate requires --profiles")
        evaluate(args.profiles, args.input, args.output, args.qualified_output)


if __name__ == "__main__":
    main()
