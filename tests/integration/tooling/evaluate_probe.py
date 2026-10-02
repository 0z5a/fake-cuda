"""Evaluate frozen probe durations against independent reports; no live binding."""
import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import statistics
import subprocess


@dataclass(frozen=True)
class Sample:
    label: str
    key: tuple[str, ...]
    duration: int


def load(path: Path, replay: Path) -> tuple[str, list[Sample]]:
    # Share the consumer's schema checks rather than maintaining another parser.
    subprocess.run([str(replay), str(path)], check=True, capture_output=True)
    headers = {}
    samples = []
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        if key != "observation":
            headers[key] = value
            continue
        fields = value.split("\t")
        identity = (headers["code"], headers["hardware"], *fields[1:7], fields[8])
        duration = int(fields[7])
        if duration <= 0:
            raise ValueError("relative accuracy requires positive reference durations")
        samples.append(Sample(f"{path.parent.name}/{fields[0]}", identity, duration))
    for header, name in [("source", "capture.ncu-rep"), ("hardware", "hardware.csv"), ("conditions", "capture.txt")]:
        with (path.parent/name).open("rb") as artifact:
            digest = hashlib.file_digest(artifact, "sha256").hexdigest()
        if digest != headers[header].rsplit(":", 1)[1]:
            raise ValueError(f"{path}: changed {name}")
    return headers["source"], samples


def read_split(paths: list[Path], replay: Path) -> tuple[set[str], list[Sample]]:
    sources = set()
    samples = []
    for path in paths:
        source, records = load(path, replay)
        if source in sources:
            raise ValueError("duplicate report within a split")
        sources.add(source)
        samples.extend(records)
    return sources, samples


def freeze(calibration: list[Sample]) -> dict[tuple[str, ...], float]:
    groups: dict[tuple[str, ...], list[int]] = {}
    for sample in calibration:
        groups.setdefault(sample.key, []).append(sample.duration)
    return {key: statistics.median(values) for key, values in groups.items()}


def metrics(pairs: list[tuple[float, int]]) -> dict[str, float]:
    if not pairs or any(actual <= 0 or not math.isfinite(predicted) or predicted < 0 for predicted, actual in pairs):
        raise ValueError("expected finite predictions and positive reference durations")
    absolute = [abs(predicted-actual) for predicted, actual in pairs]
    relative = [error/actual*100 for error, (_, actual) in zip(absolute, pairs)]
    return dict(mape=statistics.mean(relative), p95=sorted(relative)[math.ceil(0.95*len(pairs))-1],
                maximum=max(relative), wape=sum(absolute)/sum(actual for _, actual in pairs)*100,
                bias=statistics.mean((predicted-actual)/actual*100 for predicted, actual in pairs),
                within10=sum(error <= 10 for error in relative)/len(pairs)*100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--conditions", required=True,
                        help="audited equivalence of inputs, compiler, GPU and profiling conditions across runs")
    args = parser.parse_args()
    calibration_sources, calibration = read_split(args.calibration, args.replay)
    model = freeze(calibration)
    validation_sources, validation = read_split(args.validation, args.replay)
    if calibration_sources & validation_sources:
        raise ValueError("calibration/validation report leakage")
    covered = [(model[sample.key], sample.duration) for sample in validation if sample.key in model]
    print("Frozen probe lookup evaluation; no runtime binding or unseen-configuration extrapolation.\n")
    print(f"Declared equivalent conditions: {args.conditions}\n")
    print(f"Calibration: {len(calibration_sources)} reports, {len(calibration)} observations. "
          f"Validation: {len(validation_sources)} independent reports, {len(validation)} observations.\n")
    print("| Estimator / evaluated set | Coverage | MAPE (%) | P95 APE (%) | Max APE (%) | WAPE (%) | Bias (%) | Within 10% (%) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    comparisons = [("Default 10 ms / all", [(10000000, s.duration) for s in validation])]
    if covered:
        comparisons += [("Default 10 ms / covered", [(10000000, actual) for _, actual in covered]),
                        ("Frozen probe / covered", covered)]
    for label, pairs in comparisons:
        result = metrics(pairs)
        values = " | ".join(f"{value:.3f}" for value in result.values())
        print(f"| {label} | {len(pairs)}/{len(validation)} | {values} |")
    print(f"\nUnsupported observations: {len(validation)-len(covered)}. Excluded from conditional error; included in coverage.\n")
    print("| Validation observation | Frozen prediction (µs) | Actual (µs) | APE (%) |")
    print("|---|---:|---:|---:|")
    for sample in validation:
        if sample.key in model:
            prediction = model[sample.key]
            print(f"| {sample.label} | {prediction/1000:.3f} | {sample.duration/1000:.3f} | "
                  f"{abs(prediction-sample.duration)/sample.duration*100:.3f} |")
        else:
            print(f"| {sample.label} | unsupported | {sample.duration/1000:.3f} | — |")


if __name__ == "__main__":
    main()
