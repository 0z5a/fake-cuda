"""Replay explicit operator descriptors into a Markdown timeline and ledger."""
import argparse
import json
from pathlib import Path
from .cost import AnalyticCost, CalibrationStore
from .replay import replay
from .serde import read_profiles
from .work import Registry, Work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="external JSON array of schema-1 descriptors")
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--mode", choices=("strict", "analytic", "compat"), default="strict")
    parser.add_argument("--rates", nargs=3, type=float, metavar=("TC_FLOP_S", "SIMT_FLOP_S", "BYTE_S"))
    parser.add_argument("--launch-ns", type=int, default=0)
    parser.add_argument("--logical-cold-traffic", action="store_true", help="uncalibrated assumption, not measured DRAM traffic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--releases", type=Path, help="external op_id -> host release in integer ns")
    args = parser.parse_args()
    if args.mode == "analytic":
        if args.rates is None:
            parser.error("analytic mode requires explicit rates with SI units")
        model = AnalyticCost(*args.rates, args.launch_ns, args.logical_cold_traffic)
    else:
        if args.profiles is None:
            parser.error("strict/compat require a profile identity")
        identity, profiles = read_profiles(args.profiles)
        model = CalibrationStore(identity, profiles, args.mode)
    descriptors = json.loads(args.trace.read_text())
    registry = Registry(tuple(Work.loads(json.dumps(value)) for value in descriptors))
    releases = json.loads(args.releases.read_text()) if args.releases else None
    report = replay(registry, model, releases, strict=args.mode == "strict")
    lines = ["# Operator replay", "", f"Scope: `{report.scope}`; mode: `{report.schedule_mode}`; status: **{report.status}**.",
             f"Makespan ns: {report.makespan_ns}; unknown stages: {report.unknown_count}; replay SHA256: `{report.digest()}`.", "",
             "| Operator | Stage | Category | Start ns | End ns | Cost status |", "| --- | --- | --- | ---: | ---: | --- |"]
    lines += [f"| {item.op_id} | {item.stage} | {item.category} | {item.start_ns} | {item.end_ns} | {item.cost.status} |" for item in report.timeline]
    lines += ["", "| Operator | Ledger | Quantity | Value |", "| --- | --- | --- | ---: |"]
    for estimate in report.estimates:
        lowered = estimate.lowered
        for name, values in (("logical", lowered.logical), ("executed", lowered.executed), ("traffic", lowered.traffic), ("residency", lowered.residency)):
            lines += [f"| {estimate.work.op_id} | {name} | {quantity} | {value} |" for quantity, value in sorted(values.items())]
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
