"""Complete CPU trace process: same profiles/delivery, paced or offline clock."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.common.clock import Coordinator, PacedCoordinator
from sim.cost import CalibrationStore
from sim.replay import replay
from sim.serde import read_profiles
from sim.work import Registry, Work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True, help="native checked session manifest")
    parser.add_argument("--mode", choices=("paced", "offline"), required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("positive repetition count required")
    identity, profiles = read_profiles(args.profiles)
    store = CalibrationStore(identity, profiles)
    document = json.loads(args.trace.read_text())
    works = tuple(Work.loads(row["descriptor"]) for row in document["rows"])
    clock_type = Coordinator if args.mode == "offline" else PacedCoordinator
    target = hashlib.sha256()
    started = time.perf_counter_ns()
    for instance in range(args.repetitions):
        for work in works:
            bound = replace(work, execution=replace(work.execution, instance=str(instance)))
            report = replay(Registry((bound,)), store)
            assert report.makespan_ns is not None and report.unknown_count == 0
            clock = clock_type(("host", "device"))
            clock.send("host", "device", "complete", report.digest(), 0, report.makespan_ns)
            clock.certify("host", report.makespan_ns)
            clock.certify("device", report.makespan_ns)
            reply = clock.advance()[0]
            target.update(f"{reply.payload}:{reply.deliver_ns}:{reply.observed_ns}".encode())
    args.output.write_text(json.dumps({"mode": args.mode, "target_sha256": target.hexdigest(),
        "run_ns": time.perf_counter_ns() - started, "operator_replays": args.repetitions * len(works),
        "profile_sha256": hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
