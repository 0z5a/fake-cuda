"""Compare complete offline trace processes from two immutable source trees."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


@dataclass(frozen=True)
class Observation:
    session: int
    position: int
    arm: str
    wall_ns: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repetitions, args.sessions) < 1:
        parser.error("positive repetition/session counts required")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {"before": args.baseline.resolve(), "after": args.candidate.resolve()}
    records: list[Observation] = []
    reference = None
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for session in range(args.sessions):
        order = ("before", "after", "after", "before") if session % 2 == 0 else ("after", "before", "before", "after")
        for position, arm in enumerate(order):
            output = args.output / f"{session}-{position}-{arm}.json"
            command = [sys.executable, str(sources[arm] / "tests/integration/tooling/benchmark_semantic_replay.py"),
                       "--profiles", str(args.profiles.resolve()), "--trace", str(args.trace.resolve()),
                       "--mode", "offline", "--repetitions", str(args.repetitions), "--output", str(output)]
            start = time.perf_counter_ns()
            subprocess.run(command, env=environment, check=True)
            wall = time.perf_counter_ns() - start
            result = json.loads(output.read_text())
            signature = tuple(result[key] for key in ("target_sha256", "operator_replays", "profile_sha256", "trace_sha256"))
            if reference is not None and signature != reference:
                raise ValueError("trace, profile or target timeline/ledger/reply differs between arms")
            reference = signature
            records.append(Observation(session, position, arm, wall))
            print(session, position, arm, wall, flush=True)
    before, after = (statistics.median(row.wall_ns for row in records if row.arm == arm) for arm in ("before", "after"))
    source_hashes = {arm: {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("sim/replay.py", "sim/work.py", "sim/cost.py", "sim/models.py",
                                       "adapters/common/clock.py", "tests/integration/tooling/benchmark_semantic_replay.py")}
                     for arm, root in sources.items()}
    session_ratios = []
    for session in range(args.sessions):
        a, p = (statistics.median(row.wall_ns for row in records if row.arm == arm and row.session == session)
                for arm in ("before", "after"))
        session_ratios.append(a / p)
    summary = {"records": [asdict(row) for row in records], "source_hashes": source_hashes, "target_signature": reference,
               "before_ns": before, "after_ns": after, "speedup": before / after,
               "session_ratios": session_ratios, "scope": "complete_offline_trace_process",
               "calibration_fit": "frozen_and_reused_in_both_arms"}
    (args.output / "summary.json").write_text(json.dumps(summary, sort_keys=True))
    print(f"before={before/1e9:.6f}s after={after/1e9:.6f}s speedup={before/after:.3f}x", flush=True)


if __name__ == "__main__":
    main()
