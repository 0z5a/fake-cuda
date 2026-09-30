"""Compare complete fresh processes, including imports, fit, IPC and teardown."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    campaign = Path(__file__).with_name("serving_campaign.py")
    times: dict[str, list[float]] = {"before": [], "after": []}
    reference = None
    for repeat in range(3):
        order = (("before", args.baseline), ("after", args.candidate))
        for name, executable in order if repeat % 2 == 0 else reversed(order):
            start = time.perf_counter_ns()
            data = subprocess.check_output([sys.executable, str(campaign), "--evidence", str(args.evidence),
                "--config-sha256", args.config_sha256, "--bridge", str(executable)], text=True)
            wall = (time.perf_counter_ns() - start) / 1e9
            signature = json.loads(data)
            if reference is not None and signature != reference:
                raise ValueError("full campaign target outputs differ")
            reference = signature
            times[name].append(wall)
            print(f"completed {name} repeat={repeat} wall={wall:.3f}s", file=sys.stderr, flush=True)
    before, after = (statistics.median(times[name]) for name in ("before", "after"))
    print("# Complete CPU serving process comparison\n")
    print("Three repeats per version, alternating process order, same vLLM 0.30.0 original Scheduler/KV code, frozen calibration and 18 complete workloads. Wall time covers interpreter startup/imports, calibration fit, configuration loading, every scheduler/oracle/predictor/IPC operation and natural subprocess/interpreter teardown. Prior model download, native calibration and build are excluded from both versions.\n")
    print("| Complete workloads per process | Previous s | Optimized s | Simulator speedup | Wall-time reduction |\n|---:|---:|---:|---:|---:|")
    print(f"| {reference['cases']} | {before:.3f} | {after:.3f} | {before/after:.3f}× | {100*(1-after/before):.2f}% |")
    print(f"\nAll six processes produced identical virtual token/batch/arrival timelines: `{reference['target_sha256']}`. These are complete timing simulation runs; the simulator emits an explicit fixed-length token oracle and does not execute numerical inference.\n")
    print("| Binary | SHA-256 |\n|---|---|")
    for name, path in (("Before", args.baseline), ("After", args.candidate)):
        print(f"| {name} | `{hashlib.sha256(path.read_bytes()).hexdigest()}` |")


if __name__ == "__main__":
    main()
