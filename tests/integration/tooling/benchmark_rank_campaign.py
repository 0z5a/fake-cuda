"""Measure fresh rank startup versus run-epoch reuse with identical targets."""
import argparse
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
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("limit must be positive")
    campaign = Path(__file__).with_name("rank_campaign.py")
    command = [sys.executable, str(campaign), "--evidence", str(args.evidence),
               "--config-sha256", args.config_sha256, "--bridge", str(args.bridge),
               "--limit", str(args.limit)]
    times: dict[str, list[float]] = {"fresh": [], "reused": []}
    signature = None
    for repeat in range(3):
        order = ("fresh", "reused") if repeat % 2 == 0 else ("reused", "fresh")
        for mode in order:
            begin = time.perf_counter_ns()
            result = json.loads(subprocess.check_output(
                command + (["--fresh-ranks"] if mode == "fresh" else []), text=True))
            wall = (time.perf_counter_ns() - begin) / 1e9
            if result["unsupported"] or len(result["rows"]) != args.limit:
                raise ValueError("speed comparison requires complete supported workloads")
            if signature is not None and result["semantic_sha256"] != signature:
                raise ValueError("rank lifecycle optimization changed virtual targets")
            signature = result["semantic_sha256"]
            times[mode].append(wall)
            print(f"completed {mode} repeat={repeat} wall={wall:.3f}s", file=sys.stderr, flush=True)
    fresh, reused = (statistics.median(times[mode]) for mode in ("fresh", "reused"))
    print("# Independent rank process lifecycle comparison\n")
    print(f"Three repeats per mode in alternating order; {args.limit} complete retained workloads per process. Both modes use identical source, original vLLM schedulers, frozen costs, rank transport and target work. Fresh starts all CPU ranks for each workload; reused starts once, then resets every Scheduler, KV state and control ledger under a new run epoch. Outer wall time includes interpreter/imports, fitting, every rank startup, scheduling, bridge IPC, local parity oracle, and natural teardown. Prior native calibration/download/build are excluded.\n")
    print("| Complete workloads | Fresh ranks s | Reused ranks s | Speedup | Wall reduction |\n|---:|---:|---:|---:|---:|")
    print(f"| {args.limit} | {fresh:.3f} | {reused:.3f} | {fresh/reused:.3f}× | {100*(1-reused/fresh):.2f}% |")
    print(f"\nAll six runs have identical virtual finish, token, batch, arrival and control-read digest: `{signature}`. CPU rank PID and IPC wall delay are excluded from the digest. This measures timing simulation execution, not numerical inference acceleration.")


if __name__ == "__main__":
    main()
