"""Compare core bridges on identical completed-history workloads."""
import argparse
from pathlib import Path
import statistics
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--operations", type=int, default=10000)
    args = parser.parse_args()
    if args.operations <= 0:
        parser.error("operations must be positive")
    commands = []
    for i in range(args.operations):
        commands.extend((f"submit {i+1} 0 {i*1000} 1000", f"advance {(i+1)*1000}", f"status {i+1}"))
    # All retained completions remain queryable after advancing the engine.
    commands.extend(f"status {i+1}" for i in range(args.operations))
    text = "\n".join(commands) + "\n"
    outputs = []
    times = []
    for executable in (args.baseline, args.candidate):
        repeats = []
        for _ in range(3):
            start = time.perf_counter_ns()
            result = subprocess.check_output([str(executable), "1"], input=text, text=True)
            repeats.append((time.perf_counter_ns() - start) / 1e9)
            if outputs and result != outputs[0]:
                raise ValueError("core timeline or retained completions changed")
            outputs.append(result)
        times.append(statistics.median(repeats))
    print("| Operations | Before s | After s | Simulator speedup | Timeline parity |\n|---:|---:|---:|---:|---|")
    print(f"| {args.operations} | {times[0]:.6f} | {times[1]:.6f} | {times[0]/times[1]:.2f}× | Every reply identical |")
    print("Three repeats, median subprocess wall time including protocol and teardown. This measures simulator execution, not GPU performance.")


if __name__ == "__main__":
    main()
