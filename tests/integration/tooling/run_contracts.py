"""Run registered CTest commands without timeout-driven process termination."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    tests = json.loads(subprocess.check_output(
        ["ctest", "--test-dir", str(args.build), "--show-only=json-v1"], text=True))["tests"]
    failures = []
    for test in tests:
        environment = dict(os.environ)
        directory = args.build
        for prop in test["properties"]:
            if prop["name"] == "ENVIRONMENT":
                for entry in prop["value"]:
                    name, value = entry.split("=", 1)
                    environment[name] = value
            elif prop["name"] == "WORKING_DIRECTORY":
                directory = Path(prop["value"])
        outcome = subprocess.run(test["command"], cwd=directory, env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"{'FAIL' if outcome.returncode else 'PASS'} {test['name']}", flush=True)
        if outcome.returncode:
            print(outcome.stdout)
            failures.append(test["name"])
    print(f"{len(tests)-len(failures)}/{len(tests)} registered contracts passed")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
