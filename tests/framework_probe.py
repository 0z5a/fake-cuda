"""Compatibility entry point; framework probes now live in integration/frameworks."""

import argparse
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=("vllm", "sglang"))
    framework, remaining = parser.parse_known_args()
    script = Path(__file__).resolve().parent / "integration/frameworks" / f"{framework.framework}_startup.py"
    sys.argv = [str(script), *remaining]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
