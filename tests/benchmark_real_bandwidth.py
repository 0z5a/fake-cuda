"""Compatibility entry point for the optional real-GH200 bandwidth measurement."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "integration/tooling/benchmark_real_bandwidth.py"), run_name="__main__")
