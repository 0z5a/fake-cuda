"""Compatibility entry point for the optional real-GH200 profile comparison."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "integration/tooling/compare_real_gh200.py"), run_name="__main__")
