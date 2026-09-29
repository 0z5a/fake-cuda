"""Stage vLLM platform detection and optional local-model engine startup; no inference."""

import argparse
import importlib
import os
from pathlib import Path

from common import check_environment, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Local Hugging Face model directory (no download)")
    args = parser.parse_args()
    if args.model and not args.model.is_dir():
        parser.error("--model must be a local directory")

    # Diagnostic only: isolate downstream Driver/API failures from vLLM's
    # external NVML requirement. Apply before vLLM initialization, as its
    # current_platform result is cached. Never enable this for the baseline.
    if os.environ.get("FAKE_CUDA_VLLM_DIAGNOSE_WITHOUT_NVML") == "1":
        platforms = importlib.import_module("vllm.platforms")
        print("DIAGNOSTIC: bypassing vLLM's NVML platform detector; "
              "this is not a supported vLLM run", flush=True)
        platforms.builtin_platform_plugins["cuda"] = (
            lambda: "vllm.platforms.cuda.CudaPlatform"
        )
        # Importing vllm.platforms itself may have populated this cache.
        platforms._current_platform = None

    framework = check_environment("vllm")

    def require_cuda_platform():
        platform = importlib.import_module("vllm.platforms").current_platform
        if os.environ.get("FAKE_CUDA_VLLM_DIAGNOSE_WITHOUT_NVML") == "1":
            print(f"DIAGNOSTIC: resolved platform {type(platform)!r}", flush=True)
        if not platform.is_cuda():
            raise RuntimeError(
                "BLOCKED: CUDA not selected. vLLM 0.26.0 was observed to require "
                "NVML here (outside the Driver API); inspect platform logs "
                "for the cause in this installation."
            )
        return platform

    try:
        stage("vLLM platform detection", require_cuda_platform)
    except Exception as exc:
        if "nvml" in str(exc).lower() and "BLOCKED" not in str(exc):
            print("BLOCKED: NVML is outside this Driver-only project", flush=True)
        raise

    engine = stage("vLLM LLM import", lambda: framework.LLM)
    if args.model:
        stage("vLLM engine startup (local model)", lambda: engine(model=str(args.model), enforce_eager=True))
        print("PASS: vLLM engine started; no generation attempted", flush=True)
    else:
        print("SKIP vLLM engine startup: no local model supplied", flush=True)
    print("NOTE: numerical inference is impossible: fake CUDA does not store data or execute kernels", flush=True)


if __name__ == "__main__":
    main()
