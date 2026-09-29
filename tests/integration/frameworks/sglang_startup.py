"""Stage SGLang Engine import and optional local-model startup; no inference."""

import argparse
import importlib
from pathlib import Path

from common import check_environment, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Local Hugging Face model directory (no download)")
    args = parser.parse_args()
    if args.model and not args.model.is_dir():
        parser.error("--model must be a local directory")

    check_environment("sglang")
    engine = stage(
        "SGLang Engine import",
        lambda: importlib.import_module("sglang.srt.entrypoints.engine").Engine,
    )
    if args.model:
        stage(
            "SGLang engine startup (local model)",
            lambda: engine(model_path=str(args.model), skip_server_warmup=True),
        )
        print("PASS: SGLang engine started; no generation attempted", flush=True)
    else:
        print("SKIP SGLang engine startup: no local model supplied", flush=True)
    print("NOTE: numerical inference is impossible: fake CUDA does not store data or execute kernels", flush=True)


if __name__ == "__main__":
    main()
