"""Line protocol keeps AIS dependencies separate from the original vLLM env."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.aisimulate.model import Identity, Regression
from adapters.vllm.costs import Shape, StepSample, UnsupportedStep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    data = args.calibration.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.sha256:
        parser.error("frozen calibration digest mismatch")
    samples = [StepSample(Shape(**step["shape"]), step["duration_ns"])
               for case in json.loads(data) for step in case["steps"]]
    identity = Identity(**json.loads(args.identity))
    model = Regression(identity, identity, samples)
    try:
        for line in sys.stdin:
            phase, batch, context = line.split()
            try:
                prediction = str(model.predict(Shape(phase, int(batch), float(context))))
            except UnsupportedStep as error:
                prediction = f"unsupported {error}"
            print(prediction, flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()
