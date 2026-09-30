"""One complete CPU serving campaign, suitable for process-level comparison."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.vllm.costs import Shape, StepModel, StepSample
from adapters.vllm.runner import Workload, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()
    config = (args.evidence / "scheduler-config.pkl").read_bytes()
    if hashlib.sha256(config).hexdigest() != args.config_sha256 or version("vllm") != "0.30.0":
        parser.error("configuration or version mismatch")
    model = StepModel([StepSample(Shape(**step["shape"]), step["duration_ns"])
                      for case in json.loads((args.evidence / "calibration.json").read_text()) for step in case["steps"]])
    signatures = []
    for case in json.loads((args.evidence / "validation.json").read_text()):
        workload = Workload(**{**case["workload"], "arrivals_ns": tuple(case["workload"]["arrivals_ns"])})
        cfg = pickle.loads(config)
        cfg[0].scheduler_config.max_num_seqs = case["max_sequences"]
        result = run(workload, [cfg], [model], args.bridge)
        signatures.append((result.finish_ns, result.tokens, result.batch_ids, result.arrivals_observed_ns))
    print(json.dumps({"cases": len(signatures), "target_sha256": hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()}))


if __name__ == "__main__":
    main()
