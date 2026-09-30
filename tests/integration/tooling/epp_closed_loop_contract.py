"""Native EPP profile feedback with two original CPU vLLM schedulers."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.llmd_epp.closed_loop import run
from adapters.llmd_epp.client import NativeEpp
from adapters.vllm.costs import Shape, StepModel, StepSample
from adapters.vllm.runner import Workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--epp", type=Path, required=True)
    args = parser.parse_args()
    data = args.config.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.config_sha256 or version("vllm") not in ("0.29.0", "0.30.0"):
        parser.error("configuration or version mismatch")
    native = NativeEpp(args.epp)
    try:
        snapshot = [{"id": "disabled", "queue": 0, "enabled": False},
                    {"id": "0", "queue": 9, "enabled": True},
                    {"id": "1", "queue": 3, "enabled": True}]
        selected = json.loads(native.command(json.dumps(snapshot)))
        assert selected["picked"] == "1" and set(selected["scores"]) == {"0", "1"}
        print("PASS native EPP snapshot: disabled low-queue endpoint excluded before real scoring/picking")
    finally:
        native.close()
    model = StepModel([StepSample(Shape(phase, 1, context), 5000)
                       for phase in ("prefill", "decode") for context in (32, 64, 128, 256)])
    workload = Workload("route-feedback", tuple(i * 1000 for i in range(80)), 64, 8)
    print("| Endpoint service scales | Virtual finish ns | Requests to endpoint 0 | Requests to endpoint 1 |\n|---|---:|---:|---:|")
    records = []
    for scales in ((1., 1.), (.2, 1.)):
        configurations = [pickle.loads(data) for _ in scales]
        for config, kv, block, hashed in configurations:
            config.parallel_config.tensor_parallel_size = 1
            config.scheduler_config.max_num_seqs = 1
        result, routes = run(workload, configurations, model, scales, 500, args.bridge, args.epp)
        counts = Counter(route.endpoint for route in routes)
        assert result.arrivals_observed_ns == list(workload.arrivals_ns)
        assert len(routes) == 80 and result.control_reads == 640
        assert all(max(route.visible_at) <= route.arrival_ns for route in routes)
        assert all(route.scores[str(route.endpoint)] == max(route.scores.values()) for route in routes)
        print(f"| {scales} | {result.finish_ns} | {counts[0]} | {counts[1]} |")
        records.append((result, routes, counts))
    assert records[1][0].finish_ns < records[0][0].finish_ns
    assert records[1][2][0] > records[0][2][0]
    assert [r.endpoint for r in records[0][1]] != [r.endpoint for r in records[1][1]]
    print("PASS R01/R05: real EPP filter/queue scorer/picker, delayed metrics, real scheduler/cache feedback; 80 requests/640 oracle tokens, no CUDA")
    print("Scales and costs are synthetic causal-sensitivity inputs, not measured hardware speedups. Native picker tie behavior is retained.")


if __name__ == "__main__":
    main()
