"""Native Dynamo reservations driven by original vLLM completion feedback."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.dynamo.client import DynamoPolicy
from adapters.vllm.costs import Shape, StepModel, StepSample
from adapters.vllm.routing import run
from adapters.vllm.runner import Workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dynamo-python", type=Path, required=True)
    args = parser.parse_args()
    data = args.config.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.config_sha256 or version("vllm") != "0.30.0":
        parser.error("configuration or version mismatch")
    model = StepModel([StepSample(Shape(phase, 1, context), 5000)
                       for phase in ("prefill", "decode") for context in (32, 64, 128, 256)])
    workload = Workload("route-feedback", tuple(i * 1000 for i in range(80)), 64, 8)
    records = []
    print("| Endpoint service scales | Virtual finish ns | Requests to endpoint 0 | Requests to endpoint 1 |\n|---|---:|---:|---:|")
    for scales in ((1., 1.), (.2, 1.)):
        configurations = [pickle.loads(data) for _ in scales]
        for config, kv, block, hashed in configurations:
            config.scheduler_config.max_num_seqs = 1
        policy = DynamoPolicy(args.dynamo_python, 2)
        result, routes = run(workload, configurations, model, scales, 500, args.bridge, policy)
        counts = Counter(route.endpoint for route in routes)
        assert result.arrivals_observed_ns == list(workload.arrivals_ns)
        assert len(routes) == 80 and result.control_reads == 640
        assert all(max(route.visible_at) <= route.arrival_ns for route in routes)
        assert all(sum(row["active_requests"] for row in route.native_loads) <= int(route.request) for route in routes)
        assert any(row["potential_prefill_tokens"] > 0 for route in routes for row in route.native_loads)
        assert all(row["active_requests"] == row["potential_prefill_tokens"] == row["potential_decode_blocks"] == 0 for row in policy.final_loads)
        print(f"| {scales} | {result.finish_ns} | {counts[0]} | {counts[1]} |")
        records.append((result, routes, counts))
    assert records[1][0].finish_ns < records[0][0].finish_ns
    assert records[1][2][0] > records[0][2][0]
    assert [r.endpoint for r in records[0][1]] != [r.endpoint for r in records[1][1]]
    print("PASS R02/R05: native Dynamo eligibility/reservations/load tracker, original vLLM Scheduler/KV release, delayed prefill/completion feedback, no CUDA")
    print("Synthetic service perturbation tests causal sensitivity; this is not an inference speedup or a disaggregated Dynamo deployment.")


if __name__ == "__main__":
    main()
