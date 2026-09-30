"""An explicit shared Graph/KV budget changes original scheduler admission."""
import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.common.cache import CacheLedger
from adapters.vllm.costs import Shape, StepModel, StepSample
from adapters.vllm.runner import Workload, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()
    data = args.config.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.config_sha256 or version("vllm") != "0.30.0":
        parser.error("configuration or version mismatch")
    model = StepModel([StepSample(Shape(phase, batch, context), 1_000_000)
                       for phase in ("prefill", "decode") for batch in (1, 2) for context in (16, 32, 64)])
    workload = Workload("kv-budget", (0, 0), 32, 4)
    results = []
    print("| Graph reservation in block-equivalent bytes | Original KV blocks | First batch requests | Virtual finish ns |\n|---:|---:|---:|---:|")
    for graph_blocks in (0, 3):
        config, kv, block, hashed = pickle.loads(data)
        block_bytes = sum(group.kv_cache_spec.page_size_bytes * len(group.layer_names)
                          for group in kv.kv_cache_groups)
        ledger = CacheLedger(7 * block_bytes)
        if graph_blocks:
            ledger.reserve_graph("qualified-pool", graph_blocks * block_bytes)
        blocks = (ledger.capacity - ledger.used_bytes) // block_bytes
        kv = replace(kv, num_blocks=blocks)
        config.scheduler_config.max_num_seqs = 2
        result = run(workload, [(config, kv, block, hashed)], [model], args.bridge)
        assert result.control_reads == 8 and result.arrivals_observed_ns == [0, 0]
        results.append(result)
        print(f"| {graph_blocks} | {blocks} | {len(result.batch_ids[0])} | {result.finish_ns} |")
    assert len(results[0].batch_ids[0]) == 2 and len(results[1].batch_ids[0]) == 1
    assert results[0].finish_ns == 4_000_000 and results[1].finish_ns == 8_000_000
    print("PASS M03: layout-derived block bytes, explicit reservation, original KVCacheManager admission changes execution")
    print("This controlled budget is caller-declared; it does not claim to reproduce vLLM's automatic GPU profiling or measure a native CUDA Graph pool.")


if __name__ == "__main__":
    main()
