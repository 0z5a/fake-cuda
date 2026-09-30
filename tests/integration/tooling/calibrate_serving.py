"""Collect native whole-step costs, then independent fixed-length workloads."""
import argparse
import copy
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.core import resolve_kv_cache_block_sizes
from adapters.vllm.runner import ServingResult, Workload
from adapters.vllm.worker import scheduled_shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if version("vllm") != "0.30.0" or args.repeats < 2:
        parser.error("native collector requires vLLM 0.30.0 and at least two repeats")
    root = args.evidence
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    llm = LLM(model=str(args.model), dtype="bfloat16", tensor_parallel_size=1,
              enforce_eager=True, async_scheduling=False, enable_prefix_caching=False,
              enable_chunked_prefill=False, max_model_len=512, max_num_seqs=32,
              max_num_batched_tokens=4096, kv_cache_memory_bytes=1 << 30,
              attention_backend="TRITON_ATTN", disable_log_stats=True,
              compilation_config={"mode": 0})
    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    block, hashed = resolve_kv_cache_block_sizes(scheduler.kv_cache_config, scheduler.vllm_config)
    config = copy.copy(scheduler.vllm_config)
    config.compilation_config = copy.copy(config.compilation_config)
    # Loaded attention modules hold GPU storage; scheduler replay needs only
    # configuration, never the worker's static forward-context objects.
    config.compilation_config.static_forward_context = {}
    configuration = (config, scheduler.kv_cache_config, block, hashed)
    config_data = pickle.dumps(configuration)
    (root / "scheduler-config.pkl").write_bytes(config_data)
    native_schedule = scheduler.schedule
    steps: list[dict] = []
    identities: dict[str, str] = {}

    def schedule(throttle_prefills: bool = False):
        output = native_schedule(throttle_prefills)
        if output.num_scheduled_tokens:
            steps.append({"shape": asdict(scheduled_shape(scheduler, output)),
                          "ids": [identities[rid] for rid in output.num_scheduled_tokens]})
        return output

    scheduler.schedule = schedule

    def collect(workload: Workload, max_sequences: int) -> dict:
        scheduler.max_num_running_reqs = max_sequences
        begin = time.perf_counter_ns()
        submitted = 0
        tokens: dict[str, list[int]] = {}
        arrivals: list[int] = []
        steps.clear()
        while submitted < len(workload.arrivals_ns) or engine.has_unfinished_requests():
            elapsed = time.perf_counter_ns() - begin
            while submitted < len(workload.arrivals_ns) and workload.arrivals_ns[submitted] <= elapsed:
                identity = str(submitted)
                internal = engine.add_request(identity,
                    {"prompt_token_ids": [100 + submitted % 8] * workload.prompt},
                    SamplingParams(temperature=0, ignore_eos=True, max_tokens=workload.output,
                                   detokenize=False, output_kind=RequestOutputKind.DELTA),
                    arrival_time=(begin + workload.arrivals_ns[submitted]) / 1e9)
                identities[internal] = identity
                arrivals.append(time.perf_counter_ns() - begin)
                submitted += 1
            if not engine.has_unfinished_requests():
                time.sleep(max(0, (workload.arrivals_ns[submitted] - elapsed) / 1e9))
                continue
            before = time.perf_counter_ns()
            outputs = engine.step()
            after = time.perf_counter_ns()
            steps[-1]["duration_ns"] = after - before
            for output in outputs:
                tokens.setdefault(output.request_id, []).extend(
                    [after - begin] * len(output.outputs[0].token_ids))
        finish = time.perf_counter_ns() - begin
        result = ServingResult(finish, tokens, [step["ids"] for step in steps], [], arrivals, 0)
        return {"workload": asdict(workload), "max_sequences": max_sequences,
                "metrics": result.metrics(workload), "steps": list(steps),
                "tokens": tokens, "arrivals_observed_ns": arrivals}

    collect(Workload("warmup", (0,) * 32, 64, 8), 32)
    calibration = []
    for batch in (1, 8, 16, 32):
        for prompt in (32, 64, 128):
            collect(Workload("shape-warmup", (0,) * batch, prompt, 8), 32)
            for repeat in range(2):
                calibration.append(collect(Workload(f"fit-{batch}-{prompt}-{repeat}", (0,) * batch, prompt, 96), 32))
    calibration_data = json.dumps(calibration).encode()
    (root / "calibration.json").write_bytes(calibration_data)
    # The file above is immutable input to fitting; held-out measurements follow.
    validation = []
    for sequences in (8, 16, 32):
        for repeat in range(args.repeats):
            for workload in (Workload("burst-128", (0,) * 128, 64, 64),
                             Workload("two-waves-96", (0,) * 48 + (100_000_000,) * 48, 64, 64)):
                validation.append(collect(workload, sequences))
                print(f"captured {workload.name} sequences={sequences} repeat={repeat}", flush=True)
    (root / "validation.json").write_text(json.dumps(validation))
    print(f"native campaign wall seconds={(time.perf_counter_ns()-started)/1e9:.3f}")
    print(f"vllm={version('vllm')} torch={torch.__version__} CUDA={torch.version.cuda}")
    print(f"configuration_sha256={hashlib.sha256(config_data).hexdigest()}")
    print(f"calibration_sha256={hashlib.sha256(calibration_data).hexdigest()}")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
