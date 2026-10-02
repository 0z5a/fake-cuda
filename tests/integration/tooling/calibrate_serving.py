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
import socket
import subprocess
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.core import resolve_kv_cache_block_sizes
from vllm.distributed.parallel_state import get_world_group
from adapters.vllm.runner import ServingResult, Workload
from adapters.vllm.worker import scheduled_shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--transport", choices=("auto", "socket"), default="auto")
    parser.add_argument("--native-rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if version("vllm") != "0.30.0" or args.repeats < 2:
        parser.error("native collector requires vLLM 0.30.0 and at least two repeats")
    root = args.evidence
    root.mkdir(parents=True, exist_ok=True)
    if args.transport == "socket":
        os.environ.update(NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1", NCCL_NET="Socket")
    if args.tp > 1 and args.native_rank is None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        processes = []
        for rank in range(args.tp):
            environment = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank),
                               WORLD_SIZE=str(args.tp), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
            processes.append(subprocess.Popen([sys.executable, __file__, *sys.argv[1:],
                                               "--native-rank", str(rank)], env=environment))
        statuses = [process.wait() for process in processes]
        if any(statuses):
            raise RuntimeError(f"native rank failure: {statuses}")
        return
    rank = args.native_rank or 0
    started = time.perf_counter_ns()
    llm = LLM(model=str(args.model), dtype="bfloat16", tensor_parallel_size=args.tp,
              distributed_executor_backend="external_launcher" if args.tp > 1 else "uni", seed=1,
              disable_custom_all_reduce=args.transport == "socket",
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
    if rank == 0:
        (root / "scheduler-config.pkl").write_bytes(config_data)
        (root / "native-run.json").write_text(json.dumps({"tp": args.tp, "transport": args.transport,
            "vllm": version("vllm"), "torch": torch.__version__, "cuda": torch.version.cuda,
            "scope": "whole_step_including_host_and_communication"}))
    cpu_group = get_world_group().cpu_group
    native_pids = [os.getpid()] * args.tp
    if args.tp > 1:
        torch.distributed.all_gather_object(native_pids, os.getpid(), group=cpu_group)

    def background_jobs() -> list[str]:
        snapshot = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                           "--format=csv,noheader"], text=True)
        return sorted(line for line in snapshot.splitlines() if int(line.split(",", 1)[0]) not in native_pids)
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
        if rank == 0:
            print(f"begin {workload.name} sequences={max_sequences}", flush=True)
        scheduler.max_num_running_reqs = max_sequences
        background_before = background_jobs() if rank == 0 else []
        if args.tp > 1:
            torch.distributed.barrier(group=cpu_group)
        begin = time.perf_counter_ns()
        submitted = 0
        tokens: dict[str, list[int]] = {}
        arrivals: list[int] = []
        steps.clear()
        while submitted < len(workload.arrivals_ns) or engine.has_unfinished_requests():
            elapsed = time.perf_counter_ns() - begin
            if args.tp > 1 and submitted < len(workload.arrivals_ns):
                # Every native rank admits the same arrivals. This control
                # broadcast is outside the measured engine.step() boundary.
                shared_elapsed = torch.tensor([elapsed if rank == 0 else 0], dtype=torch.int64)
                torch.distributed.broadcast(shared_elapsed, src=0, group=cpu_group)
                elapsed = shared_elapsed.item()
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
        background_after = background_jobs() if rank == 0 else []
        if rank == 0:
            print(f"finished {workload.name} steps={len(steps)} wall={finish/1e9:.3f}s", flush=True)
        result = ServingResult(finish, tokens, [step["ids"] for step in steps], [], arrivals, 0)
        if args.tp > 1:
            digests = [""] * args.tp
            semantic = [(step["shape"], step["ids"]) for step in steps]
            digest = hashlib.sha256(json.dumps((semantic, {rid: len(t) for rid, t in tokens.items()}), sort_keys=True).encode()).hexdigest()
            torch.distributed.all_gather_object(digests, digest, group=cpu_group)
            if len(set(digests)) != 1:
                raise RuntimeError("native TP rank batch/token mismatch")
        return {"workload": asdict(workload), "max_sequences": max_sequences,
                "metrics": result.metrics(workload), "steps": list(steps),
                "tokens": tokens, "arrivals_observed_ns": arrivals,
                "background_jobs_before": background_before, "background_jobs_after": background_after}

    collect(Workload("warmup", (0,) * 32, 64, 8), 32)
    calibration = []
    for batch in (1, 8, 16, 32):
        for prompt in (32, 64, 128):
            collect(Workload("shape-warmup", (0,) * batch, prompt, 8), 32)
            for repeat in range(2):
                calibration.append(collect(Workload(f"fit-{batch}-{prompt}-{repeat}", (0,) * batch, prompt, 96), 32))
    calibration_data = json.dumps(calibration).encode()
    if rank == 0:
        (root / "calibration.json").write_bytes(calibration_data)
    # The file above is immutable input to fitting; held-out measurements follow.
    validation = []
    for sequences in (8, 16, 32):
        for repeat in range(args.repeats):
            for workload in (Workload("burst-128", (0,) * 128, 64, 64),
                             Workload("two-waves-96", (0,) * 48 + (100_000_000,) * 48, 64, 64)):
                validation.append(collect(workload, sequences))
                if rank == 0:
                    print(f"captured {workload.name} sequences={sequences} repeat={repeat}", flush=True)
    if rank == 0:
        (root / "validation.json").write_text(json.dumps(validation))
        print(f"native TP{args.tp} campaign wall seconds={(time.perf_counter_ns()-started)/1e9:.3f}")
        print(f"vllm={version('vllm')} torch={torch.__version__} CUDA={torch.version.cuda}")
        print(f"configuration_sha256={hashlib.sha256(config_data).hexdigest()}")
        print(f"calibration_sha256={hashlib.sha256(calibration_data).hexdigest()}")
    engine.engine_core.engine_core.shutdown()


if __name__ == "__main__":
    main()
