"""Complete checkpoint request/graph/chunk contracts and recorded-control replay."""
import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@dataclass(frozen=True)
class Input:
    prompt: int
    output: int
    release_step: int = 0


@dataclass(frozen=True)
class Case:
    name: str
    requests: tuple[Input, ...]
    abort_after: int = 0


CASES = (Case("single-output", (Input(48, 1),)),
         Case("uniform", (Input(128, 16),) * 8),
         Case("heterogeneous", tuple(Input(p, 8) for p in (32, 96, 112, 192))),
         Case("two-waves", (Input(64, 8),) * 2 + (Input(128, 8, 2),) * 2),
         Case("cancel-decode", (Input(48, 8),) * 2, 2),
         Case("cancel-prefill", (Input(192, 8),) * 2, 1))


def items(scheduler, output, identities: dict[str, str]) -> list[dict[str, int | str]]:
    return [{"id": identities[rid], "tokens": count,
             "computed_before": scheduler.requests[rid].num_computed_tokens - count,
             "prompt": scheduler.requests[rid].num_prompt_tokens}
            for rid, count in output.num_scheduled_tokens.items()]


def loaded_libraries() -> dict[str, str]:
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "/vllm/" in line and ".so" in line or "/libnccl.so" in line}
    hashes = {}
    for name in sorted(paths):
        with Path(name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def native(args) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import CUDAGraphMode
    from vllm.distributed.parallel_state import get_world_group
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.core import resolve_kv_cache_block_sizes
    if args.tp > 1 and args.native_rank is None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); port = listener.getsockname()[1]
        processes = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--native-rank", str(rank)],
                      env={**os.environ, "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(args.tp),
                           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}) for rank in range(args.tp)]
        statuses = [p.wait() for p in processes]
        if any(statuses):
            raise RuntimeError(f"native rank failures: {statuses}")
        return
    rank = args.native_rank or 0
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    began = time.perf_counter_ns()
    graph = "graph" in args.variant
    chunk = "chunk" in args.variant
    compilation = {"mode": 0, "cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY if graph else CUDAGraphMode.NONE,
                   "cudagraph_capture_sizes": [1, 2, 4, 8] if graph else []}
    llm = LLM(model=str(args.model), dtype="bfloat16", tensor_parallel_size=args.tp,
              distributed_executor_backend="external_launcher" if args.tp > 1 else "uni",
              enable_expert_parallel=args.ep, all2all_backend="allgather_reducescatter", seed=1,
              enforce_eager=not graph, async_scheduling=False, enable_prefix_caching=False,
              enable_chunked_prefill=chunk, max_model_len=512, max_num_seqs=8,
              max_num_batched_tokens=64 if chunk else 512,
              kv_cache_memory_bytes=args.kv_bytes // args.tp,
              gpu_memory_utilization=args.memory_fraction,
              attention_backend=None if args.attention_backend == "auto" else args.attention_backend,
              compilation_config=compilation,
              disable_custom_all_reduce=True, trust_remote_code=args.trust_model_code, disable_log_stats=True)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    runner = core.model_executor.driver_worker.worker.model_runner
    model = runner.get_model()
    layers = model.model.layers
    assert len(layers) == scheduler.vllm_config.model_config.hf_config.num_hidden_layers
    parameters = list(model.named_parameters())
    assert all(p.device.type == "cuda" for _, p in parameters)
    expert_shapes = [(name, list(p.shape)) for name, p in parameters if name.endswith("experts.w13_weight")]
    assert expert_shapes
    fields = scheduler.vllm_config.model_config.hf_config.to_dict()
    experts = fields["num_local_experts"] if "num_local_experts" in fields else fields["num_experts"]
    assert all(shape[0] == experts // (args.tp if args.ep else 1) for _, shape in expert_shapes)
    block, hashed = resolve_kv_cache_block_sizes(scheduler.kv_cache_config, scheduler.vllm_config)
    config = copy.copy(scheduler.vllm_config)
    config.compilation_config = copy.copy(config.compilation_config)
    config.compilation_config.static_forward_context = {}
    configuration = pickle.dumps((config, scheduler.kv_cache_config, block, hashed))
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / f"rank-{rank}-scheduler.pkl").write_bytes(configuration)
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_schedule, original_replay = scheduler.schedule, torch.cuda.CUDAGraph.replay
    identities, records, current, replay_counts = {}, [], [], []
    def schedule(throttle_prefills: bool = False):
        output = original_schedule(throttle_prefills)
        if output.num_scheduled_tokens:
            current.append({"items": items(scheduler, output, identities)})
        return output
    def graph_replay(instance):
        replay_counts.append(1)
        return original_replay(instance)
    scheduler.schedule = schedule
    torch.cuda.CUDAGraph.replay = graph_replay
    for case in CASES:
        current.clear(); identities.clear()
        submitted, generated, token_times, arrivals, aborted = set(), {}, {}, [], False
        start = time.perf_counter_ns()
        initial_replays = len(replay_counts)
        while len(submitted) < len(case.requests) or engine.has_unfinished_requests():
            step = len(current)
            for index, request in enumerate(case.requests):
                if index not in submitted and request.release_step <= step:
                    rid = str(index)
                    internal = engine.add_request(rid, {"prompt_token_ids": [100 + index % 8] * request.prompt},
                        SamplingParams(temperature=0, ignore_eos=True, max_tokens=request.output, detokenize=False,
                                       output_kind=RequestOutputKind.DELTA))
                    identities[internal] = rid
                    submitted.add(index)
                    arrivals.append({"id": rid, "step": step, "time_ns": time.perf_counter_ns() - start})
            before = time.perf_counter_ns()
            outputs = engine.step()
            torch.cuda.synchronize()
            after = time.perf_counter_ns()
            assert len(current) == step + 1 and current[-1]["items"]
            samples = {identities[rid]: [] for rid in scheduler.requests if rid in identities}
            for output in outputs:
                values = list(output.outputs[0].token_ids)
                generated.setdefault(output.request_id, []).extend(values)
                token_times.setdefault(output.request_id, []).extend([after - start] * len(values))
                samples[output.request_id] = values
            current[-1].update(duration_ns=after - before, samples=samples)
            if case.abort_after and len(current) == case.abort_after:
                engine.abort_request(["0"]); aborted = True
            assert len(current) <= sum(r.prompt + r.output for r in case.requests)
        for index, request in enumerate(case.requests):
            count = len(generated.get(str(index), []))
            assert count <= request.output if aborted and index == 0 else count == request.output
        records.append({"case": asdict(case), "steps": list(current), "arrivals": arrivals,
                        "token_ids": generated, "token_times_ns": token_times,
                        "wall_ns": time.perf_counter_ns() - start,
                        "graph_replays": len(replay_counts) - initial_replays})
        print(f"PASS TP{args.tp} EP={args.ep} {args.variant} {case.name}: {len(current)} steps", flush=True)
    assert bool(replay_counts) == graph
    scheduler.schedule, torch.cuda.CUDAGraph.replay = original_schedule, original_replay
    examples = llm.chat([[{"role": "user", "content": text}] for text in
                         ("Reply yes or no: Is 123 a prime number?", "Reply with a single word: What is the capital of France?")],
                        SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
    assert all(item.outputs[0].text.strip() for item in examples)
    document = {"variant": args.variant, "tp": args.tp, "ep": args.ep, "records": records,
                "examples": [{"prompt_token_ids": e.prompt_token_ids, "text": e.outputs[0].text,
                              "token_ids": list(e.outputs[0].token_ids)} for e in examples],
                "layers": len(layers), "experts": experts, "expert_shapes": expert_shapes,
                "parameter_bytes": sum(p.numel() * p.element_size() for _, p in parameters),
                "parameter_dtypes": sorted({str(p.dtype) for _, p in parameters}),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "gpu_uuid": str(torch.cuda.get_device_properties(rank).uuid),
                "source_sha256": source, "configuration_sha256": hashlib.sha256(configuration).hexdigest(),
                "torch": torch.__version__, "pid": os.getpid(), "complete_process_work_ns": time.perf_counter_ns() - began,
                "identity": {"python": sys.executable,
                    "packages": {p: importlib.metadata.version(p) for p in ("vllm", "torch", "triton", "transformers")},
                    "sources": {inspect.getfile(cls): hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                                for cls in (type(model), type(runner), type(scheduler))},
                    "loaded_libraries": loaded_libraries(),
                    "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                    "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}},
                "scope": "native_complete_checkpoint_functional; shared_host; no_speedup_qualification"}
    if args.tp > 1:
        digests = [""] * args.tp
        payload = {"records": [(r["case"], r["token_ids"], [(s["items"], s["samples"]) for s in r["steps"]])
                               for r in records], "examples": document["examples"]}
        torch.distributed.all_gather_object(digests, hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                                             group=get_world_group().cpu_group)
        assert len(set(digests)) == 1
    (args.evidence / f"rank-{rank}-native.json").write_text(json.dumps(document, sort_keys=True))
    if args.tp > 1:
        torch.distributed.barrier(group=get_world_group().cpu_group)
    core.shutdown()


def replay(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    import torch
    from vllm import SamplingParams
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from adapters.vllm.worker import make_scheduler
    from adapters.common.control import ControlLedger, ControlValue, DataKind
    records = json.loads((args.evidence / f"rank-{args.rank}-native.json").read_text())
    config_data = (args.evidence / f"rank-{args.rank}-scheduler.pkl").read_bytes()
    assert hashlib.sha256(config_data).hexdigest() == args.config_sha256 == records["configuration_sha256"]
    ledger, targets = ControlLedger(), []
    for record in records["records"]:
        scheduler = make_scheduler(pickle.loads(config_data), allow_chunked=True)
        case = record["case"]
        for step, expected in enumerate(record["steps"]):
            for arrival in record["arrivals"]:
                if arrival["step"] == step:
                    rid = arrival["id"]; request = case["requests"][int(rid)]
                    scheduler.add_request(Request(rid, [100 + int(rid) % 8] * request["prompt"],
                        SamplingParams(temperature=0, ignore_eos=True, max_tokens=request["output"], detokenize=False), None))
            output = scheduler.schedule()
            identities = {rid: rid for rid in output.num_scheduled_tokens}
            assert items(scheduler, output, identities) == expected["items"]
            ids, values = list(output.num_scheduled_tokens), []
            for rid in ids:
                key = f"{case['name']}/{step}/{rid}"
                ledger.produce(key, ControlValue("recorded_native_sampler", DataKind.HOST_VALID, tuple(expected["samples"].get(rid, []))))
                values.append(list(ledger.read(key, "vllm.Scheduler.update_from_output")))
            scheduler.update_from_output(output, ModelRunnerOutput(ids, {rid: i for i, rid in enumerate(ids)}, values))
            if case["abort_after"] == step + 1:
                scheduler.finish_requests(["0"], RequestStatus.FINISHED_ABORTED)
            targets.append((expected["items"], values))
        assert not scheduler.has_unfinished_requests()
    assert not torch.cuda.is_initialized()
    result = {"cases": len(records["records"]), "steps": len(targets), "control_reads": len(ledger.reads),
              "target_sha256": hashlib.sha256(json.dumps(targets, sort_keys=True).encode()).hexdigest(),
              "scope": "original_scheduler_recorded_native_control_replay; not_accuracy_prediction", "cuda_initialized": False}
    (args.evidence / f"rank-{args.rank}-replay.json").write_text(json.dumps(result, sort_keys=True))
    print("PASS original scheduler native control, mixed/chunk counts, cancellation and natural drain", result)


def reference(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                attn_implementation="eager", local_files_only=True).eval()
    examples = []
    for prompt in ("Reply yes or no: Is 123 a prime number?", "Reply with a single word: What is the capital of France?"):
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                              add_generation_prompt=True)["input_ids"]
        with torch.inference_mode():
            output = model.generate(torch.tensor([tokens]), attention_mask=torch.ones(1, len(tokens), dtype=torch.long),
                                    do_sample=False, max_new_tokens=32, use_cache=True)
        generated = output[0, len(tokens):].tolist()
        examples.append({"prompt_token_ids": tokens, "token_ids": generated,
                         "text": tokenizer.decode(generated, skip_special_tokens=True)})
    assert not torch.cuda.is_initialized()
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / "cpu-reference.json").write_text(json.dumps({"examples": examples,
        "parameters": sum(p.numel() for p in model.parameters()), "dtype": "bfloat16", "torch": torch.__version__,
        "scope": "independent_transformers_complete_checkpoint_CPU_reference", "cuda_initialized": False}, sort_keys=True))
    print("PASS independent complete-checkpoint CPU reference", [e["text"] for e in examples], flush=True)


def verify(args) -> None:
    documents = {variant: json.loads((args.evidence / variant / f"rank-{args.rank}-native.json").read_text())
                 for variant in ("eager", "graph", "chunk", "graph-chunk")}
    reference_path = args.evidence / "cpu-reference.json"
    baseline = documents["eager"]
    differences, matched_differences, natural_differences, replay_differences = [], [], [], []
    for variant, document in documents.items():
        assert [record["case"]["name"] for record in document["records"]] == [case.name for case in CASES]
        assert (document["tp"], document["ep"]) == (args.tp, args.ep)
        assert document["source_sha256"] == baseline["source_sha256"]
        assert document["identity"]["model_config_sha256"] == baseline["identity"]["model_config_sha256"]
        assert document["parameter_bytes"] == baseline["parameter_bytes"]
        matched = documents["chunk"] if "chunk" in variant else baseline
        for expected, observed in zip(matched["records"], document["records"], strict=True):
            assert expected["case"] == observed["case"]
            assert [step["items"] for step in expected["steps"]] == [step["items"] for step in observed["steps"]]
            for index in range(len(expected["case"]["requests"])):
                rid = str(index)
                left, right = expected["token_ids"].get(rid, []), observed["token_ids"].get(rid, [])
                if left != right:
                    index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
                    matched_differences.append({"variant": variant, "case": expected["case"]["name"], "request": rid,
                        "first_different_token": index, "expected": left[index:], "observed": right[index:]})
        for example, (expected, observed) in enumerate(zip(baseline["examples"], document["examples"], strict=True)):
            assert expected["prompt_token_ids"] == observed["prompt_token_ids"]
            if expected != observed:
                natural_differences.append({"variant": variant, "example": example,
                                            "expected": expected, "observed": observed})
        replay_result = json.loads((args.evidence / variant / f"rank-{args.rank}-replay.json").read_text())
        targets = [(step["items"], [step["samples"].get(item["id"], []) for item in step["items"]])
                   for record in document["records"] for step in record["steps"]]
        expected_replay = {"cases": len(CASES), "steps": len(targets),
                           "control_reads": sum(len(items) for items, _ in targets),
                           "target_sha256": hashlib.sha256(json.dumps(targets, sort_keys=True).encode()).hexdigest(),
                           "cuda_initialized": False}
        for field, expected in expected_replay.items():
            if replay_result[field] != expected:
                replay_differences.append({"variant": variant, "field": field,
                                           "expected": expected, "observed": replay_result[field]})
        if "chunk" in variant:
            assert any(0 < item["computed_before"] < item["prompt"]
                       for record in document["records"] for step in record["steps"] for item in step["items"])
            cancelled = next(r for r in document["records"] if r["case"]["name"] == "cancel-prefill")
            assert not cancelled["token_ids"].get("0", [])
        if "graph" in variant:
            assert sum(r["graph_replays"] for r in document["records"]) > 0
    for expected, observed in zip(baseline["records"], documents["chunk"]["records"], strict=True):
        for rid, left in expected["token_ids"].items():
            right = observed["token_ids"].get(rid, [])
            for index, (a, b) in enumerate(zip(left, right)):
                if a != b:
                    differences.append({"case": expected["case"]["name"], "request": rid,
                                        "first_different_token": index, "eager": a, "chunk": b})
                    break
    reference_differences = []
    if reference_path.exists():
        reference_document = json.loads(reference_path.read_text())
        assert not reference_document["cuda_initialized"]
        for example, (expected, observed) in enumerate(zip(reference_document["examples"], baseline["examples"], strict=True)):
            assert expected["prompt_token_ids"] == observed["prompt_token_ids"]
            left, right = expected["token_ids"], observed["token_ids"]
            if left != right:
                index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
                reference_differences.append({"example": example, "first_different_token": index,
                                              "reference": left[index:], "native": right[index:]})
    (args.evidence / "comparison.json").write_text(json.dumps({"graph_matched_policy_exact": not matched_differences,
        "original_scheduler_replay_verified": not replay_differences,
        "replay_integrity_differences": replay_differences,
        "matched_policy_differences": matched_differences,
        "natural_examples_all_variants_exact": not natural_differences,
        "natural_example_differences": natural_differences, "independent_reference_present": reference_path.exists(),
        "independent_reference_verified": reference_path.exists() and not reference_differences,
        "independent_reference_differences": reference_differences,
        "cross_prefill_policy_differences": differences}, sort_keys=True))
    print("Cross-prefill-policy exact-token differences:", differences)
    assert not replay_differences, replay_differences
    assert not matched_differences, matched_differences
    assert not natural_differences, natural_differences
    assert not reference_differences, reference_differences
    print("PASS matched-policy Graph parity and original scheduler replay")


def matrix(args) -> None:
    """Finite independent native processes followed by original CPU rank replay."""
    args.evidence.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    source = hashlib.sha256(script.read_bytes()).hexdigest()
    rows = []
    for variant in ("eager", "graph", "chunk", "graph-chunk"):
        assert hashlib.sha256(script.read_bytes()).hexdigest() == source
        target = args.evidence / variant
        target.mkdir(exist_ok=True)
        command = [sys.executable, str(script), "native", "--model", str(args.model),
                   "--evidence", str(target), "--variant", variant, "--tp", str(args.tp),
                   "--kv-bytes", str(args.kv_bytes), "--memory-fraction", str(args.memory_fraction),
                   "--attention-backend", args.attention_backend]
        if args.ep:
            command.append("--ep")
        if args.trust_model_code:
            command.append("--trust-model-code")
        start = time.perf_counter_ns()
        with (target / "native.log").open("w") as output:
            status = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
        row = {"variant": variant, "native_status": status, "wall_ns": time.perf_counter_ns() - start}
        rows.append(row)
        for rank in range(args.tp) if status == 0 else ():
            config = (target / f"rank-{rank}-scheduler.pkl").read_bytes()
            command = [sys.executable, str(script), "replay", "--evidence", str(target),
                       "--rank", str(rank), "--config-sha256", hashlib.sha256(config).hexdigest()]
            with (target / f"replay-{rank}.log").open("w") as output:
                status = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                        stdout=output, stderr=subprocess.STDOUT).returncode
            row[f"replay_{rank}_status"] = status
            if status:
                break
        (args.evidence / "matrix-processes.json").write_text(json.dumps({"source_sha256": source,
            "pid": os.getpid(), "rows": rows}, sort_keys=True))
        print(row, flush=True)
        if status:
            raise SystemExit(status)
    for rank in range(args.tp):
        args.rank = rank
        verify(args)
    for variant in ("eager", "graph", "chunk", "graph-chunk"):
        ranks = [json.loads((args.evidence / variant / f"rank-{rank}-replay.json").read_text())
                 for rank in range(args.tp)]
        assert len({r["target_sha256"] for r in ranks}) == 1
    print("PASS finite native/CPU matrix and cross-rank replay targets")


def reference_trace(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(8)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                attn_implementation="eager", local_files_only=True).eval()
    rows = []
    comparison = json.loads((args.evidence / "comparison.json").read_text())
    for difference in comparison["cross_prefill_policy_differences"]:
        for variant in ("eager", "chunk"):
            document = json.loads((args.evidence / variant / "rank-0-native.json").read_text())
            record = next(r for r in document["records"] if r["case"]["name"] == difference["case"])
            rid = difference["request"]
            prompt = [100 + int(rid) % 8] * record["case"]["requests"][int(rid)]["prompt"]
            generated = record["token_ids"][rid]
            index = difference["first_different_token"]
            sequence = prompt + generated[:index]
            with torch.inference_mode():
                logits = model(torch.tensor([sequence]), use_cache=False).logits[0, -1].float()
            top = torch.topk(logits, 2)
            selected = generated[index]
            row = {**difference, "variant": variant, "selected": selected,
                   "cpu_top1": top.indices[0].item(), "cpu_top2": top.indices[1].item(),
                   "cpu_top1_minus_selected": (top.values[0] - logits[selected]).item(),
                   "cpu_top1_minus_top2": (top.values[0] - top.values[1]).item()}
            rows.append(row)
            print(row, flush=True)
    baseline = json.loads((args.evidence / "eager" / "rank-0-native.json").read_text())
    for difference in comparison.get("independent_reference_differences", []):
        example = baseline["examples"][difference["example"]]
        index = difference["first_different_token"]
        sequence = example["prompt_token_ids"] + example["token_ids"][:index]
        selected = example["token_ids"][index]
        with torch.inference_mode():
            logits = model(torch.tensor([sequence]), use_cache=False).logits[0, -1].float()
        top = torch.topk(logits, 2)
        row = {**difference, "variant": "native-reference", "selected": selected,
               "cpu_top1": top.indices[0].item(), "cpu_top2": top.indices[1].item(),
               "cpu_top1_minus_selected": (top.values[0] - logits[selected]).item(),
               "cpu_top1_minus_top2": (top.values[0] - top.values[1]).item()}
        rows.append(row)
        print(row, flush=True)
    assert not torch.cuda.is_initialized()
    (args.evidence / "cpu-trace-diagnostic.json").write_text(json.dumps({"rows": rows,
        "scope": "independent_CPU_teacher_forcing_diagnostic; not_native_logit_error", "cuda_initialized": False}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "replay", "reference", "verify", "reference-trace", "matrix"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--variant", choices=("eager", "graph", "chunk", "graph-chunk"), default="eager")
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ep", action="store_true")
    parser.add_argument("--kv-bytes", type=int, default=64 << 20)
    parser.add_argument("--memory-fraction", type=float, default=.04)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--trust-model-code", action="store_true")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--native-rank", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--config-sha256")
    args = parser.parse_args()
    {"native": native, "replay": replay, "reference": reference, "verify": verify,
     "reference-trace": reference_trace, "matrix": matrix}[args.mode](args)


if __name__ == "__main__":
    main()
