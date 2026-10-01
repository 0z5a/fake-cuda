"""Native M1 stages. Raw evidence goes outside git; never preload fake CUDA."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.models import lower
from sim.work import KDA, MoE, TopK, Work


class NativeMoE:
    """Recorded routes, explicit per-expert padding, three gated expert matrices."""
    def __init__(self, torch, n: int, routing: str):
        self.t = torch
        self.n, self.d, self.e, self.k, self.m, self.tile = n, 128, 256, 8, 128, 4
        self.x = torch.randn(n, self.d, device="cuda", dtype=torch.bfloat16) * .1
        self.router_w = torch.randn(self.d, self.e, device="cuda", dtype=torch.bfloat16) / math.sqrt(self.d)
        self.gu = torch.randn(self.e, self.d, 2 * self.m, device="cuda", dtype=torch.bfloat16) / math.sqrt(self.d)
        self.dw = torch.randn(self.e, self.m, self.d, device="cuda", dtype=torch.bfloat16) / math.sqrt(self.m)
        self.bias = torch.zeros(self.e, device="cuda")
        if routing == "hot":
            self.bias[:64] = 5
        self.router(); self.score(); self.topk()
        recorded = tuple(tuple(row) for row in self.experts.cpu().tolist())
        self.spec = MoE(n, self.d, self.e, self.k, self.m, recorded, tile_m=self.tile, routing="recorded")
        self.slices = []
        offset = 0
        self.inputs = []
        for expert, count in enumerate(self.spec.counts):
            if count:
                padded = ((count + self.tile - 1) // self.tile) * self.tile
                self.slices.append((expert, offset, count))
                self.inputs.append(torch.zeros(padded, self.d, device="cuda", dtype=torch.bfloat16))
                offset += count
        self.ones = torch.ones(n * self.k, device="cuda", dtype=torch.int64)
        self.stage_names = ("router", "score", "topk", "route", "permute", "gate_up", "activation", "down", "combine")
        self.calls = (self.router, self.score, self.topk, self.route, self.permute, self.gate_up, self.activation, self.down, self.combine)

    def router(self):
        self.scores = self.x @ self.router_w

    def score(self):
        self.probs = self.t.softmax(self.scores.float() + self.bias, dim=-1)

    def topk(self):
        self.weights, self.experts = self.t.topk(self.probs, self.k, dim=-1, sorted=True)
        self.weights = self.weights / self.weights.sum(-1, keepdim=True)

    def route(self):
        flat = self.experts.flatten()
        self.order = self.t.argsort(flat, stable=True)
        self.counts = self.t.zeros(self.e, device="cuda", dtype=self.t.int64).scatter_add_(0, flat, self.ones)
        self.offsets = self.counts.cumsum(0)

    def permute(self):
        self.sorted_x = self.x.repeat_interleave(self.k, dim=0)[self.order]
        for buffer, (_, offset, count) in zip(self.inputs, self.slices):
            buffer[:count].copy_(self.sorted_x[offset:offset + count])

    def gate_up(self):
        self.gated = [buffer @ self.gu[expert] for buffer, (expert, _, _) in zip(self.inputs, self.slices)]

    def activation(self):
        self.activated = [self.t.nn.functional.silu(value[:, :self.m]) * value[:, self.m:] for value in self.gated]

    def down(self):
        self.outputs = [value @ self.dw[expert] for value, (expert, _, _) in zip(self.activated, self.slices)]

    def combine(self):
        values = self.t.cat([value[:count] for value, (_, _, count) in zip(self.outputs, self.slices)])
        values = values * self.weights.flatten()[self.order, None]
        self.output = self.t.zeros(self.n, self.d, device="cuda", dtype=self.t.float32).index_add_(0, self.order // self.k, values.float())

    def full(self):
        for call in self.calls:
            call()

    def check(self):
        self.full()
        self.t.testing.assert_close(self.counts.cpu(), self.t.tensor(self.spec.counts), atol=0, rtol=0)
        x, gu, dw, weights = self.x.cpu().float(), self.gu.cpu().float(), self.dw.cpu().float(), self.weights.cpu()
        reference = self.t.zeros(self.n, self.d)
        for token, row in enumerate(self.spec.routes):
            for slot, expert in enumerate(row):
                value = x[token] @ gu[expert]
                value = self.t.nn.functional.silu(value[:self.m]) * value[self.m:]
                reference[token] += weights[token, slot] * (value @ dw[expert])
        self.t.testing.assert_close(self.output.cpu(), reference, atol=2e-3, rtol=3e-2)


class NativeTopK:
    def __init__(self, torch, n: int, routing: str):
        self.t = torch
        self.spec = TopK(n, 256, 8)
        self.scores = torch.randn(n, 256, device="cuda")
        self.stage_names, self.calls = ("topk",), (self.full,)

    def full(self):
        self.values, self.indices = self.t.topk(self.scores, 8, dim=-1, sorted=True)

    def check(self):
        self.full()
        values, indices = self.t.topk(self.scores.cpu(), 8, dim=-1, sorted=True)
        self.t.testing.assert_close(self.values.cpu(), values, atol=0, rtol=0)
        self.t.testing.assert_close(self.indices.cpu(), indices, atol=0, rtol=0)
        for e, k in ((257, 257), (257, 1), (256, 8)):
            scores = self.t.ones(2, e, device="cuda")
            selected, indices = self.t.topk(scores, k, sorted=True)
            self.t.testing.assert_close(selected.cpu(), self.t.ones(2, k), atol=0, rtol=0)
            assert all(len(set(row)) == k for row in indices.cpu().tolist())


class NativeKDA:
    def __init__(self, torch, n: int, routing: str):
        from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
        self.t, self.fused = torch, fused_recurrent_kda_fwd
        self.b, self.d, self.h, self.dk = n, 2304, 32, 128
        self.spec = KDA(n, self.d, self.h, self.h, self.dk, self.dk, tuple(range(n)), (2048,) * n)
        width = self.h * self.dk
        self.x = torch.randn(n, self.d, device="cuda", dtype=torch.bfloat16) * .1
        self.qkv_w = torch.randn(self.d, 3 * width, device="cuda", dtype=torch.bfloat16) / math.sqrt(self.d)
        self.o_w = torch.randn(width, self.d, device="cuda", dtype=torch.bfloat16) / math.sqrt(width)
        self.conv_w = torch.randn(4, 3 * width, device="cuda", dtype=torch.bfloat16) * .1
        self.conv_state = torch.zeros(n, 3, 3 * width, device="cuda", dtype=torch.bfloat16)
        self.state = torch.randn(n, self.h, self.dk, self.dk, device="cuda", dtype=torch.float32) * .01
        self.slots = torch.arange(n, device="cuda", dtype=torch.int32)
        self.stage_names = ("qkv", "gates", "core", "out_gate", "o_proj")
        self.calls = (self.qkv, self.gates, self.core, self.out_gate, self.o_proj)

    def qkv(self):
        self.projected = self.x @ self.qkv_w

    def gates(self):
        window = self.t.cat((self.conv_state, self.projected[:, None]), dim=1)
        self.conv_state.copy_(window[:, 1:])
        projected = self.t.nn.functional.silu((window * self.conv_w).sum(dim=1))
        q, k, v = projected.reshape(self.b, 1, 3, self.h, self.dk).unbind(2)
        self.q = self.t.nn.functional.normalize(q.float(), dim=-1).to(self.t.bfloat16)
        self.k = self.t.nn.functional.normalize(k.float(), dim=-1).to(self.t.bfloat16)
        self.v = v.contiguous()
        self.g = self.t.full_like(self.q, math.log(.9))
        self.beta = self.t.full((self.b, 1, self.h), .1, device="cuda", dtype=self.t.bfloat16)

    def core(self):
        self.attended, _ = self.fused(self.q, self.k, self.v, self.g, self.beta, scale=1.,
                                      initial_state=self.state, output_final_state=True,
                                      inplace_final_state=True, ssm_state_indices=self.slots)

    def out_gate(self):
        value = self.attended.float()
        value = value * self.t.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5)
        self.gated_output = (value * self.t.sigmoid(self.q.float())).reshape(self.b, -1).to(self.t.bfloat16)

    def o_proj(self):
        self.output = self.gated_output @ self.o_w

    def full(self):
        for call in self.calls:
            call()

    def check(self):
        self.qkv(); self.gates()
        state = self.state.clone()
        q, k, v = self.q.float()[:, 0], self.k.float()[:, 0], self.v.float()[:, 0]
        decayed = state * self.g.float()[:, 0].exp()[..., None]
        residual = (v - (decayed * k[..., None]).sum(-2)) * self.beta.float()[:, 0, :, None]
        expected = decayed + k[..., None] * residual[..., None, :]
        output = (expected * q[..., None]).sum(-2)
        self.core()
        self.t.testing.assert_close(self.state, expected, atol=2e-5, rtol=2e-3)
        self.t.testing.assert_close(self.attended.float()[:, 0], output, atol=2e-4, rtol=2e-2)
        self.state.copy_(state)
        # Permuting slots must update the selected storage rows, not row=token.
        self.slots = self.slots.flip(0)
        selected = state[self.slots.long()]
        decayed = selected * self.g.float()[:, 0].exp()[..., None]
        residual = (v - (decayed * k[..., None]).sum(-2)) * self.beta.float()[:, 0, :, None]
        expected = decayed + k[..., None] * residual[..., None, :]
        self.core()
        self.t.testing.assert_close(self.state[self.slots.long()], expected, atol=2e-5, rtol=2e-3)
        self.slots = self.slots.flip(0)
        self.state.copy_(state)
        self.full()
        assert self.t.isfinite(self.output).all().item()


def measure(torch, call: Callable[[], None], iterations: int, mode: str) -> dict[str, object]:
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        invoke = graph.replay
    else:
        invoke = call
    for _ in range(5):
        invoke()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
    for start, end in events:
        start.record(); end.record()
    torch.cuda.synchronize()
    # Queue the short graphs before their timing events execute. Otherwise an
    # idle GPU's event interval measures the Python submission gap as well.
    torch.cuda._sleep(50_000_000)
    host_ns = []
    for start, end in events:
        start.record()
        before = time.perf_counter_ns()
        invoke()
        host_ns.append(time.perf_counter_ns() - before)
        end.record()
    events[-1][1].synchronize()
    device_ns = [round(start.elapsed_time(end) * 1e6) for start, end in events]
    return {"device_ns": device_ns, "host_submission_ns": host_ns,
            "median_ns": round(statistics.median(device_ns)),
            "scope": "operator_stage_cuda_event_interval"}


def session(args) -> None:
    if os.environ.get("LD_PRELOAD") or "fakecuda" in os.environ.get("LD_LIBRARY_PATH", "").lower():
        raise ValueError("native measurement must not load the fake Driver")
    import torch
    import fla.ops.kda.fused_recurrent as kda_source
    torch.cuda.set_device(args.device)
    torch.set_num_threads(1)
    mapped = {Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
              if len(line.split()) >= 6 and any(name in line.split()[-1] for name in ('libtorch_cuda.so', 'libcuda.so', 'libcublas.so', 'libcublasLt.so'))}
    source_files = (Path(__file__), Path(kda_source.__file__), Path(torch.__file__), *sorted(mapped))
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    versions = {name: metadata.version(name) for name in ("torch", "triton", "fla-core", "einops")}
    revision = hashlib.sha256(json.dumps({"sources": sources, "versions": versions}, sort_keys=True).encode()).hexdigest()
    identity = {"hardware": str(torch.cuda.get_device_properties(args.device).uuid),
                "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[args.device],
                "backend": "native_m1", "revision": revision, "mode": args.mode, "cache": "measured-steady"}
    constructors = {"moe": NativeMoE, "topk": NativeTopK, "kda": NativeKDA}
    rows = []
    for n in map(int, args.batches.split(",")):
        torch.manual_seed(1000 * args.session + n + (10000 if args.split == "validation" else 0))
        native = constructors[args.family](torch, n, args.routing)
        native.check()
        variants = ()
        if args.family == "moe":
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as diagnostic:
                native.route()
            names = tuple(event.name for event in diagnostic.events() if event.device_type == torch.autograd.DeviceType.CUDA)
            if not names:
                raise ValueError("route compiled-path launch trace is unavailable")
            variants = (("route", hashlib.sha256(json.dumps(names).encode()).hexdigest()),)
        for _ in range(args.warmup):
            native.full()
        torch.cuda.synchronize()
        op = Work(f"{args.family}-{n}-{args.routing}", native.spec, identity["backend"], revision,
                  graph_id="native-fixed-address" if args.mode == "graph" else "", variants=variants)
        lowered = lower(op)
        assert native.stage_names == tuple(stage.name for stage in lowered.stages)
        measurements = {name: measure(torch, call, args.iterations, args.mode) for name, call in zip(native.stage_names, native.calls)}
        measurements["whole"] = measure(torch, native.full, args.iterations, args.mode)
        row = {"descriptor": op.dumps(), "stages": [asdict(stage) for stage in lowered.stages], "measurements": measurements,
               "compiled_route_launches": names if args.family == "moe" else (),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        rows.append(row)
        print(args.family, args.split, args.routing, args.session, n, measurements["whole"]["median_ns"], flush=True)
        del native
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {"schema": 1, "identity": identity, "versions": versions, "sources": sources,
                "split": args.split, "routing": args.routing, "session": args.session,
                "warmup": args.warmup, "iterations": args.iterations, "queue_prefill_cycles": 50_000_000,
                "pid": os.getpid(), "python": sys.executable, "torch_git": torch.version.git_version,
                "correctness": "passed", "rows": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.family}-{args.split}-{args.routing}-{args.mode}-{args.session}.json").write_text(json.dumps(manifest, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("moe", "topk", "kda"), required=True)
    parser.add_argument("--batches", default="1,8,32,64,128")
    parser.add_argument("--routing", choices=("uniform", "hot"), default="uniform")
    parser.add_argument("--split", choices=("calibration", "validation"), required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = args.output.parent / "compile-cache"
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ["CUDA_CACHE_PATH"] = str(cache / "cuda")
    if min(args.warmup, args.iterations, args.sessions) < 1:
        parser.error("positive warmup, iterations and sessions required")
    if args.session:
        session(args)
    else:
        for number in range(1, args.sessions + 1):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--session", str(number)], check=True)


if __name__ == "__main__":
    main()
