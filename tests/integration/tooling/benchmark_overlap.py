"""Independent native P2P/compute pairing; frozen discrete concurrency costs."""
import argparse
import ctypes
from dataclasses import asdict, replace
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.cost import Identity, Profile
from sim.serde import read_profiles
from benchmark_semantic import NativeKDA
from evaluate_semantic import nonnegative_fit, percentile


def collect(args) -> None:
    if os.environ.get("LD_PRELOAD") or "fakecuda" in os.environ.get("LD_LIBRARY_PATH", "").lower():
        raise ValueError("native pairing must not use the fake Driver")
    import torch
    if torch.cuda.device_count() != 2 or not torch.cuda.can_device_access_peer(0, 1):
        raise ValueError("pairing needs the two explicitly visible peer-accessible devices")
    torch.set_num_threads(1)
    torch.manual_seed(args.session + (10000 if args.split == "validation" else 0))
    source = torch.full((args.copy_bytes // 4,), .125, dtype=torch.float32, device="cuda:0")
    torch.cuda.set_device(1)
    destination = torch.empty_like(source, device="cuda:1")
    destination.copy_(source)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    torch.testing.assert_close(destination.cpu(), source.cpu(), rtol=0, atol=0)
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxEnablePeerAccess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    driver.cuMemcpyDtoDAsync_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
    contexts = []
    for device in (0, 1):
        torch.cuda.set_device(device)
        context = ctypes.c_void_p()
        if driver.cuCtxGetCurrent(ctypes.byref(context)) or not context.value:
            raise RuntimeError("native peer-copy context query failed")
        contexts.append(context)
    code = driver.cuCtxEnablePeerAccess(contexts[0], 0)
    if code not in (0, 704):  # CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED is idempotent.
        raise RuntimeError(f"native peer mapping failed: CUDA {code}")
    import fla.ops.kda.fused_recurrent as kda_source
    copy_stream, compute_stream = torch.cuda.Stream(), torch.cuda.Stream()
    copy_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(copy_graph, stream=copy_stream):
        for _ in range(args.copies):
            code = driver.cuMemcpyDtoDAsync_v2(destination.data_ptr(), source.data_ptr(), args.copy_bytes, copy_stream.cuda_stream)
            if code:
                raise RuntimeError(f"native peer-copy capture failed: CUDA {code}")
    if args.family == "gemm":
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16) * .01
        w = torch.randn_like(x)
        output = torch.empty_like(x)
        def kernel():
            torch.mm(x, w, out=output)
        kernel()
        torch.testing.assert_close(output.float(), x.float() @ w.float(), atol=3e-4, rtol=3e-2)
    else:
        native = NativeKDA(torch, 64, "uniform")
        native.check(); native.qkv(); native.gates()
        kernel = native.core
    uuids = tuple(str(torch.cuda.get_device_properties(i).uuid) for i in range(2))
    def background() -> tuple[str, ...]:
        rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader"], text=True).splitlines()
        others = tuple(sorted(row for row in rows if row.split(",")[0].strip() in uuids and int(row.split(",")[1]) != os.getpid()))
        if any(int(row.split(",")[1]) not in args.allow_resident_pid for row in others):
            raise ValueError("unexpected GPU workload during native pairing")
        return others
    before = background()
    mapped = {Path(row.split()[-1]) for row in Path('/proc/self/maps').read_text().splitlines()
              if len(row.split()) >= 6 and any(name in row.split()[-1] for name in ('libtorch_cuda.so', 'libcuda.so', 'libcublas.so', 'libcublasLt.so'))}
    files = (Path(__file__), Path(__file__).with_name("benchmark_semantic.py"), Path(kda_source.__file__), *sorted(mapped))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    versions = {name: metadata.version(name) for name in ("torch", "triton", "fla-core")}
    revision = hashlib.sha256(json.dumps((hashes, versions, before), sort_keys=True).encode()).hexdigest()
    identity = Identity("->".join(uuids), subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0],
                        "peer_copy_compute_event_pair", revision, "graph", "measured-steady")
    family = (args.family, "source0_destination1", str(args.copy_bytes), str(args.copies), "gemm2048" if args.family == "gemm" else "KDA64x32x128x128_fp32_state")
    records = []
    for repeats in map(int, args.repeats.split(",")):
        if repeats <= 0:
            raise ValueError("positive compute repetitions required")
        graph = torch.cuda.CUDAGraph()
        for _ in range(3):
            kernel()
        with torch.cuda.graph(graph, stream=compute_stream):
            for _ in range(repeats):
                kernel()
        def enqueue(mode, start, copy_end, compute_end, end):
            start.record()
            if mode != "compute":
                copy_stream.wait_event(start)
                with torch.cuda.stream(copy_stream):
                    copy_graph.replay(); copy_end.record()
                torch.cuda.current_stream().wait_event(copy_end)
            if mode != "copy":
                compute_stream.wait_event(start)
                with torch.cuda.stream(compute_stream):
                    graph.replay(); compute_end.record()
                torch.cuda.current_stream().wait_event(compute_end)
            end.record()
        measurements = {}
        for mode in ("copy", "compute", "pair"):
            events = [tuple(torch.cuda.Event(enable_timing=True) for _ in range(4)) for _ in range(args.iterations)]
            warm = tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
            for _ in range(args.warmup):
                enqueue(mode, *warm)
            torch.cuda.synchronize()
            torch.cuda._sleep(50_000_000)
            for event in events:
                enqueue(mode, *event)
            events[-1][-1].synchronize()
            measurements[mode] = [round(event[0].elapsed_time(event[-1]) * 1e6) for event in events]
        after = background()
        if before != after:
            raise ValueError("native pairing background changed")
        records.append({"repeats": repeats, "measurements": measurements})
        print(args.family, args.split, args.session, repeats,
              {mode: round(statistics.median(values)) for mode, values in measurements.items()}, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    document = {"schema": 1, "identity": asdict(identity), "family": family, "split": args.split,
                "session": args.session, "warmup": args.warmup, "iterations": args.iterations,
                "background": before, "sources": hashes, "versions": versions, "records": records,
                "scope": "queued_native_CUDA_event_pair; P2P_copy_not_DeepEP; no_NCU_DRAM_inference"}
    (args.output / f"{args.family}-{args.split}-{args.session}.json").write_text(json.dumps(document, sort_keys=True))


def freeze(inputs: list[Path], output: Path) -> None:
    profiles, hashes, identities = [], {}, set()
    groups = {}
    for path in inputs:
        data = path.read_bytes(); document = json.loads(data)
        if document["split"] != "calibration":
            raise ValueError("only calibration may fit concurrency profiles")
        identity = Identity(**document["identity"]); identities.add(identity)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        for row in document["records"]:
            for mode, times in row["measurements"].items():
                groups.setdefault((identity, mode, tuple(document["family"])), {}).setdefault((row["repeats"],), []).append(round(statistics.median(times)))
    if len(identities) != 1:
        raise ValueError("concurrency input identities differ")
    for (identity, mode, family), samples in groups.items():
        keys = sorted(samples); values = [round(statistics.median(samples[k])) for k in keys]
        coefficients = nonnegative_fit(keys, values)
        profile_id = hashlib.sha256(json.dumps((asdict(identity), mode, family, keys, values), sort_keys=True).encode()).hexdigest()
        profiles.append(Profile(profile_id, identity, mode, family, tuple(zip(keys, values)), coefficients, keys[0], keys[-1]))
    output.write_text(json.dumps({"schema": 1, "identity": asdict(next(iter(identities))),
                                 "training_hashes": hashes, "profiles": [asdict(p) for p in profiles]}, sort_keys=True))


def evaluate(args) -> None:
    identity, profiles = read_profiles(args.profiles)
    frozen = json.loads(args.profiles.read_text())
    hashes = set(frozen["training_hashes"].values())
    table = {(p.stage, p.family): p for p in profiles}
    records, sessions, shapes = [], set(), set()
    for path in args.input:
        document = json.loads(path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() in hashes or document["split"] != "validation" or Identity(**document["identity"]) != identity:
            raise ValueError("validation leakage or identity mismatch")
        if document["warmup"] < 30 or document["iterations"] < 200:
            raise ValueError("publication needs 30 warmups and 200 samples")
        sessions.add(document["session"])
        family = tuple(document["family"])
        for row in document["records"]:
            shapes.add((family, row["repeats"]))
            native = {mode: round(statistics.median(values)) for mode, values in row["measurements"].items()}
            for mode, ns in native.items():
                profile = table[mode, family]
                if any(features == (row["repeats"],) for features, _ in profile.samples):
                    raise ValueError("validation repeats were used in fitting")
                cost = profile.lookup((row["repeats"],), candidate=True)
                if cost.standalone_ns is None:
                    raise ValueError("unsupported concurrency query")
                records.append((family[0], row["repeats"], document["session"], mode, ns, cost.standalone_ns, abs(cost.standalone_ns/ns - 1),
                                (native["copy"] + native["compute"]) / native["pair"]))
    errors = [row[6] for row in records]
    gate = len(sessions) >= 5 and len(shapes) >= 3 and percentile(errors, .5) <= .15 and percentile(errors, .95) <= .30
    lines = ["# Native communication/compute concurrency", "", f"Frozen SHA256: `{hashlib.sha256(args.profiles.read_bytes()).hexdigest()}`.",
             f"Gate: **{'PASS' if gate else 'FAIL'}**, {len(shapes)} held-out configurations, {len(sessions)} sessions; P50/P95 error {percentile(errors,.5)*100:.2f}% / {percentile(errors,.95)*100:.2f}%.", "",
             "P2P copy and device computation use separate streams. CUDA-event windows include the common release/barrier and exclude interpreter/JIT time. The ratio is the sum of separately measured copy/compute windows divided by the paired window; no serialized arm was measured. Static resident leases are part of the retained background identity. This does not qualify NCCL/DeepEP, SM usage or DRAM counters.", "",
             "| Family | Repeats | Session | Phase | Native us | Predicted us | Error | Sum of standalone / paired |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {family} | {repeats} | {session} | {mode} | {native/1000:.3f} | {pred/1000:.3f} | {error*100:.2f}% | {ratio:.3f}x |"
              for family, repeats, session, mode, native, pred, error, ratio in records]
    args.output.write_text("\n".join(lines) + "\n")
    if not gate:
        raise SystemExit(1)
    if args.qualified_output:
        frozen["profiles"] = [asdict(replace(p, qualified=True)) for p in profiles]
        frozen["validation_hashes"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.input}
        args.qualified_output.write_text(json.dumps(frozen, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "freeze", "evaluate"))
    parser.add_argument("--family", choices=("gemm", "kda"), default="gemm")
    parser.add_argument("--split", choices=("calibration", "validation"), default="calibration")
    parser.add_argument("--repeats", default="1,4,8,16,32")
    parser.add_argument("--copy-bytes", type=int, default=128 * 1024**2)
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--allow-resident-pid", type=int, nargs="*", default=[])
    parser.add_argument("--input", type=Path, nargs="+", default=[])
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--qualified-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.copy_bytes, args.copies, args.warmup, args.iterations, args.sessions) <= 0 or args.copy_bytes % 4:
        parser.error("positive counts and a float32-aligned byte count required")
    if args.mode == "freeze":
        freeze(args.input, args.output)
    elif args.mode == "evaluate":
        if args.profiles is None:
            parser.error("evaluate requires --profiles")
        evaluate(args)
    elif args.session:
        os.environ["TRITON_CACHE_DIR"] = str(args.output.parent / "compile-cache" / "triton")
        os.environ["CUDA_CACHE_PATH"] = str(args.output.parent / "compile-cache" / "cuda")
        collect(args)
    else:
        for session in range(1, args.sessions + 1):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--session", str(session)], check=True)


if __name__ == "__main__":
    main()
