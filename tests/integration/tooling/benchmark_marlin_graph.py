"""Real Marlin W4A16 correctness and eager/Graph timing; no model checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repetitions", type=int, default=100)
args = parser.parse_args()
assert args.repetitions > 0
torch.manual_seed(20260929)
results = []
for m in (1, 16, 64):
    k, n = 1024, 2048
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(k, n, device="cuda", dtype=torch.float16) / k**0.5
    ref_weight, packed, scales, g_idx, perm, _ = marlin_quantize(
        weight, scalar_types.uint4b8, 128, False)
    workspace = marlin_make_workspace_new(a.device)
    output = torch.empty(m, n, device="cuda", dtype=torch.float16)

    def run() -> torch.Tensor:
        return ops.marlin_gemm(a, output, packed, None, scales, None, None,
                               None, g_idx, perm, workspace, scalar_types.uint4b8,
                               m, n, k, use_fp32_reduce=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            run()
    stream.synchronize()
    reference = a @ ref_weight
    torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for factor in (1.0, -0.5, 2.0):
        a.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, a @ ref_weight, atol=0.02, rtol=0.02)

    arms = []
    for block, order in enumerate(("APPA", "PAAP", "APPA", "PAAP")):
        for position, arm in enumerate(order):
            operation = run if arm == "A" else graph.replay
            for _ in range(5):
                operation()
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            for _ in range(args.repetitions):
                operation()
            end.record()
            end.synchronize()
            wall_us = (time.perf_counter() - start) * 1e6 / args.repetitions
            arms.append({"block": block, "position": position, "arm": arm,
                         "device_us": begin.elapsed_time(end) * 1000 / args.repetitions,
                         "wall_us": wall_us})
    eager = statistics.median(row["wall_us"] for row in arms if row["arm"] == "A")
    replay = statistics.median(row["wall_us"] for row in arms if row["arm"] == "P")
    results.append({"m": m, "k": k, "n": n, "correctness": "PASS",
                    "eager_wall_us": eager, "graph_wall_us": replay,
                    "speedup": eager / replay, "arms": arms})
    del graph, output, workspace, packed, scales, g_idx, perm, ref_weight, reference, weight, a
    torch.cuda.empty_cache()

extension = Path(ops.__file__).parent / "_C_stable_libtorch.abi3.so"
report = {"scope": "real Marlin W4A16 op; not full-model E2E or simulator prediction",
          "torch": torch.__version__, "cuda": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(), "repetitions": args.repetitions,
          "extension_sha256": hashlib.file_digest(extension.open("rb"), "sha256").hexdigest(),
          "results": results}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
