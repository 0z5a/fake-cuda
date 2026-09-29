"""Exercise an installed Marlin binary through the fake Driver, without values."""
import argparse
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("extension", type=Path)
args = parser.parse_args()
assert not list(Path("/dev").glob("nvidia*"))
torch.ops.load_library(str(args.extension))

m, k, n = 16, 1024, 2048
a = torch.empty((m, k), device="cuda", dtype=torch.float16)
output = torch.empty((m, n), device="cuda", dtype=torch.float16)
packed = torch.empty((k // 16, n * 2), device="cuda", dtype=torch.int32)
scales = torch.empty((k // 128, n), device="cuda", dtype=torch.float16)
empty = torch.empty(0, device="cuda", dtype=torch.int32)
workspace = torch.empty(torch.cuda.get_device_properties(0).multi_processor_count,
                        device="cuda", dtype=torch.int32)
# uint4b8 identifier from the installed vLLM ScalarType ABI, not a model value.
uint4b8 = 1125899907892224

def run() -> torch.Tensor:
    return torch.ops._C.marlin_gemm(a, output, packed, None, scales, None, None,
                                  None, empty, empty, workspace, uint4b8,
                                  m, n, k, True, False, True, False)

stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    run()
stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    run()
for _ in range(3):
    graph.replay()
torch.cuda.synchronize()
print(json.dumps({"status": "PASS", "replays": 3,
                  "scope": "Marlin binary control flow only; no numerical execution"}))
