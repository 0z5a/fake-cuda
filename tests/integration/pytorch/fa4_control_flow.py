from pathlib import Path
assert not list(Path('/dev').glob('nvidia*'))
import torch
from flash_attn.cute import flash_attn_func
q, k, v = [torch.empty(1, 512, 8, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
print('FA4 import and virtual allocation PASS', flush=True)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    flash_attn_func(q, k, v)
stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    output = flash_attn_func(q, k, v)
for _ in range(4):
    graph.replay()
torch.cuda.synchronize()
print('FA4 control flow PASS; no numerical validation')
