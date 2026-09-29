"""GPU-free Nunchaku control-flow probe; deliberately makes no value assertions."""
from pathlib import Path
assert not list(Path('/dev').glob('nvidia*'))
import torch
from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda

m, k, n, rank = 256, 1024, 2048, 32

def half(*shape: int) -> torch.Tensor:
    return torch.empty(shape, device='cuda', dtype=torch.float16)

a, down, up, smooth = half(m, k), half(k, rank), half(n, rank), half(k)
packed_a = torch.empty(m, k//2, device='cuda', dtype=torch.uint8)
packed_weight = torch.empty(n, k//2, device='cuda', dtype=torch.int8)
a_scales, w_scales, output = half(k//64, m), half(k//64, n), half(m, n)
lowrank_a = torch.empty(m, rank, device='cuda', dtype=torch.float32)
print('SVDQuant import and virtual allocations PASS', flush=True)

def run() -> None:
    svdq_quantize_w4a4_act_fuse_lora_cuda(a, packed_a, a_scales, down, lowrank_a, smooth)
    svdq_gemm_w4a4_cuda(packed_a, packed_weight, out=output, ascales=a_scales,
                       wscales=w_scales, lora_act_in=lowrank_a, lora_up=up)

stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    run()
stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    run()
for _ in range(4):
    graph.replay()
torch.cuda.synchronize()
print('SVDQuant control flow PASS; no numerical validation')
