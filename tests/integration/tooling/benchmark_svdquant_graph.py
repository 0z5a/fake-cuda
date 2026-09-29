"""Official Nunchaku W4A4 plus nonzero rank-32 branch, no model checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku import _C

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repetitions', type=int, default=100)
args = parser.parse_args()
assert args.repetitions > 0
torch.manual_seed(20260929)
packer = NunchakuWeightPacker(4)
results = []
for m in (256, 1024, 4096):
    k, n, rank = 1024, 2048, 32
    # Exact quantization grid makes the independent FP32 oracle unambiguous.
    a = torch.randint(-7, 8, (m, k), device='cuda').half() / 8
    a[:, ::64] = 7 / 8
    weight_int = torch.randint(-7, 8, (n, k), device='cuda', dtype=torch.int32)
    weight = weight_int.float() / 32
    packed_weight = packer.pack_weight(weight_int)
    weight_scales = packer.pack_scale(torch.full((n, k//64), 1/32, device='cuda', dtype=torch.float16), 64)
    down = torch.randn(rank, k, device='cuda', dtype=torch.float16) * 0.03
    up = torch.randn(n, rank, device='cuda', dtype=torch.float16) * 0.03
    packed_down = packer.pack_lowrank_weight(down, down=True)
    packed_up = packer.pack_lowrank_weight(up, down=False)
    torch.testing.assert_close(packer.unpack_lowrank_weight(packed_down, down=True), down, atol=0, rtol=0)
    torch.testing.assert_close(packer.unpack_lowrank_weight(packed_up, down=False), up, atol=0, rtol=0)
    smooth = torch.ones(k, device='cuda', dtype=torch.float16)
    packed_a = torch.empty(m, k//2, device='cuda', dtype=torch.uint8)
    a_scales = torch.empty(k//64, m, device='cuda', dtype=torch.float16)
    lowrank_a = torch.empty(m, rank, device='cuda', dtype=torch.float32)
    output = torch.empty(m, n, device='cuda', dtype=torch.float16)

    def run() -> torch.Tensor:
        svdq_quantize_w4a4_act_fuse_lora_cuda(a, packed_a, a_scales, packed_down, lowrank_a, smooth)
        svdq_gemm_w4a4_cuda(packed_a, packed_weight, out=output, ascales=a_scales,
                           wscales=weight_scales, lora_act_in=lowrank_a, lora_up=packed_up)
        return output

    def reference() -> torch.Tensor:
        return a.float() @ weight.T + (a.float() @ down.float().T) @ up.float().T

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            run()
    stream.synchronize()
    torch.testing.assert_close(output.float(), reference(), atol=0.02, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    errors = []
    for factor in (1.0, -0.5, 2.0):
        a.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        expected = reference()
        torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
        errors.append((output.float()-expected).abs().max().item())
    lowrank_max = ((a.float() @ down.float().T) @ up.float().T).abs().max().item()
    assert lowrank_max > 0.1, 'Low-rank branch must materially affect the output'
    arms = []
    for block, order in enumerate(('APPA', 'PAAP', 'APPA', 'PAAP')):
        for position, arm in enumerate(order):
            operation = run if arm == 'A' else graph.replay
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
            arms.append(dict(block=block, position=position, arm=arm,
                             wall_us=(time.perf_counter()-start)*1e6/args.repetitions,
                             device_us=begin.elapsed_time(end)*1000/args.repetitions))
    eager = statistics.median(x['wall_us'] for x in arms if x['arm']=='A')
    replay = statistics.median(x['wall_us'] for x in arms if x['arm']=='P')
    results.append(dict(m=m, k=k, n=n, rank=rank, group_size=64, correctness='PASS',
                        max_abs_errors=errors, lowrank_max_abs=lowrank_max,
                        eager_wall_us=eager, graph_wall_us=replay, speedup=eager/replay, arms=arms))
    del graph, output, expected, a, weight_int, weight, packed_weight, weight_scales
    del down, up, packed_down, packed_up, smooth, packed_a, a_scales, lowrank_a
    torch.cuda.empty_cache()
extension = Path(_C.__file__)
report = dict(scope='real SVDQuant INT4 linear pipeline including fused low-rank branch; not model E2E',
              torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
              extension_sha256=hashlib.file_digest(extension.open('rb'), 'sha256').hexdigest(),
              repetitions=args.repetitions, results=results)
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
