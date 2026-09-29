"""Official FA4 forward: numerical validation and real eager/Graph timing."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from flash_attn.cute import flash_attn_func

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repetitions', type=int, default=100)
args = parser.parse_args()
assert args.repetitions > 0
torch.manual_seed(20260929)
results = []
for length, causal in ((512, False), (2048, False), (2048, True)):
    q, k, v = [torch.randn(1, length, 8, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

    def run() -> torch.Tensor:
        return flash_attn_func(q, k, v, causal=causal)[0]

    def reference() -> torch.Tensor:
        scores = q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2) / 8
        if causal:
            scores.masked_fill_(torch.ones(length, length, device='cuda', dtype=torch.bool).triu(1), -torch.inf)
        return (scores.softmax(-1) @ v.float().transpose(1, 2)).transpose(1, 2)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            eager_output = run()
    stream.synchronize()
    torch.testing.assert_close(eager_output.float(), reference(), atol=0.003, rtol=0.003)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    errors = []
    for factor in (1.0, -0.5, 2.0):
        q.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        expected = reference()
        torch.testing.assert_close(output.float(), expected, atol=0.003, rtol=0.003)
        errors.append((output.float() - expected).abs().max().item())
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
    results.append(dict(length=length, causal=causal, heads=8, head_dim=64,
                        correctness='PASS', max_abs_errors=errors, eager_wall_us=eager,
                        graph_wall_us=replay, speedup=eager/replay, arms=arms))
    del graph, output, eager_output, expected, q, k, v
    torch.cuda.empty_cache()
report = dict(scope='real FA4 forward; not full-model E2E or simulator prediction',
              torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
              repetitions=args.repetitions, results=results)
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
