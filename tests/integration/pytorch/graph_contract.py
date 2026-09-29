"""Run the same graph control-flow fixture on CUDA or a GPU-free fake Driver."""
import argparse
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("real", "fake"), required=True)
parser.add_argument("--kernel", choices=("copy", "add"), default="copy")
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
if args.mode == "fake":
    assert not list(Path("/dev").glob("nvidia*"))

torch.cuda.set_device(args.device)
a, b = torch.cuda.Stream(), torch.cuda.Stream()
source = torch.empty(4096, device="cuda", dtype=torch.float32)
target = torch.empty_like(source)
side = torch.empty_like(source)
fork, join = torch.cuda.Event(), torch.cuda.Event()
graph = torch.cuda.CUDAGraph()
with torch.cuda.stream(a):
    if args.kernel == "add":
        torch.add(source, source, out=target)
    else:
        target.copy_(source)
    fork.record()
with torch.cuda.stream(b):
    b.wait_event(fork)
    side.copy_(target)
    join.record()
a.wait_event(join)
a.synchronize()

with torch.cuda.stream(a):
    graph.capture_begin()
    if args.kernel == "add":
        torch.add(source, source, out=target)
    else:
        target.copy_(source)
    fork.record()
with torch.cuda.stream(b):
    b.wait_event(fork)
    side.copy_(target)
    join.record()
with torch.cuda.stream(a):
    a.wait_event(join)
    graph.capture_end()

saved = []
for iteration in range(4):
    stream = (a, b)[iteration % 2]
    with torch.cuda.stream(stream):
        if args.mode == "real":
            source.fill_(iteration + 1)
        graph.replay()
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    assert done.query()
    if args.mode == "real":
        expected = (iteration + 1) * (2 if args.kernel == "add" else 1)
        torch.testing.assert_close(side, torch.full_like(side, expected), rtol=0, atol=0)
        saved.append(side.cpu())

if args.mode == "real":
    for iteration, output in enumerate(saved):
        expected = (iteration + 1) * (2 if args.kernel == "add" else 1)
        torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
print(json.dumps({"mode": args.mode, "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "kernel": args.kernel,
                  "replays": 4, "cross_stream_capture": "PASS",
                  "numerical_checks": "PASS" if args.mode == "real" else "NOT_APPLICABLE",
                  "status": "PASS"}))
