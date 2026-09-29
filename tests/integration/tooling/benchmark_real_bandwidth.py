"""Optional real-GH200 reference, NOT a fake-driver or no-GPU acceptance test.

Run with the real NVIDIA driver and bind CPU+memory to the Grace NUMA node for
this GPU, e.g. numactl --cpunodebind=0 --membind=0 ... --device 0.
"""
import argparse

import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--mib", type=int, default=32)
parser.add_argument("--repeats", type=int, default=10)
args = parser.parse_args()
if not 1 <= args.mib <= 256 or not 1 <= args.repeats <= 20:
    parser.error("use 1..256 MiB and 1..20 repetitions")
assert torch.cuda.is_available(), "requires the real NVIDIA driver"
name = torch.cuda.get_device_name(args.device)
assert "GH200" in name, name
assert torch.cuda.device_count() > args.device

size = args.mib * 1024 * 1024
cpu = torch.empty(size, dtype=torch.uint8, pin_memory=True).fill_(7)
returned = torch.empty_like(cpu, pin_memory=True)
gpu = torch.empty(size, dtype=torch.uint8, device=f"cuda:{args.device}")
start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def measure(direction, destination, source):
    destination.copy_(source, non_blocking=True)
    torch.cuda.synchronize(args.device)
    start.record()
    for _ in range(args.repeats):
        destination.copy_(source, non_blocking=True)
    stop.record()
    stop.synchronize()
    elapsed = start.elapsed_time(stop) / 1000.0
    print(f"{direction}: {size * args.repeats / elapsed / 1e9:.1f} GB/s"
          f" (pinned, {args.mib} MiB x {args.repeats}, {elapsed:.4f} s)")


print(f"{name}, GPU {args.device} (real hardware; CUDA events)")
measure("H2D", gpu, cpu)
measure("D2H", returned, gpu)
assert torch.equal(returned, cpu), "transferred data changed"
