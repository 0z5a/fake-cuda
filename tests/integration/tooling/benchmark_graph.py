"""Validate and time a capture-safe operation on real CUDA with stable inputs."""
import argparse
from collections.abc import Callable
import statistics
import time

import torch


def benchmark_graph(
    run: Callable[[], torch.Tensor],
    reference: Callable[[], torch.Tensor],
    update_input: Callable[[float], None],
    *,
    atol: float,
    rtol: float,
    repetitions: int = 100,
) -> tuple[float, float]:
    """Return median eager/replay microseconds, including submission and completion.

    Callbacks own inputs and weights. update_input changes values in place;
    reference computes expected values independently of run's output.
    """
    assert repetitions > 0
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            output = run()
    stream.synchronize()
    torch.testing.assert_close(output.float(), reference().float(), atol=atol, rtol=rtol)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    for factor in (1.0, -0.5, 2.0):
        update_input(factor)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.float(), reference().float(), atol=atol, rtol=rtol)

    times: dict[str, list[float]] = {"A": [], "P": []}
    for order in ("APPA", "PAAP", "APPA", "PAAP"):
        for arm in order:
            operation = run if arm == "A" else graph.replay
            for _ in range(5):
                operation()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repetitions):
                operation()
            torch.cuda.synchronize()
            times[arm].append((time.perf_counter() - start) * 1e6 / repetitions)
    return statistics.median(times["A"]), statistics.median(times["P"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("copy", "add"), default="add")
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(20260929)
    source = torch.randn(args.elements, device="cuda")
    target = torch.empty_like(source)

    def run() -> torch.Tensor:
        if args.operation == "copy":
            return target.copy_(source)
        return torch.add(source, source, out=target)

    def reference() -> torch.Tensor:
        return source.clone() if args.operation == "copy" else source * 2

    def update_input(factor: float) -> None:
        source.mul_(factor)

    eager, replay = benchmark_graph(run, reference, update_input, atol=0, rtol=0,
                                    repetitions=args.repetitions)
    print(f"{torch.cuda.get_device_name()}; torch {torch.__version__}; CUDA {torch.version.cuda}\n")
    print("| Operation | Values/replay | Eager µs | Graph µs | Speedup |")
    print("| --- | --- | ---: | ---: | ---: |")
    print(f"| {args.operation} ({args.elements}) | PASS | {eager:.3f} | {replay:.3f} | {eager/replay:.3f}× |")


if __name__ == "__main__":
    main()
