"""PyTorch CUDA graph capture/replay control flow; no tensor values are computed."""

import os
import sys
from pathlib import Path


assert not list(Path("/dev").glob("nvidia*")), "test must run without NVIDIA devices"
expected_runtime = os.environ["FAKE_CUDA_EXPECTED_RUNTIME"]
expected_venv = Path(__file__).resolve().parents[3] / f".venv-cu{expected_runtime.replace('.', '')}"
assert Path(sys.prefix).resolve() == expected_venv.resolve(), (
    f"Python must run from {expected_venv}, got {sys.prefix}"
)

import torch

assert Path(torch.__file__).resolve().is_relative_to(
    expected_venv / "lib/python3.12/site-packages/torch"
)
assert torch.version.cuda == expected_runtime, (torch.version.cuda, expected_runtime)
assert torch.cuda.is_available(), "PyTorch could not discover the fake device"
assert torch.cuda.device_count() == 1, "expected one virtual GPU"

stage = "create graph, streams, and events"
try:
    capture_stream = torch.cuda.Stream()
    wait_stream = torch.cuda.Stream()
    captured_event = torch.cuda.Event(enable_timing=False)
    replay_event = torch.cuda.Event(enable_timing=False)
    graph = torch.cuda.CUDAGraph()
    source = torch.empty(64, device="cuda:0")
    target = torch.empty_like(source)

    # Allocate outside capture: async allocation/free during capture is unsupported.
    # Graph nodes schedule virtual copies/events, never copy device bytes.
    stage = "prewarm capture event on side stream"
    captured_event.record(capture_stream)
    capture_stream.synchronize()

    stage = "CUDAGraph.capture_begin on side stream"
    with torch.cuda.stream(capture_stream):
        graph.capture_begin()
        stage = "verify active graph capture"
        assert torch.cuda.is_current_stream_capturing(), "capture did not become active"
        stage = "capture virtual D2D copy, event record and in-stream wait"
        target.copy_(source, non_blocking=True)
        captured_event.record(capture_stream)
        capture_stream.wait_event(captured_event)
        stage = "CUDAGraph.capture_end"
        graph.capture_end()

    for attempt in range(2):
        stage = f"CUDAGraph.replay #{attempt + 1} and cross-stream event wait"
        with torch.cuda.stream(capture_stream):
            graph.replay()
            replay_event.record(capture_stream)
        wait_stream.wait_event(replay_event)
        wait_stream.synchronize()
        assert replay_event.query(), f"replay #{attempt + 1} completion event not ready"

    stage = "final stream synchronization"
    capture_stream.synchronize()
except Exception:
    print(f"FAIL: {stage} (see traceback for CUDA error)", file=sys.stderr, flush=True)
    raise

print("PASS: PyTorch CUDA graph D2D/event capture, two replays, and cross-stream event waits (no data copy)")
