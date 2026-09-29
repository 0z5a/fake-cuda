"""PyTorch stream/event control-flow probe; no tensor contents are computed."""
import os
import sys
from pathlib import Path

assert not list(Path("/dev").glob("nvidia*")), "test must run without NVIDIA devices"
expected_runtime = os.environ["FAKE_CUDA_EXPECTED_RUNTIME"]
expected_venv = Path(__file__).resolve().parents[3] / f".venv-cu{expected_runtime.replace('.', '')}"
assert Path(sys.prefix).resolve() == expected_venv.resolve()

import torch

assert Path(torch.__file__).resolve().is_relative_to(expected_venv / "lib/python3.12/site-packages/torch")
assert torch.version.cuda == expected_runtime
assert torch.cuda.is_available() and torch.cuda.device_count() == 1

def check_metadata(tensor):
    assert tensor.device == torch.device("cuda:0")
    assert tensor.shape == (1024,) and tensor.dtype == torch.float32


host = torch.empty(1024, dtype=torch.float32)
default = torch.cuda.current_stream(0)
producer = torch.cuda.Stream(device=0)
consumer = torch.cuda.Stream(device=0)
independent = torch.cuda.Stream(device=0)
assert len({default.cuda_stream, producer.cuda_stream, consumer.cuda_stream, independent.cuda_stream}) == 4

with torch.cuda.stream(producer):
    assert torch.cuda.current_stream(0) == producer
    source = torch.empty(1024, dtype=torch.float32, device="cuda:0")
    produced = torch.empty_like(source)
    check_metadata(source)
    check_metadata(produced)
    produced.copy_(host, non_blocking=True)  # enqueue virtual H2D before the event
    ready = torch.cuda.Event(enable_timing=False)
    ready.record()

    with torch.cuda.stream(consumer):
        assert torch.cuda.current_stream(0) == consumer
        consumer.wait_event(ready)
        dependent = torch.empty_like(produced)
        check_metadata(dependent)
        dependent.copy_(produced, non_blocking=True)  # virtual D2D after wait
        finished = torch.cuda.Event(enable_timing=False)
        finished.record()
    assert torch.cuda.current_stream(0) == producer
assert torch.cuda.current_stream(0) == default

with torch.cuda.stream(independent):
    assert torch.cuda.current_stream(0) == independent
    other = torch.empty_like(source)
    check_metadata(other)
    other.copy_(host, non_blocking=True)  # independent virtual H2D
    other_finished = torch.cuda.Event(enable_timing=False)
    other_finished.record()
assert torch.cuda.current_stream(0) == default

# Synchronize each stream/event separately; do not rely on another stream's progress.
producer.synchronize()
assert ready.query()
finished.synchronize()
assert finished.query() and consumer.query()
independent.synchronize()
assert other_finished.query() and independent.query()

# Releasing a stream with an outstanding event must not invalidate event synchronization.
short_lived = torch.cuda.Stream(device=0)
with torch.cuda.stream(short_lived):
    last = torch.empty_like(source)
    check_metadata(last)
    teardown_event = torch.cuda.Event(enable_timing=False)
    teardown_event.record()
assert torch.cuda.current_stream(0) == default
del short_lived
teardown_event.synchronize()
assert teardown_event.query()

print("PASS: PyTorch CUDA stream isolation, metadata, event wait/query/sync and teardown")
# Keep tensors and events alive through interpreter shutdown to exercise Runtime teardown.
