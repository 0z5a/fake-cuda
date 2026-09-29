"""CUDA 12/13 two-device metadata and cross-device event smoke, without numerical work."""
import os
import sys
from pathlib import Path

assert not list(Path("/dev").glob("nvidia*")), "test must run without NVIDIA devices"
assert os.environ["FAKE_CUDA_DEVICE_COUNT"] == "2"
expected_runtime = os.environ["FAKE_CUDA_EXPECTED_RUNTIME"]
expected_venv = Path(__file__).resolve().parents[3] / f".venv-cu{expected_runtime.replace('.', '')}"
assert Path(sys.prefix).resolve() == expected_venv.resolve()

import torch

assert Path(torch.__file__).resolve().is_relative_to(expected_venv / "lib/python3.12/site-packages/torch")
assert torch.version.cuda == expected_runtime
assert torch.cuda.device_count() == 2
for i in range(2):
    props = torch.cuda.get_device_properties(i)
    assert props.name == "NVIDIA GH200 144G HBM3e", props
    tensor = torch.empty((8,), device=f"cuda:{i}")
    assert tensor.device.index == i and tuple(tensor.shape) == (8,)

stream0 = torch.cuda.Stream(device=0)
stream1 = torch.cuda.Stream(device=1)
event = torch.cuda.Event()
event.record(stream0)
stream1.wait_event(event)
stream1.synchronize()
print("PASS: PyTorch two-device discovery, tensor metadata and cross-device event wait")
