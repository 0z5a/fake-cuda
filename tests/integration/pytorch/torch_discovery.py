"""Check CUDA discovery with the shim in an offline, no-GPU container."""
import os
import sys
from pathlib import Path

assert not list(Path("/dev").glob("nvidia*")), "test must run without NVIDIA devices"

expected_runtime = os.environ["FAKE_CUDA_EXPECTED_RUNTIME"]
expected_wheel = {"12.8": "2.11.0+cu128", "13.0": "2.11.0+cu130"}[expected_runtime]
expected_venv = Path(__file__).resolve().parents[3] / f".venv-cu{expected_runtime.replace('.', '')}"
assert Path(sys.prefix).resolve() == expected_venv.resolve(), (
    f"Python must run from {expected_venv}, got {sys.prefix}"
)

import torch

torch_file = Path(torch.__file__).resolve()
assert torch_file.is_relative_to(expected_venv / "lib/python3.12/site-packages/torch"), (
    f"torch must come from {expected_venv}, got {torch_file}"
)
assert torch.__version__ == expected_wheel, (
    f"expected torch {expected_wheel}, got {torch.__version__}"
)
assert torch.version.cuda == expected_runtime, (
    f"expected CUDA runtime {expected_runtime}, got {torch.version.cuda}"
)
print("torch:", torch.__version__, "runtime:", torch.version.cuda, "file:", torch_file)
assert torch.cuda.is_available(), "PyTorch could not discover the fake device"
count = torch.cuda.device_count()
assert count == 1, f"expected one fake device, got {count}"
print("device_count:", count)
properties = torch.cuda.get_device_properties(0)
assert properties.name == "NVIDIA GH200 144G HBM3e", properties
assert (properties.major, properties.minor) == (9, 0), properties
assert properties.total_memory == 153008209920, properties
assert properties.multi_processor_count == 132, properties
assert properties.L2_cache_size == 62914560, properties
print("device:", properties)
assert torch.cuda.is_initialized(), "PyTorch CUDA lazy initialization failed"
print("PASS: PyTorch CUDA discovery, properties and lazy initialization")
