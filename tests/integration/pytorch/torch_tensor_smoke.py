"""Metadata-only PyTorch CUDA tensor smoke test; contents are not computed."""
import os
from pathlib import Path

import torch

assert not list(Path("/dev").glob("nvidia*"))
assert torch.version.cuda == os.environ["FAKE_CUDA_EXPECTED_RUNTIME"]
tensor = torch.empty(1024, dtype=torch.float32, device="cuda:0")
assert tensor.device.type == "cuda" and tensor.shape == (1024,)
torch.cuda.synchronize()
print("PASS: PyTorch CUDA tensor metadata/allocation and normal interpreter teardown")
# Keep the tensor alive until Python shutdown to exercise CUDA Runtime teardown.
