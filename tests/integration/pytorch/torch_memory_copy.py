"""PyTorch allocation and copy API/control-flow probe; no device bytes are stored."""

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

stage = "query memory and allocate tensors"
try:
    free_before, total = torch.cuda.mem_get_info(0)
    assert 0 < free_before <= total
    host = torch.empty((256, 128), dtype=torch.float32, device="cpu")
    device_a = torch.empty_like(host, device="cuda:0")
    device_b = torch.empty_like(device_a)
    assert device_a.device == device_b.device == torch.device("cuda:0")
    assert device_a.shape == device_b.shape == host.shape
    assert device_a.data_ptr() and device_b.data_ptr() and device_a.data_ptr() != device_b.data_ptr()
    free_live, total_live = torch.cuda.mem_get_info(0)
    assert total_live == total and free_live < free_before

    stage = "H2D (CPU tensor to virtual CUDA tensor)"
    device_a.copy_(host)
    stage = "D2D (virtual CUDA tensor to CUDA tensor)"
    device_b.copy_(device_a)
    stage = "D2H (virtual CUDA tensor to CPU tensor)"
    result = torch.empty_like(host)
    result.copy_(device_b)

    stage = "asynchronous H2D/D2D/D2H API and stream sync"
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device_a.copy_(host, non_blocking=True)
        device_b.copy_(device_a, non_blocking=True)
        result.copy_(device_b, non_blocking=True)
    stream.synchronize()
    torch.cuda.synchronize()
    assert result.shape == host.shape and result.dtype == host.dtype
    # No result equality check: this simulator never copies bytes or computes values.
except Exception:
    print(f"FAIL: {stage} (see traceback for CUDA error)", file=sys.stderr, flush=True)
    raise

print("PASS: PyTorch CUDA allocation and H2D/D2D/D2H call/synchronization paths (no data transfer)")
