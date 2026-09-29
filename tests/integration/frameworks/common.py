"""Shared checks for framework startup probes in the isolated Docker runner."""

import importlib
import os
from pathlib import Path
import sys


def stage(name, action):
    print(f"STAGE {name}", flush=True)
    try:
        result = action()
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"PASS {name}: {result}", flush=True)
    return result


def check_environment(framework_name):
    assert not list(Path("/dev").glob("nvidia*")), "container has NVIDIA devices"
    assert os.environ.get("LD_PRELOAD") == "/usr/lib/aarch64-linux-gnu/libcuda.so.1"
    venv = Path(os.environ["FRAMEWORK_VENV"])
    assert Path(sys.prefix).resolve() == venv.resolve(), (sys.prefix, venv)

    torch = stage("torch import", lambda: importlib.import_module("torch"))
    print("torch version:", torch.__version__, "CUDA runtime:", torch.version.cuda, flush=True)
    assert torch.__version__ == "2.11.0+cu130"
    assert Path(torch.__file__).resolve().is_relative_to(venv.resolve())
    def require_cuda():
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch could not discover the fake CUDA device")
        return True

    def require_one_device():
        count = torch.cuda.device_count()
        if count != 1:
            raise RuntimeError(f"expected one fake CUDA device, got {count}")
        return count

    stage("torch CUDA availability", require_cuda)
    stage("torch CUDA device count", require_one_device)
    stage("torch CUDA initialization", lambda: torch.cuda.get_device_properties(0))

    framework = stage(f"{framework_name} import", lambda: importlib.import_module(framework_name))
    print("framework version:", framework.__version__, flush=True)
    assert Path(framework.__file__).resolve().is_relative_to(venv.resolve())
    return framework
