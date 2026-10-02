"""Native captured peer-pointer copies must overwrite changed destination data."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("LD_PRELOAD"):
        raise ValueError("native peer correctness must not use the fake Driver")
    import torch
    if torch.cuda.device_count() != 2:
        raise ValueError("two explicitly visible native devices required")
    torch.set_num_threads(1)
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxEnablePeerAccess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    driver.cuMemcpyDtoDAsync_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
    contexts = []
    for device in range(2):
        torch.cuda.set_device(device)
        torch.empty(1, device=f"cuda:{device}")
        context = ctypes.c_void_p()
        if driver.cuCtxGetCurrent(ctypes.byref(context)) or not context.value:
            raise RuntimeError("native context query failed")
        contexts.append(context)
    cases = []
    for source_device, target_device in ((0, 1), (1, 0)):
        source = torch.empty(4096, dtype=torch.float32, device=f"cuda:{source_device}")
        torch.cuda.set_device(target_device)
        destination = torch.empty_like(source, device=f"cuda:{target_device}")
        code = driver.cuCtxEnablePeerAccess(contexts[source_device], 0)
        if code not in (0, 704):
            raise RuntimeError(f"native peer mapping failed: CUDA {code}")
        stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            code = driver.cuMemcpyDtoDAsync_v2(destination.data_ptr(), source.data_ptr(), source.numel() * 4, stream.cuda_stream)
            if code:
                raise RuntimeError(f"native peer capture failed: CUDA {code}")
        for value in (.125, -.75, 2.):
            source.fill_(value)
            torch.cuda.synchronize(source_device)
            destination.zero_()
            torch.cuda.synchronize(target_device)
            graph.replay()
            torch.cuda.synchronize(target_device)
            torch.testing.assert_close(destination.cpu(), torch.full((4096,), value), rtol=0, atol=0)
            cases.append((source_device, target_device, value))
    document = {"scope": "native_peer_pointer_D2D_graph; no_fake_Driver_capture_claim", "correctness": "passed",
                "cases": cases, "pid": os.getpid(), "torch": torch.__version__,
                "gpu_uuids": [str(torch.cuda.get_device_properties(i).uuid) for i in range(2)],
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(document, sort_keys=True))
    print("PASS captured mapped-peer copies overwrite changed data in both directions")


if __name__ == "__main__":
    main()
