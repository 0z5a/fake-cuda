"""Two-device NCCL dispatch/expert/combine with explicit, frozen assignment order."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.ep import Payload, TokenRoute, ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("LD_PRELOAD"):
        raise ValueError("native EP contract must not use the fake Driver")
    import torch
    if torch.cuda.device_count() != 2:
        raise ValueError("two explicitly visible devices required")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    placement = (0, 1, 1, 0)
    routes = (TokenRoute(0, 0, (1, 2)), TokenRoute(0, 1, (0, 3)),
              TokenRoute(1, 0, (0, 1)), TokenRoute(1, 1, (0, 3)))
    traffic = ledger(routes, placement, 2, 4, Payload("bf16", "assignments"),
                     Payload("fp32", "assignments"), "NCCL_assignment_payload; frozen_host_route_metadata")
    x = torch.randn(2, 2, 4).bfloat16() * .1
    gu, down = torch.randn(4, 4, 16).bfloat16() * .1, torch.randn(4, 8, 4).bfloat16() * .1
    packets = {(r, s): tuple((route.token, expert) for route in routes if route.source == r
                             for expert in route.experts if placement[expert] == s) for r in range(2) for s in range(2)}
    inputs, incoming, outputs = {}, {}, {}
    streams = []
    for rank in range(2):
        torch.cuda.set_device(rank)
        streams.append(torch.cuda.Stream())
        for peer in range(2):
            inputs[rank, peer] = torch.stack([x[rank, token] for token, _ in packets[rank, peer]]).to(f"cuda:{rank}")
            incoming[rank, peer] = torch.empty(len(packets[peer, rank]), 4, device=f"cuda:{rank}", dtype=torch.bfloat16)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    torch.cuda.nccl.version()  # Load the private environment's actual NCCL DSO.
    libraries = {Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
                 if len(line.split()) >= 6 and 'libnccl.so' in line.split()[-1]}
    if len(libraries) != 1:
        raise ValueError("ambiguous native NCCL library identity")
    library = libraries.pop(); nccl = ctypes.CDLL(str(library))
    nccl.ncclCommInitAll.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    send_packet, receive_packet = nccl["ncclSend"], nccl["ncclRecv"]
    for function in (send_packet, receive_packet):
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    nccl.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    communicators = (ctypes.c_void_p * 2)()
    def check(code: int) -> None:
        if code:
            raise RuntimeError(f"native NCCL operation failed: {code}")
    check(nccl.ncclCommInitAll(communicators, 2, (ctypes.c_int * 2)(0, 1)))

    def exchange(send, receive, dtype: int) -> None:
        check(nccl.ncclGroupStart())
        for rank in range(2):
            torch.cuda.set_device(rank)
            peer = 1 - rank
            check(send_packet(send[rank, peer].data_ptr(), send[rank, peer].numel(), dtype, peer,
                                communicators[rank], streams[rank].cuda_stream))
            check(receive_packet(receive[rank, peer].data_ptr(), receive[rank, peer].numel(), dtype, peer,
                                communicators[rank], streams[rank].cuda_stream))
        check(nccl.ncclGroupEnd())
        for stream in streams:
            stream.synchronize()

    try:
        exchange(inputs, incoming, 9)  # ncclBfloat16
        for rank in range(2):
            torch.cuda.set_device(rank)
            experts = {expert: (gu[expert].to(f"cuda:{rank}"), down[expert].to(f"cuda:{rank}"))
                       for expert, owner in enumerate(placement) if owner == rank}
            for source in range(2):
                rows = inputs[rank, rank] if source == rank else incoming[rank, source]
                values = []
                for row, (_, expert) in zip(rows, packets[source, rank]):
                    gate_up, projection = experts[expert]
                    gated = row @ gate_up
                    values.append((torch.nn.functional.silu(gated[:8]) * gated[8:]) @ projection)
                outputs[rank, source] = torch.stack(values).float()
        returning = {(rank, peer): torch.empty(len(packets[rank, peer]), 4, device=f"cuda:{rank}")
                     for rank in range(2) for peer in range(2)}
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        exchange(outputs, returning, 7)  # ncclFloat32, separate combine payload
        result = torch.zeros_like(x, dtype=torch.float32)
        for source in range(2):
            for target in range(2):
                values = outputs[source, source] if target == source else returning[source, target]
                for value, (token, _) in zip(values.cpu(), packets[source, target]):
                    result[source, token] += value * .5
        reference = torch.zeros_like(result)
        for route in routes:
            for expert in route.experts:
                gated = x[route.source, route.token].float() @ gu[expert].float()
                reference[route.source, route.token] += .5 * (torch.nn.functional.silu(gated[:8]) * gated[8:]) @ down[expert].float()
        torch.testing.assert_close(result, reference, atol=2e-5, rtol=3e-2)
        assert traffic.assignments[0][1] == 2 and traffic.unique_tokens[0][1] == 1
        assert inputs[0, 1].numel() * inputs[0, 1].element_size() == traffic.dispatch_bytes[0][1]
        assert outputs[1, 0].numel() * outputs[1, 0].element_size() == traffic.combine_bytes[1][0]
        document = {"scope": "native_two_rank_small_MoE_protocol; frozen_host_routes; no_full_model_EP_claim",
                    "correctness": "passed", "torch": torch.__version__, "nccl_version": torch.cuda.nccl.version(),
                    "gpu_uuids": [str(torch.cuda.get_device_properties(i).uuid) for i in range(2)],
                    "assignments": traffic.assignments, "unique_tokens": traffic.unique_tokens,
                    "dispatch_bytes": traffic.dispatch_bytes, "combine_bytes": traffic.combine_bytes,
                    "max_abs_error": (result-reference).abs().max().item(), "pid": os.getpid(),
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "nccl_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
        args.output.write_text(json.dumps(document, sort_keys=True))
        print("PASS native NCCL assignment dispatch, resident experts, float32 combine and local bypass")
    finally:
        for communicator in communicators:
            check(nccl.ncclCommDestroy(communicator))


if __name__ == "__main__":
    main()
