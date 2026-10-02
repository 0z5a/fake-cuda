"""Native short-row selection semantics across widths, precision and edge cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("LD_PRELOAD"):
        raise ValueError("native selector correctness must not use the fake Driver")
    import torch
    torch.cuda.set_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        for index, rows in enumerate((1, 8, 32, 64, 128)):
            for width in (64, 128, 256):
                k = (1, 2, 4, 8)[index % 4]
                scores = torch.randn(rows, width, device="cuda", dtype=dtype)
                values, indices = torch.topk(scores, k, sorted=True)
                expected, _ = torch.topk(scores.cpu(), k, sorted=True)
                torch.testing.assert_close(values.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(scores.gather(1, indices), values, rtol=0, atol=0)
                assert all(len(set(row)) == k for row in indices.cpu().tolist())
                cases.append((str(dtype), rows, width, k, "finite"))
        for k in (1, 257):
            scores = torch.ones(2, 257, device="cuda", dtype=dtype)
            values, indices = torch.topk(scores, k, sorted=True)
            assert bool((values == 1).all()) and all(len(set(row)) == k for row in indices.cpu().tolist())
            cases.append((str(dtype), 2, 257, k, "ties_backend_unspecified_index_order"))
        scores = torch.tensor([[float("nan"), float("inf"), 3., -float("inf")]], device="cuda", dtype=dtype)
        values, indices = torch.topk(scores, 4, sorted=True)
        expected, _ = torch.topk(scores.cpu(), 4, sorted=True)
        torch.testing.assert_close(values.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        cases.append((str(dtype), 1, 4, 4, "NaN_then_positive_inf_then_finite_then_negative_inf"))
    document = {"scope": "native_numerical_semantics; timing_domains_remain_separate", "torch": torch.__version__,
                "gpu_uuid": str(torch.cuda.get_device_properties(args.device).uuid), "pid": os.getpid(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": cases, "correctness": "passed"}
    args.output.write_text(json.dumps(document, sort_keys=True))
    print(f"PASS {len(cases)} native selector configurations; tie indices remain backend-unspecified")


if __name__ == "__main__":
    main()
