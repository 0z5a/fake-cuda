"""CPU-only original scheduler smoke, causal arrival and query-delay invariance."""
import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.vllm.costs import Shape, StepModel, StepSample
from adapters.vllm.runner import Workload, run


class DelayedModel(StepModel):
    def predict(self, shape: Shape) -> int:
        time.sleep(.001)
        return super().predict(shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()
    if version("vllm") not in ("0.29.0", "0.30.0"):
        parser.error("qualified vLLM versions are 0.29.0 and 0.30.0")
    data = args.config.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.config_sha256:
        parser.error("trusted scheduler configuration digest mismatch")
    samples = [StepSample(Shape(phase, batch, context), 1000)
               for phase in ("prefill", "decode") for batch in (1, 8, 16, 32) for context in (32, 64, 128, 256)]
    workload = Workload("controlled-40", (0,) * 32 + (500,) * 8, 64, 4)
    wrong_scope = StepModel(samples)
    wrong_scope.scope = "isolated_kernel"
    try:
        run(workload, [pickle.loads(data)], [wrong_scope], args.bridge)
    except ValueError as error:
        assert "whole-step scope" in str(error)
    else:
        raise AssertionError("kernel-only costs must not be accepted as whole-step costs")
    reference = None
    for ranks, model_type in ((1, StepModel), (2, StepModel), (2, DelayedModel)):
        configs = [pickle.loads(data) for _ in range(ranks)]
        for config, kv, block, hashed in configs:
            config.parallel_config.tensor_parallel_size = ranks
        result = run(workload, configs, [model_type(samples) for _ in range(ranks)], args.bridge)
        assert result.finish_ns == 8000 and len(result.tokens) == 40
        assert result.tokens["0"] == [1000, 2000, 3000, 4000]
        assert result.tokens["39"] == [5000, 6000, 7000, 8000]
        assert result.arrivals_observed_ns == list(workload.arrivals_ns)
        assert result.control_reads == 160 * ranks
        if reference is not None:
            assert result.tokens == reference.tokens and result.batch_ids == reference.batch_ids
        reference = result
        print(f"PASS ranks={ranks}, model={model_type.__name__}: original vLLM scheduler, 40 requests, 160 tokens, exact arrivals, U1 fixed-token provenance, no CUDA")
    # The paced reference uses identical actor/dependency logic. Wall waiting
    # is only a verification mode; it does not intercept arbitrary OS timers.
    timed = [StepSample(sample.shape, 2_000_000) for sample in samples]
    configs = [pickle.loads(data)]
    delayed_arrivals = Workload("paced-pair", (0,) * 32 + (1_000_000,) * 8, 64, 4)
    coordinated = run(delayed_arrivals, configs, [StepModel(timed)], args.bridge)
    started = time.perf_counter_ns()
    paced = run(delayed_arrivals, [pickle.loads(data)], [StepModel(timed)], args.bridge, paced=True)
    wall = time.perf_counter_ns() - started
    assert paced.tokens == coordinated.tokens and paced.batch_ids == coordinated.batch_ids
    assert paced.arrivals_observed_ns == list(delayed_arrivals.arrivals_ns)
    assert paced.finish_ns == coordinated.finish_ns == 16_000_000 and wall >= paced.finish_ns
    print("PASS paced/coordinated pair: complete token/batch parity, arrival during service, real waits preserved; registered actors only")


if __name__ == "__main__":
    main()
