"""Native AIS regression units, qualified identity and explicit support boundary."""
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.aisimulate.identity import Identity
from adapters.aisimulate.model import Regression, metrics
from adapters.vllm.costs import Shape, StepSample, UnsupportedStep


def main() -> None:
    identity = Identity("fixture-model", "fixture-device", "vllm", "0.30.0", "bfloat16",
                        "TRITON_ATTN", "eager", 1, 1, 1, 1, 1, 1, 16,
                        "whole_step_including_host_and_communication")
    samples = [StepSample(Shape(phase, batch, context), 1_000_000 + batch * context * 1000)
               for phase in ("prefill", "decode") for batch in (1, 8, 16, 32)
               for context in (32, 64, 128) for _ in range(2)]
    for altered in (replace(identity, hardware_profile_sha256="other"),
                    replace(identity, backend_version="0.14.0"),
                    replace(identity, scope="isolated_kernel")):
        try:
            Regression(altered, identity, samples)
        except ValueError:
            pass
        else:
            raise AssertionError("identity mismatch must fail before fitting")
    model = Regression(identity, identity, samples)
    try:
        shape = Shape("prefill", 8, 64)
        assert metrics(shape, 1_512_000)["wall_time"] == .001512
        estimate = model.predict(shape)
        assert abs(estimate / 1_512_000 - 1) < .1, estimate
        query = metrics(shape)
        direct = model.model.estimate_forward_pass_time_ms(query)
        query["wall_time"] = 1000
        assert model.model.estimate_forward_pass_time_ms(query) == direct
        assert estimate == round(direct * 1e6)
        try:
            model.predict(Shape("prefill", 64, 64))
        except UnsupportedStep:
            pass
        else:
            raise AssertionError("native extrapolation must not escape recorded support")
        print("PASS AISimulate 0.12.0 native regression: seconds/ms/ns units, frozen estimates, identity rejection, unsupported shape")
    finally:
        model.close()


if __name__ == "__main__":
    main()
