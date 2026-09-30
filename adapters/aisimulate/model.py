"""Pinned AISimulate regression over observed steps; no implicit native fallback."""
from importlib.metadata import version
import math

from aisimulate_core.sdk import RustForwardPassPerfModel
from adapters.vllm.costs import Shape, StepModel, StepSample, UnsupportedStep
from adapters.aisimulate.identity import Identity


def metrics(shape: Shape, duration_ns: int = 0) -> dict:
    if shape.phase not in ("prefill", "decode") or shape.batch <= 0 or shape.context <= 0 or not math.isfinite(shape.context) or duration_ns < 0:
        raise ValueError("unsupported FPM step")
    prefill = shape.phase == "prefill"
    return {"version": 1, "worker_id": "calibrated-rank", "dp_rank": 0,
        "counter_id": 0, "wall_time": duration_ns / 1e9,
        "scheduled_requests": {
            "num_prefill_requests": shape.batch if prefill else 0,
            "sum_prefill_tokens": round(shape.batch * shape.context) if prefill else 0,
            "var_prefill_length": 0., "sum_prefill_kv_tokens": 0,
            "num_decode_requests": 0 if prefill else shape.batch,
            "sum_decode_kv_tokens": 0 if prefill else round(shape.batch * shape.context),
            "var_decode_kv_tokens": 0.},
        "queued_requests": {"num_prefill_requests": 0, "sum_prefill_tokens": 0,
            "var_prefill_length": 0., "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0, "var_decode_kv_tokens": 0.}}


class Regression:
    scope = "whole_step_including_host_and_communication"

    def __init__(self, identity: Identity, expected: Identity, samples: list[StepSample]):
        if identity != expected or version("aisimulate") != identity.aisimulate_version:
            raise ValueError("AIS version/model/hardware/backend identity mismatch")
        if identity.scope != self.scope or (identity.tp, identity.pp, identity.attention_dp, identity.cp, identity.moe_tp, identity.moe_ep) != (1, 1, 1, 1, 1, 1):
            raise ValueError("qualified regression is single-rank whole-step only")
        if identity.graph_mode != "eager" or identity.kv_block_size <= 0:
            raise ValueError("unsupported graph mode or KV layout")
        self.identity = identity
        self.domain = StepModel(samples)
        options = {"max_observations": max(64, len(samples)), "min_observations": 5,
                   "max_batch_size": max(sample.shape.batch for sample in samples),
                   "max_num_tokens": max(round(sample.shape.batch * sample.shape.context) for sample in samples),
                   "max_kv_tokens": max(round(sample.shape.batch * sample.shape.context) for sample in samples)}
        self.model = RustForwardPassPerfModel.from_regression(options)
        self.model.tune_with_fpms([[metrics(sample.shape, sample.duration_ns)] for sample in samples])

    def predict(self, shape: Shape) -> int:
        # The upstream regression extrapolates. This adapter requires the same
        # recorded shape domain as the measured-step comparator.
        self.domain.predict(shape)
        value = self.model.estimate_forward_pass_time_ms(metrics(shape))
        if value is None or not math.isfinite(value) or value <= 0:
            raise UnsupportedStep("AIS regression has no supported positive estimate")
        return max(1, round(value * 1e6))

    def close(self) -> None:
        self.model.close()
