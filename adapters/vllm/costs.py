"""Frozen per-rank whole-step costs; host and collective work are included."""
from bisect import bisect_left
from dataclasses import dataclass
import math
import statistics
from typing import Protocol


@dataclass(frozen=True)
class Shape:
    phase: str
    batch: int
    context: float


@dataclass(frozen=True)
class StepSample:
    shape: Shape
    duration_ns: int


class StepCost(Protocol):
    scope: str
    def predict(self, shape: Shape) -> int: ...


class UnsupportedStep(ValueError):
    """A well-formed step lies outside the frozen predictor's support."""


def interpolate(points: dict[float, float], value: float) -> float:
    keys = sorted(points)
    if not keys or not keys[0] <= value <= keys[-1]:
        raise UnsupportedStep("outside calibration domain")
    i = bisect_left(keys, value)
    if keys[i] == value:
        return points[value]
    left, right = keys[i - 1], keys[i]
    weight = (value - left) / (right - left)
    return points[left] * (1 - weight) + points[right] * weight


class StepModel:
    scope = "whole_step_including_host_and_communication"

    def __init__(self, samples: list[StepSample]):
        groups: dict[tuple[str, int, float], list[int]] = {}
        for sample in samples:
            shape = sample.shape
            if shape.phase not in ("prefill", "decode") or shape.batch <= 0 or not math.isfinite(shape.context) or shape.context <= 0 or sample.duration_ns <= 0:
                raise ValueError("invalid calibrated step")
            context = shape.context if shape.phase == "prefill" else round(shape.context / 16) * 16
            groups.setdefault((shape.phase, shape.batch, context), []).append(sample.duration_ns)
        self.table: dict[str, dict[int, dict[float, float]]] = {}
        for (phase, batch, context), durations in groups.items():
            self.table.setdefault(phase, {}).setdefault(batch, {})[context] = statistics.median(durations)

    def predict(self, shape: Shape) -> int:
        if shape.phase not in self.table:
            raise UnsupportedStep("uncalibrated step phase")
        batches = self.table[shape.phase]
        keys = sorted(batches)
        i = bisect_left(keys, shape.batch)
        if i == len(keys) or shape.batch < keys[0]:
            raise UnsupportedStep("batch outside calibration domain")
        neighbors = [keys[i]] if keys[i] == shape.batch else keys[i - 1:i + 1]
        costs = {float(batch): interpolate(batches[batch], shape.context) for batch in neighbors}
        return round(interpolate(costs, shape.batch))
