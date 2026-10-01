"""Stage costs never enter the Driver's kernel-only PerformanceModel."""
from dataclasses import dataclass
import math
from typing import Protocol
from .models import Lowered, Stage, lower
from .work import Unknown, Work, checked


@dataclass(frozen=True)
class Identity:
    hardware: str
    driver: str
    backend: str
    revision: str
    mode: str
    cache: str

    def __post_init__(self) -> None:
        if not all((self.hardware, self.driver, self.backend, self.revision)) or self.mode not in ("eager", "graph") or self.cache not in ("cold", "measured-steady"):
            raise ValueError("incomplete hardware/backend identity or unsupported condition")


@dataclass(frozen=True)
class Cost:
    standalone_ns: int | None
    status: str
    profile_id: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.standalone_ns is not None:
            checked(self.standalone_ns)
            if not self.profile_id or self.status == "unsupported":
                raise ValueError("a known duration needs a source and supported accounting")


@dataclass(frozen=True)
class Estimate:
    work: Work
    lowered: Lowered
    costs: tuple[Cost, ...]


class CostModel(Protocol):
    def estimate(self, work: Work) -> Estimate: ...


class ConstantCost:
    """Synthetic contract fixture; never hardware calibration."""
    def __init__(self, ns: int):
        self.ns = checked(ns)

    def estimate(self, work: Work) -> Estimate:
        lowered = lower(work)
        return Estimate(work, lowered, tuple(Cost(self.ns if any(stage.features) else 0, "synthetic", "constant_contract") for stage in lowered.stages))


@dataclass(frozen=True)
class Profile:
    profile_id: str
    identity: Identity
    stage: str
    family: tuple[str, ...]
    samples: tuple[tuple[tuple[int, ...], int], ...]
    coefficients: tuple[float, ...] = ()
    lower: tuple[int, ...] = ()
    upper: tuple[int, ...] = ()
    qualified: bool = False
    schema: int = 1

    def __post_init__(self) -> None:
        if self.schema != 1 or not self.profile_id or not self.stage or not self.samples:
            raise ValueError("incomplete profile or unsupported version")
        dimensions = len(self.samples[0][0])
        if len({features for features, _ in self.samples}) != len(self.samples):
            raise ValueError("duplicate measured shape")
        for features, duration in self.samples:
            checked(duration)
            if len(features) != dimensions:
                raise ValueError("profile feature dimensions differ")
            for value in features:
                checked(value)
        if self.coefficients:
            if len(self.coefficients) != dimensions + 1 or len(self.lower) != dimensions or len(self.upper) != dimensions:
                raise ValueError("invalid fitted interpolation domain")
            if any(not math.isfinite(c) or c < 0 for c in self.coefficients) or any(lo > hi for lo, hi in zip(self.lower, self.upper)):
                raise ValueError("negative/nonfinite fit or reversed domain")

    def lookup(self, features: tuple[int, ...], candidate: bool) -> Cost:
        for key, ns in self.samples:
            if features == key:
                if not self.qualified and not candidate:
                    return Cost(None, "unsupported", self.profile_id, "profile has not passed its independent shape gate")
                return Cost(ns, "calibrated_estimate" if self.qualified else "validation_candidate", self.profile_id)
        inside = len(features) == len(self.lower) and all(lo <= x <= hi for lo, x, hi in zip(self.lower, features, self.upper))
        if self.coefficients and inside and (self.qualified or candidate):
            if len(features) == 1:
                left = max((sample for sample in self.samples if sample[0][0] < features[0]), key=lambda sample: sample[0])
                right = min((sample for sample in self.samples if sample[0][0] > features[0]), key=lambda sample: sample[0])
                fraction = (features[0] - left[0][0]) / (right[0][0] - left[0][0])
                ns = checked(round(left[1] + fraction * (right[1] - left[1])))
            else:
                ns = checked(round(self.coefficients[0] + sum(c * x for c, x in zip(self.coefficients[1:], features))))
            return Cost(ns, "calibrated_estimate" if self.qualified else "validation_candidate", self.profile_id)
        return Cost(None, "unsupported", self.profile_id, "unqualified interpolation or out-of-domain shape")


class CalibrationStore:
    def __init__(self, identity: Identity, profiles: tuple[Profile, ...], mode: str = "strict", candidate: bool = False):
        if mode not in ("strict", "analytic", "compat"):
            raise ValueError("unsupported cost mode")
        self.identity, self.mode, self.candidate = identity, mode, candidate
        self.profiles: dict[tuple[str, tuple[str, ...]], Profile] = {}
        for profile in profiles:
            key = (profile.stage, profile.family)
            if profile.identity != identity or key in self.profiles:
                raise ValueError("hardware/backend/condition mismatch or duplicate profile")
            self.profiles[key] = profile

    def estimate(self, work: Work) -> Estimate:
        lowered = lower(work)
        if (work.backend, work.revision) != (self.identity.backend, self.identity.revision):
            raise ValueError("descriptor backend revision differs from calibration")
        costs = []
        for stage in lowered.stages:
            if not any(stage.features) and not isinstance(work.spec, Unknown):
                cost = Cost(0, "empty", "zero_work")
            else:
                profile = self.profiles.get((stage.name, stage.family))
                cost = profile.lookup(stage.features, self.candidate) if profile else Cost(None, "unsupported", "", "missing profile")
            if cost.standalone_ns is None and self.mode == "compat":
                cost = Cost(10_000_000, "compat_unknown", "synthetic_constant_v1", cost.reason)
            costs.append(cost)
        return Estimate(work, lowered, tuple(costs))


class AnalyticCost:
    """Explicit uncalibrated rates; unknown auxiliary algorithms remain unsupported."""
    def __init__(self, tc_flops_s: float, simt_flops_s: float, bytes_s: float, launch_ns: int, logical_as_dram: bool = False):
        if any(not math.isfinite(rate) or rate <= 0 for rate in (tc_flops_s, simt_flops_s, bytes_s)):
            raise ValueError("positive finite rates in FLOP/s and byte/s required")
        self.rates = tc_flops_s, simt_flops_s, bytes_s
        self.launch_ns = checked(launch_ns)
        self.logical_as_dram = logical_as_dram

    def estimate(self, work: Work) -> Estimate:
        lowered = lower(work)
        costs = []
        for stage in lowered.stages:
            if not any(stage.features) and not isinstance(work.spec, Unknown):
                costs.append(Cost(0, "empty", "zero_work"))
            elif stage.simt_flops is None or stage.selection_items:
                costs.append(Cost(None, "unsupported", "", "selection/auxiliary algorithm needs measurement"))
            elif stage.logical_bytes and stage.dram_bytes is None and not self.logical_as_dram:
                costs.append(Cost(None, "unsupported", "", "DRAM demand is unknown; cold logical-traffic assumption not declared"))
            else:
                quantities = stage.tc_flops, stage.simt_flops, stage.dram_bytes if stage.dram_bytes is not None else stage.logical_bytes
                ns = checked(self.launch_ns + math.ceil(1e9 * max(q / rate for q, rate in zip(quantities, self.rates))))
                costs.append(Cost(ns, "analytic_estimate", "declared_rates_unvalidated"))
        return Estimate(work, lowered, tuple(costs))
