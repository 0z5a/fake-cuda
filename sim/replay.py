"""A serial operator clock, distinct from kernel service and live serving actors."""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from .cost import Cost, CostModel, Estimate
from .models import Stage
from .work import Registry, Work, checked


@dataclass(frozen=True)
class Interval:
    op_id: str
    stage: str
    category: str
    rank: int
    stream: int
    ready_ns: int
    start_ns: int
    end_ns: int
    cost: Cost


@dataclass(frozen=True)
class Report:
    timeline: tuple[Interval, ...]
    estimates: tuple[Estimate, ...]
    makespan_ns: int | None
    host_gap_ns: int
    unknown_count: int
    status: str
    schedule_mode: str = "serial_device"
    scope: str = "provided_operator_trace"
    release_times: tuple[tuple[str, int], ...] = ()

    def accounting(self) -> dict[str, int | str | None]:
        expected = sum(len(e.lowered.stages) for e in self.estimates if not e.work.fusion_group)
        expected += len({e.work.fusion_group for e in self.estimates if e.work.fusion_group})
        coverage = {"count_coverage": f"{len(self.timeline)}/{expected}", "scope": self.scope}
        if self.makespan_ns is None:
            return {**coverage, "makespan_ns": None, "status": "incomplete"}
        sums = {name: 0 for name in ("attn_core", "attn_module", "moe", "other", "fused_shared")}
        for item in self.timeline:
            sums[item.category] += item.end_ns - item.start_ns
        sums["attn_module"] += sums["attn_core"]
        return {**coverage, **{f"{name}_ns": value for name, value in sums.items()}, "host_gap_ns": self.host_gap_ns,
                "makespan_ns": self.makespan_ns, "unknown_count": self.unknown_count,
                "schedule_mode": self.schedule_mode, "status": self.status}

    def digest(self) -> str:
        payload = json.dumps(self, default=vars, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


def replay(registry: Registry, model: CostModel, releases: dict[str, int] | None = None,
           fused: dict[str, Cost] | None = None, strict: bool = True) -> Report:
    """Fusion costs cover an entire physical group; no proportional time allocation."""
    releases, fused = releases or {}, fused or {}
    works = registry.works
    if set(releases) - {work.op_id for work in works}:
        raise ValueError("unknown release identity")
    for value in releases.values():
        checked(value)
    estimates = tuple(model.estimate(work) for work in works)
    groups: dict[tuple[str, str], list[Estimate]] = {}
    owners: dict[str, tuple[str, str]] = {}
    for estimate in estimates:
        work = estimate.work
        owner = ("fusion", work.fusion_group) if work.fusion_group else ("op", work.op_id)
        groups.setdefault(owner, []).append(estimate)
        owners[work.op_id] = owner
    dependencies = {key: {owners[dep] for estimate in group for dep in estimate.work.deps if owners[dep] != key}
                    for key, group in groups.items()}
    # Stream sequence is an additional edge, independent of input list order.
    streams: dict[tuple[str, str, str, int, int], list[Estimate]] = {}
    for estimate in estimates:
        x = estimate.work.execution
        streams.setdefault((x.process, x.context, x.instance, x.rank, x.stream), []).append(estimate)
    for stream in streams.values():
        stream.sort(key=lambda estimate: estimate.work.execution.sequence)
        if len({estimate.work.execution.sequence for estimate in stream}) != len(stream):
            raise ValueError("duplicate stream sequence")
        for left, right in zip(stream, stream[1:]):
            a, b = owners[left.work.op_id], owners[right.work.op_id]
            if a != b:
                dependencies[b].add(a)
    completion: dict[tuple[str, str], int] = {}
    timeline: list[Interval] = []
    device_ready = host_gaps = unknown = 0
    missing = sum(sum(cost.standalone_ns is None for estimate in group for cost in estimate.costs)
                  if key[0] == "op" else int(key[1] not in fused or fused[key[1]].standalone_ns is None)
                  for key, group in groups.items())
    while len(completion) < len(groups):
        ready = []
        for key, group in groups.items():
            if key in completion or not dependencies[key] <= completion.keys():
                continue
            arrival = max(releases.get(estimate.work.op_id, 0) for estimate in group)
            at = max([arrival] + [completion[dep] for dep in dependencies[key]])
            x = group[0].work.execution
            name = f"fusion:{key[1]}" if key[0] == "fusion" else key[1]
            ready.append((at, x.rank, x.stream, x.sequence, name, key))
        if not ready:
            raise ValueError("cyclic dependencies")
        at, rank, stream, _, name, key = min(ready)
        group = groups[key]
        if key[0] == "fusion":
            cost = fused.get(key[1], Cost(None, "unsupported", "", "missing physical fusion cost"))
            physical = [(group[0].work.op_id, Stage(name, "fused_shared", (), ()), cost)]
        else:
            estimate = group[0]
            physical = [(estimate.work.op_id, stage, cost) for stage, cost in zip(estimate.lowered.stages, estimate.costs)]
        for op_id, stage, cost in physical:
            if cost.standalone_ns is None:
                if strict:
                    raise ValueError(f"{op_id}/{stage.name}: {cost.reason}")
                return Report(tuple(timeline), estimates, None, host_gaps, missing, "incomplete", release_times=tuple(sorted(releases.items())))
            if cost.status == "compat_unknown":
                unknown += 1
            start = max(at, device_ready)
            host_gaps = checked(host_gaps + start - device_ready)
            end = checked(start + cost.standalone_ns)
            timeline.append(Interval(op_id, stage.name, stage.category, rank, stream, at, start, end, cost))
            device_ready = at = end
        completion[key] = device_ready
    statuses = {item.cost.status for item in timeline}
    status = "compat" if unknown else "analytic" if "analytic_estimate" in statuses else "synthetic" if "synthetic" in statuses else "validation_candidate" if "validation_candidate" in statuses else "complete"
    return Report(tuple(timeline), estimates, device_ready, host_gaps, unknown, status, release_times=tuple(sorted(releases.items())))


def counterfactual(report: Report, category: str, speedup: float) -> float:
    if not report.makespan_ns or not math.isfinite(speedup) or speedup < 1 or report.schedule_mode != "serial_device":
        raise ValueError("counterfactual requires a complete serial report and speedup >= 1")
    costs = {(item.op_id, item.stage): replace(item.cost, standalone_ns=math.ceil((item.end_ns - item.start_ns) / speedup))
             for item in report.timeline if item.category == category or category == "attn_module" and item.category == "attn_core"}
    originals = {estimate.work.op_id: estimate for estimate in report.estimates}

    class CachedCosts:
        def estimate(self, work: Work) -> Estimate:
            original = originals[work.op_id]
            return replace(original, costs=tuple(costs.get((work.op_id, stage.name), cost)
                           for stage, cost in zip(original.lowered.stages, original.costs)))

    fusion = {item.stage.removeprefix("fusion:"): costs.get((item.op_id, item.stage), item.cost)
              for item in report.timeline if item.category == "fused_shared"}
    revised = replay(Registry(tuple(estimate.work for estimate in report.estimates)), CachedCosts(), dict(report.release_times), fusion)
    assert revised.makespan_ns is not None
    return report.makespan_ns / revised.makespan_ns
