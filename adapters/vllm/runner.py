"""Original vLLM schedulers receive arrivals and explicit token-oracle outputs."""
from dataclasses import dataclass
from pathlib import Path
import statistics

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from adapters.common.bridge import ResourceBridge
from adapters.common.clock import Coordinator, PacedCoordinator
from adapters.common.control import ControlLedger
from adapters.vllm.costs import StepCost
from adapters.vllm.worker import add_request, complete_step, make_scheduler, scheduled_shape


@dataclass(frozen=True)
class Workload:
    name: str
    arrivals_ns: tuple[int, ...]
    prompt: int
    output: int


@dataclass
class ServingResult:
    finish_ns: int
    tokens: dict[str, list[int]]
    batch_ids: list[list[str]]
    waiting: list[int]
    arrivals_observed_ns: list[int]
    control_reads: int

    def metrics(self, workload: Workload) -> dict[str, float]:
        if len(self.tokens) != len(workload.arrivals_ns) or any(len(times) != workload.output for times in self.tokens.values()):
            raise ValueError("incomplete serving workload")
        ttft = [self.tokens[str(i)][0] - arrival for i, arrival in enumerate(workload.arrivals_ns)]
        intervals = [b - a for times in self.tokens.values() for a, b in zip(times, times[1:])]
        if min(ttft) < 0 or (intervals and min(intervals) <= 0):
            raise ValueError("invalid token timeline")
        return {"finish_ns": self.finish_ns, "tokens_per_second": len(ttft) * workload.output * 1e9 / self.finish_ns,
                "ttft_median_ms": statistics.median(ttft) / 1e6,
                "itl_median_ms": statistics.median(intervals) / 1e6 if intervals else 0}


def run(workload: Workload, configurations: list[tuple[VllmConfig, KVCacheConfig, int, int]],
        models: list[StepCost], bridge_path: Path, scale: float = 1, paced: bool = False) -> ServingResult:
    if not workload.arrivals_ns or sorted(workload.arrivals_ns) != list(workload.arrivals_ns) or min(workload.arrivals_ns) < 0 or workload.prompt <= 0 or workload.output <= 0 or scale <= 0:
        raise ValueError("invalid serving workload")
    if not models or len(configurations) != len(models):
        raise ValueError("every rank needs a configuration and cost model")
    if any(model.scope != "whole_step_including_host_and_communication" for model in models):
        raise ValueError("scheduler boundary requires whole-step scope; included host/communication must not be billed again")
    schedulers = [make_scheduler(configuration) for configuration in configurations]
    clock_type = PacedCoordinator if paced else Coordinator
    clock = clock_type(("requests", "scheduler", "device"))
    for i, arrival in enumerate(workload.arrivals_ns):
        clock.send("requests", "scheduler", "arrival", str(i), arrival, arrival)
    forever = (1 << 63) - 1
    clock.certify("requests", forever)
    clock.certify("device", forever)
    ledger = ControlLedger()
    result = ServingResult(0, {}, [], [], [], 0)
    submitted = 0
    step = 0
    bridge = ResourceBridge(bridge_path, len(models))
    try:
        while submitted < len(workload.arrivals_ns) or schedulers[0].get_num_unfinished_requests():
            if not schedulers[0].get_num_unfinished_requests():
                clock.certify("scheduler", forever)
                events = clock.advance()
                for event in events:
                    if event.kind != "arrival":
                        raise ValueError("unexpected idle scheduler event")
                    index = int(event.payload)
                    for scheduler in schedulers:
                        add_request(scheduler, str(index), workload.prompt, workload.output, event.created_ns)
                    result.arrivals_observed_ns.append(clock.now)
                    submitted += 1
            scheduled = [scheduler.schedule() for scheduler in schedulers]
            if any(out.num_scheduled_tokens != scheduled[0].num_scheduled_tokens for out in scheduled):
                raise ValueError("cross-rank batch mismatch")
            shapes = [scheduled_shape(scheduler, out) for scheduler, out in zip(schedulers, scheduled)]
            if any(shape != shapes[0] for shape in shapes):
                raise ValueError("cross-rank shape mismatch")
            step += 1
            identities = [step * len(models) + rank for rank in range(len(models))]
            bridge.advance(clock.now)
            for rank, (identity, model, shape) in enumerate(zip(identities, models, shapes)):
                bridge.submit(identity, rank, clock.now, max(1, round(model.predict(shape) * scale)))
            # A synchronous TP step completes only after every rank. Its calibrated
            # costs already include communication; no extra NCCL cost is charged.
            while any(bridge.completion(identity) < 0 for identity in identities):
                end = bridge.next_event()
                clock.resume("device")
                clock.send("device", "scheduler", "completion", str(step), end, end)
                clock.certify("device", end)
                while clock.now < end:
                    clock.certify("scheduler", end)
                    for event in clock.advance():
                        if event.kind == "arrival":
                            index = int(event.payload)
                            for scheduler in schedulers:
                                add_request(scheduler, str(index), workload.prompt, workload.output, event.created_ns)
                            result.arrivals_observed_ns.append(clock.now)
                            submitted += 1
                bridge.advance(end)
            clock.certify("device", forever)
            visible = [complete_step(scheduler, out, ledger, f"{step}/{rank}") for rank, (scheduler, out) in enumerate(zip(schedulers, scheduled))]
            if any(rank != visible[0] for rank in visible):
                raise ValueError("cross-rank control output mismatch")
            for rid, tokens, finished in visible[0]:
                result.tokens.setdefault(rid, []).extend([clock.now] * len(tokens))
            result.batch_ids.append(list(scheduled[0].num_scheduled_tokens))
            result.waiting.append(len(schedulers[0].waiting))
        result.finish_ns = clock.now
        result.control_reads = len(ledger.reads)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU adapter initialized CUDA")
        result.metrics(workload)
        return result
    finally:
        bridge.close()
