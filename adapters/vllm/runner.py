"""Original vLLM schedulers receive arrivals and explicit token-oracle outputs."""
from dataclasses import dataclass
from pathlib import Path
import statistics

import torch

from adapters.common.bridge import ResourceBridge
from adapters.common.clock import Coordinator, Event, PacedCoordinator
from adapters.vllm.costs import Shape, StepCost
from adapters.vllm.ranks import Configuration, Operation, RankGroup, Reply


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
    rank_pids: tuple[int, ...] = ()
    ipc_messages: int = 0

    def metrics(self, workload: Workload) -> dict[str, float | None]:
        if len(self.tokens) != len(workload.arrivals_ns) or any(len(times) != workload.output for times in self.tokens.values()):
            raise ValueError("incomplete serving workload")
        ttft = [self.tokens[str(i)][0] - arrival for i, arrival in enumerate(workload.arrivals_ns)]
        intervals = [b - a for times in self.tokens.values() for a, b in zip(times, times[1:])]
        if min(ttft) < 0 or (intervals and min(intervals) <= 0):
            raise ValueError("invalid token timeline")
        return {"finish_ns": self.finish_ns, "tokens_per_second": len(ttft) * workload.output * 1e9 / self.finish_ns,
                "ttft_median_ms": statistics.median(ttft) / 1e6,
                "itl_median_ms": statistics.median(intervals) / 1e6 if intervals else None}


def run(workload: Workload, configurations: list[Configuration],
        models: list[StepCost], bridge_path: Path, scale: float = 1, paced: bool = False,
        ranks: RankGroup | None = None) -> ServingResult:
    if not workload.arrivals_ns or sorted(workload.arrivals_ns) != list(workload.arrivals_ns) or min(workload.arrivals_ns) < 0 or workload.prompt <= 0 or workload.output <= 0 or scale <= 0:
        raise ValueError("invalid serving workload")
    if not models or len(configurations) != len(models):
        raise ValueError("every rank needs a configuration and cost model")
    if any(model.scope != "whole_step_including_host_and_communication" for model in models):
        raise ValueError("scheduler boundary requires whole-step scope; included host/communication must not be billed again")
    group = ranks or RankGroup(configurations)
    if group.size != len(models):
        raise ValueError("every rank needs a configuration and cost model")
    clock_type = PacedCoordinator if paced else Coordinator
    rank_names = tuple(f"rank-{i}" for i in range(len(models)))
    clock = clock_type(("requests", "scheduler", "device", *rank_names))
    for i, arrival in enumerate(workload.arrivals_ns):
        clock.send("requests", "scheduler", "arrival", str(i), arrival, arrival)
    forever = (1 << 63) - 1
    clock.certify("requests", forever)
    clock.certify("device", forever)
    result = ServingResult(0, {}, [], [], [], 0)
    submitted = 0
    step = 0
    bridge = ResourceBridge(bridge_path, len(models))

    def exchange(operation: Operation, arrivals: tuple[tuple[str, int, int, int], ...] = (),
                 configurations: list[Configuration] | None = None) -> list[Reply]:
        identities = []
        for actor in rank_names:
            clock.resume(actor)
            identities.append(clock.begin_send(actor))
        replies = group.command(operation, clock.now, arrivals, configurations)
        if any((r.unfinished, r.waiting, r.control_reads) !=
               (replies[0].unfinished, replies[0].waiting, replies[0].control_reads) for r in replies):
            raise RuntimeError("cross-rank control state mismatch")
        # All pipes have replied before any virtual-time advance. IPC wall delay
        # cannot rewrite the event's logical creation/delivery timestamp.
        for actor, identity, reply in zip(rank_names, identities, replies):
            clock.register(identity, "scheduler", "rank_reply", f"{reply.epoch}:{reply.sequence}",
                           reply.virtual_ns, reply.virtual_ns)
            clock.certify(actor, forever)
        return replies

    def deliver(events: list[Event]) -> list[tuple[str, int, int, int]]:
        arrivals = []
        for event in events:
            if event.kind == "arrival":
                arrivals.append((event.payload, workload.prompt, workload.output, event.created_ns))
                result.arrivals_observed_ns.append(clock.now)
            elif event.kind == "rank_reply":
                if event.created_ns != event.deliver_ns or event.observed_ns != event.created_ns:
                    raise RuntimeError("protocol_failure: IPC acknowledgement timestamp changed")
                result.ipc_messages += 1
            elif event.kind != "completion":
                raise ValueError("unexpected serving event")
        return arrivals

    try:
        replies = exchange("reset", configurations=configurations)
        while submitted < len(workload.arrivals_ns) or replies[0].unfinished:
            while not replies[0].unfinished:
                clock.certify("scheduler", forever)
                arrivals = deliver(clock.advance())
                if arrivals:
                    replies = exchange("arrive", arrivals=tuple(arrivals))
                    submitted += len(arrivals)
            scheduled = exchange("schedule")
            if any(out.token_counts != scheduled[0].token_counts for out in scheduled):
                raise ValueError("cross-rank batch mismatch")
            shapes: list[Shape] = []
            for reply in scheduled:
                assert reply.shape is not None
                shapes.append(reply.shape)
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
                    arrivals = deliver(clock.advance())
                    if arrivals:
                        replies = exchange("arrive", arrivals=tuple(arrivals))
                        submitted += len(arrivals)
                bridge.advance(end)
            clock.certify("device", forever)
            replies = exchange("complete")
            visible = [reply.visible for reply in replies]
            if any(rank != visible[0] for rank in visible):
                raise ValueError("cross-rank control output mismatch")
            for rid, tokens, finished in visible[0]:
                result.tokens.setdefault(rid, []).extend([clock.now] * len(tokens))
            assert scheduled[0].token_counts is not None
            result.batch_ids.append(list(scheduled[0].token_counts))
            result.waiting.append(replies[0].waiting)
        clock.certify("scheduler", forever)
        deliver(clock.advance())
        result.finish_ns = clock.now
        result.control_reads = sum(reply.control_reads for reply in replies)
        result.rank_pids = tuple(sorted(group.pids))
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU adapter initialized CUDA")
        result.metrics(workload)
        return result
    finally:
        bridge.close()
        if ranks is None:
            group.close()
