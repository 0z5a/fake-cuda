"""Causal endpoint feedback shared by native routing policies."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.core.sched.output import SchedulerOutput

from adapters.common.bridge import ResourceBridge
from adapters.common.clock import Coordinator
from adapters.common.control import ControlLedger
from adapters.vllm.costs import StepCost
from adapters.vllm.runner import Workload, ServingResult
from adapters.vllm.worker import add_request, complete_step, make_scheduler, prompt_tokens, scheduled_shape


@dataclass(frozen=True)
class Decision:
    endpoint: int
    scores: dict[str, float]
    loads: tuple[dict, ...] = ()


class RoutingPolicy(Protocol):
    def choose(self, request: str, tokens: list[int], output: int, queues: list[int]) -> Decision: ...
    def feedback(self, prefills: list[str], finished: list[str]) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Route:
    arrival_ns: int
    request: str
    endpoint: int
    visible_queue: tuple[int, ...]
    visible_at: tuple[int, ...]
    scores: dict[str, float]
    native_loads: tuple[dict, ...]


def run(workload: Workload, configurations: list[tuple[VllmConfig, KVCacheConfig, int, int]],
        model: StepCost, scales: tuple[float, ...], metrics_delay_ns: int,
        bridge_path: Path, policy: RoutingPolicy) -> tuple[ServingResult, list[Route]]:
    count = len(configurations)
    if count != len(scales) or not count or min(scales) <= 0 or metrics_delay_ns < 0:
        raise ValueError("invalid endpoint configuration")
    if model.scope != "whole_step_including_host_and_communication":
        raise ValueError("routing boundary requires whole-step scope")
    schedulers = [make_scheduler(config) for config in configurations]
    actors = ("requests", "router", *(f"worker{i}" for i in range(count)))
    clock = Coordinator(actors)
    for i, arrival in enumerate(workload.arrivals_ns):
        clock.send("requests", "router", "arrival", str(i), arrival, arrival)
    visible = [0] * count
    visible_at = [0] * count
    sequences = [0] * count
    accepted = [0] * count
    updates: dict[tuple[int, int], tuple[int, list[str], list[str]]] = {}
    active: dict[int, tuple[int, SchedulerOutput]] = {}
    initial_free = [s.kv_cache_manager.block_pool.get_num_free_blocks() for s in schedulers]
    ledger = ControlLedger()
    result = ServingResult(0, {}, [], [], [], 0)
    routes: list[Route] = []
    identity = 0
    bridge = ResourceBridge(bridge_path, count)

    def publish(endpoint: int, prefills: list[str], finished: list[str]) -> None:
        sequences[endpoint] += 1
        sequence = sequences[endpoint]
        updates[(endpoint, sequence)] = (len(schedulers[endpoint].waiting), prefills, finished)
        actor = f"worker{endpoint}"
        clock.resume(actor)
        clock.send(actor, "router", "metrics", f"{endpoint}/{sequence}", clock.now, clock.now + metrics_delay_ns)

    def start(endpoint: int) -> None:
        nonlocal identity
        scheduler = schedulers[endpoint]
        if endpoint in active or not scheduler.get_num_unfinished_requests():
            return
        output = scheduler.schedule()
        cost = max(1, round(model.predict(scheduled_shape(scheduler, output)) * scales[endpoint]))
        identity += 1
        bridge.submit(identity, endpoint, clock.now, cost)
        active[endpoint] = (identity, output)
        actor = f"worker{endpoint}"
        clock.resume(actor)
        clock.send(actor, actor, "completion", str(endpoint), clock.now + cost, clock.now + cost)

    try:
        while clock.events:
            # Each callback has run to completion; all future causal events are
            # registered. No other actors or wall-clock timers participate.
            for actor in actors: clock.certify(actor, (1 << 63) - 1)
            events = clock.advance()
            bridge.advance(clock.now)
            for event in events:
                if event.kind == "arrival":
                    decision = policy.choose(event.payload, prompt_tokens(event.payload, workload.prompt), workload.output, visible)
                    endpoint = decision.endpoint
                    routes.append(Route(event.created_ns, event.payload, endpoint, tuple(visible), tuple(visible_at), decision.scores, decision.loads))
                    add_request(schedulers[endpoint], event.payload, workload.prompt, workload.output, event.created_ns)
                    result.arrivals_observed_ns.append(clock.now)
                    start(endpoint)
                    publish(endpoint, [], [])
                elif event.kind == "completion":
                    endpoint = int(event.payload)
                    invocation, output = active.pop(endpoint)
                    if bridge.completion(invocation) != clock.now:
                        raise ValueError("worker completion differs from ResourceEngine")
                    prefills, finished_requests = [], []
                    for rid, tokens, finished in complete_step(schedulers[endpoint], output, ledger, str(invocation)):
                        if tokens and rid not in result.tokens:
                            prefills.append(rid)
                        if finished:
                            finished_requests.append(rid)
                        result.tokens.setdefault(rid, []).extend([clock.now] * len(tokens))
                    result.batch_ids.append(list(output.num_scheduled_tokens))
                    start(endpoint)
                    publish(endpoint, prefills, finished_requests)
                elif event.kind == "metrics":
                    endpoint, sequence = map(int, event.payload.split("/"))
                    queue, prefills, finished = updates.pop((endpoint, sequence))
                    policy.feedback(prefills, finished)
                    if sequence > accepted[endpoint]:
                        visible[endpoint] = queue
                        accepted[endpoint] = sequence
                        visible_at[endpoint] = event.deliver_ns
                else:
                    raise ValueError("unknown routing event")
        if active or any(s.get_num_unfinished_requests() for s in schedulers):
            raise RuntimeError("dependency_blocked: routing campaign incomplete")
        if [s.kv_cache_manager.block_pool.get_num_free_blocks() for s in schedulers] != initial_free:
            raise ValueError("original scheduler did not release its physical KV blocks")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU routing adapter initialized CUDA")
        result.finish_ns = max(times[-1] for times in result.tokens.values())
        result.control_reads = len(ledger.reads)
        result.metrics(workload)
        return result, routes
    finally:
        bridge.close()
        policy.close()
