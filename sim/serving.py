"""FCFS request replay with explicit physical memory and whole-step costs."""
from dataclasses import dataclass, field
import heapq
import math
from typing import Literal, Protocol

from .cost import Cost
from .work import checked

Phase = Literal["prefill", "decode"]
State = Literal["waiting", "prefill", "decode", "cancelling", "completed", "cancelled", "rejected"]
SCOPE = "whole_step_including_host_and_communication"


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_ns: int
    prompt_len: int
    output_len: int
    cancel_ns: int | None = None

    def __post_init__(self) -> None:
        for value in (self.arrival_ns, self.prompt_len, self.output_len):
            checked(value)
        if not self.request_id or not self.prompt_len or not self.output_len:
            raise ValueError("request identity and positive lengths required")
        checked(self.prompt_len + self.output_len - 1)
        if self.cancel_ns is not None and checked(self.cancel_ns) < self.arrival_ns:
            raise ValueError("cancellation precedes arrival")


@dataclass(frozen=True)
class PageLayout:
    """Aggregate bytes/page on this physical device, from its actual backend layout."""
    tokens_per_page: int
    bytes_per_page: int
    identity: str

    def __post_init__(self) -> None:
        if not checked(self.tokens_per_page) or not checked(self.bytes_per_page) or not self.identity:
            raise ValueError("explicit page geometry and layout identity required")


@dataclass(frozen=True)
class DeviceMemory:
    device: str
    capacity_bytes: int
    weight_bytes: int
    pages: tuple[PageLayout, ...] = ()
    state_slots: int = 0
    state_bytes_per_slot: int = 0
    conv_bytes_per_slot: int = 0
    workspace_bytes: int = 0
    graph_bytes: int = 0
    communication_bytes: int = 0
    allocator_bytes: int = 0
    kv_pool_bytes: int = 0

    def __post_init__(self) -> None:
        for value in (self.capacity_bytes, self.weight_bytes, self.state_slots, self.state_bytes_per_slot,
                      self.conv_bytes_per_slot, self.workspace_bytes, self.graph_bytes,
                      self.communication_bytes, self.allocator_bytes, self.kv_pool_bytes):
            checked(value)
        if not self.device or not self.capacity_bytes or self.fixed_bytes > self.capacity_bytes:
            raise ValueError("invalid device identity or fixed residency exceeds capacity")
        if (self.state_bytes_per_slot or self.conv_bytes_per_slot) and not self.state_slots:
            raise ValueError("recurrent/conv storage needs an explicit slot pool")

    @property
    def fixed_bytes(self) -> int:
        return checked(self.weight_bytes + self.state_slots * (self.state_bytes_per_slot + self.conv_bytes_per_slot)
                       + self.workspace_bytes + self.graph_bytes + self.communication_bytes + self.allocator_bytes + self.kv_pool_bytes)

    def request_bytes(self, request: Request) -> int:
        # Reserve final context up front. This conservative policy prevents later
        # growth from silently evicting live state or granting free offload.
        length = request.prompt_len + request.output_len - 1
        return checked(sum(((length + page.tokens_per_page - 1) // page.tokens_per_page)
                           * page.bytes_per_page for page in self.pages))


@dataclass(frozen=True)
class BatchItem:
    request_id: str
    tokens: int
    context: int


@dataclass(frozen=True)
class Batch:
    phase: Phase
    items: tuple[BatchItem, ...]


class ServingCost(Protocol):
    scope: str
    def estimate(self, batch: Batch) -> Cost: ...


@dataclass(frozen=True)
class Policy:
    max_sequences: int
    prefill_token_budget: int
    admission: str = "fcfs_reserve_final_context"
    scheduling: str = "decode_first_homogeneous_phase"

    def __post_init__(self) -> None:
        if not checked(self.max_sequences) or not checked(self.prefill_token_budget):
            raise ValueError("positive sequence and prefill budgets required")
        if (self.admission, self.scheduling) != ("fcfs_reserve_final_context", "decode_first_homogeneous_phase"):
            raise ValueError("unsupported serving policy")


@dataclass
class Outcome:
    request: Request
    state: State = "waiting"
    admitted_ns: int | None = None
    terminal_ns: int | None = None
    prompt_processed: int = 0
    tokens_ns: list[int] = field(default_factory=list)
    discarded_tokens: int = 0


@dataclass(frozen=True)
class MemoryEvent:
    time_ns: int
    device: str
    request_id: str
    action: str
    used_bytes: int
    live_requests: int
    reserved_kv_bytes: int
    state_slot: int | None


@dataclass(frozen=True)
class BatchRun:
    batch: Batch
    start_ns: int
    end_ns: int
    cost: Cost


def percentiles(values: list[int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    def quantile(q: float, minimum: int) -> float | None:
        if len(ordered) < minimum:
            return None
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return {"samples": len(ordered), "p50_ns": quantile(.5, 1),
            "p95_ns": quantile(.95, 20), "p99_ns": quantile(.99, 100)}


@dataclass
class ServingReport:
    outcomes: tuple[Outcome, ...]
    batches: list[BatchRun]
    memory: list[MemoryEvent]
    peak_bytes: dict[str, int]
    begin_ns: int
    finish_ns: int
    policy: Policy
    oracle: str = "declared_fixed_length_tokens"
    observation: str = "first_arrival_through_drain; warmup=0"

    def metrics(self) -> dict[str, object]:
        completed = [out for out in self.outcomes if out.state == "completed"]
        visible = [out for out in self.outcomes if out.tokens_ns]
        ttft = [out.tokens_ns[0] - out.request.arrival_ns for out in visible]
        itl = [b - a for out in visible for a, b in zip(out.tokens_ns, out.tokens_ns[1:])]
        waits = [out.admitted_ns - out.request.arrival_ns for out in self.outcomes if out.admitted_ns is not None]
        emitted = sum(len(out.tokens_ns) for out in self.outcomes)
        return {"completed": len(completed), "cancelled": sum(out.state == "cancelled" for out in self.outcomes),
                "rejected": sum(out.state == "rejected" for out in self.outcomes), "emitted_tokens": emitted,
                "discarded_tokens": sum(out.discarded_tokens for out in self.outcomes),
                "latency_population": "all_requests_with_visible_tokens_including_cancelled",
                "tokens_per_second": emitted * 1e9 / (self.finish_ns - self.begin_ns) if emitted else 0,
                "ttft": percentiles(ttft), "itl": percentiles(itl), "waiting": percentiles(waits),
                "peak_bytes": self.peak_bytes, "begin_ns": self.begin_ns, "finish_ns": self.finish_ns,
                "observation": self.observation, "oracle": self.oracle}


class MemoryLedger:
    def __init__(self, devices: tuple[DeviceMemory, ...]):
        if not devices or len({d.device for d in devices}) != len(devices):
            raise ValueError("distinct physical device identities required")
        self.devices = devices
        self.allocations: dict[str, tuple[int, ...]] = {}
        self.slots: dict[str, tuple[int | None, ...]] = {}
        self.reserved = [0] * len(devices)
        self.used = [device.fixed_bytes for device in devices]
        self.peak = self.used.copy()
        self.events: list[MemoryEvent] = []

    def fits(self, request: Request, empty: bool = False) -> bool:
        return all((not device.state_slots or empty or len(self.allocations) < device.state_slots)
                   and ((0 if empty else reserved) + device.request_bytes(request) <= device.kv_pool_bytes if device.kv_pool_bytes
                        else (device.fixed_bytes if empty else used) + device.request_bytes(request) <= device.capacity_bytes)
                   for device, used, reserved in zip(self.devices, self.used, self.reserved))

    def reserve(self, request: Request, now: int) -> None:
        if request.request_id in self.allocations or not self.fits(request):
            raise ValueError("duplicate reservation or capacity violation")
        sizes = tuple(device.request_bytes(request) for device in self.devices)
        slots = []
        for rank, device in enumerate(self.devices):
            occupied = {values[rank] for values in self.slots.values()}
            slots.append(next(slot for slot in range(device.state_slots) if slot not in occupied) if device.state_slots else None)
        self.allocations[request.request_id] = sizes
        self.slots[request.request_id] = tuple(slots)
        self._record(request.request_id, now, sizes, tuple(slots), "reserve", 1)

    def release(self, request_id: str, now: int) -> None:
        self._record(request_id, now, self.allocations.pop(request_id), self.slots.pop(request_id), "release", -1)

    def _record(self, request_id: str, now: int, sizes: tuple[int, ...], slots: tuple[int | None, ...], action: str, sign: int) -> None:
        for rank, (device, size) in enumerate(zip(self.devices, sizes)):
            self.reserved[rank] += sign * size
            if not device.kv_pool_bytes:
                self.used[rank] += sign * size
            self.peak[rank] = max(self.peak[rank], self.used[rank])
            self.events.append(MemoryEvent(now, device.device, request_id, action, self.used[rank],
                                           len(self.allocations), self.reserved[rank], slots[rank]))


def run(requests: tuple[Request, ...], devices: tuple[DeviceMemory, ...], policy: Policy,
        costs: ServingCost) -> ServingReport:
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ValueError("nonempty unique request identities required")
    if costs.scope != SCOPE:
        raise ValueError("serving needs whole-step costs including sampling, host and communication")
    ledger = MemoryLedger(devices)
    outcomes = tuple(Outcome(request) for request in requests)
    controls = [(r.arrival_ns, 0, i) for i, r in enumerate(requests)]
    controls += [(r.cancel_ns, 1, i) for i, r in enumerate(requests) if r.cancel_ns is not None]
    heapq.heapify(controls)
    waiting: list[Outcome] = []
    active: list[Outcome] = []
    pending: BatchRun | None = None
    now, terminal = 0, 0
    report = ServingReport(outcomes, [], ledger.events, {}, min(r.arrival_ns for r in requests), 0, policy)

    def finish(out: Outcome, state: State) -> None:
        nonlocal terminal
        if out in active:
            active.remove(out)
            ledger.release(out.request.request_id, now)
        if out in waiting:
            waiting.remove(out)
        out.state, out.terminal_ns = state, now
        terminal += 1

    while terminal < len(requests):
        # A cancellation at the completion timestamp suppresses visibility but
        # cannot release storage until the already-submitted batch has completed.
        inflight = {item.request_id for item in pending.batch.items} if pending else set()
        while controls and controls[0][0] <= now:
            _, kind, index = heapq.heappop(controls)
            out = outcomes[index]
            if not kind:
                waiting.append(out)
            elif out.terminal_ns is None:
                if out.request.request_id in inflight:
                    out.state = "cancelling"
                else:
                    finish(out, "cancelled")
        if pending and pending.end_ns == now:
            selected = {item.request_id: item for item in pending.batch.items}
            for out in tuple(active):
                if out.request.request_id not in selected:
                    continue
                if pending.batch.phase == "prefill":
                    out.prompt_processed += selected[out.request.request_id].tokens
                if out.state == "cancelling":
                    out.discarded_tokens += int(pending.batch.phase == "decode" or out.prompt_processed == out.request.prompt_len)
                    finish(out, "cancelled")
                    continue
                if pending.batch.phase == "prefill":
                    if out.prompt_processed < out.request.prompt_len:
                        continue
                    out.state = "decode"
                out.tokens_ns.append(now)
                if len(out.tokens_ns) == out.request.output_len:
                    finish(out, "completed")
            pending = None
        while waiting and len(active) < policy.max_sequences:
            out = waiting[0]
            if not ledger.fits(out.request, empty=True):
                finish(out, "rejected")
            elif not ledger.fits(out.request):
                break  # FCFS: a smaller later request cannot overtake this one.
            else:
                waiting.pop(0)
                ledger.reserve(out.request, now)
                out.state, out.admitted_ns = "prefill", now
                active.append(out)
        if not pending and active:
            decoding = [out for out in active if out.state == "decode"]
            if decoding:
                batch = Batch("decode", tuple(BatchItem(out.request.request_id, 1,
                              out.request.prompt_len + len(out.tokens_ns)) for out in decoding))
            else:
                budget, items = policy.prefill_token_budget, []
                for out in active:
                    count = min(budget, out.request.prompt_len - out.prompt_processed)
                    if count:
                        items.append(BatchItem(out.request.request_id, count, out.prompt_processed + count))
                        budget -= count
                batch = Batch("prefill", tuple(items))
            cost = costs.estimate(batch)
            if cost.standalone_ns is None or cost.standalone_ns <= 0:
                raise ValueError(f"unsupported/nonpositive {batch.phase} cost: {cost.reason}")
            pending = BatchRun(batch, now, checked(now + cost.standalone_ns), cost)
            report.batches.append(pending)
        if terminal == len(requests):
            break
        next_times = ([controls[0][0]] if controls else []) + ([pending.end_ns] if pending else [])
        if not next_times:
            raise RuntimeError("request replay deadlocked")
        now = min(next_times)
    report.finish_ns = now
    report.peak_bytes = {device.device: peak for device, peak in zip(devices, ledger.peak)}
    if ledger.allocations or any(len(out.tokens_ns) > out.request.output_len for out in outcomes):
        raise RuntimeError("request/token or memory conservation failed")
    return report
