"""Explicit EP payloads and chunk barriers, using the shared C++ resource clock."""
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

from .resources import Capacity, Engine, Phase, Work
from .work import checked, size

Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TokenRoute:
    source: int
    token: int
    experts: tuple[int, ...]


@dataclass(frozen=True)
class Payload:
    dtype: str
    rows: Literal["assignments", "unique_tokens"]
    assignment_metadata_bytes: int = 0
    token_metadata_bytes: int = 0
    message_bytes: int = 0

    def __post_init__(self) -> None:
        size(self.dtype)
        for value in (self.assignment_metadata_bytes, self.token_metadata_bytes, self.message_bytes):
            checked(value)
        if self.rows not in ("assignments", "unique_tokens"):
            raise ValueError("explicit dispatch deduplication/combine reduction required")

    def bytes(self, assignments: int, unique: int, hidden: int) -> int:
        rows = unique if self.rows == "unique_tokens" else assignments
        return checked(rows * hidden * size(self.dtype) + assignments * self.assignment_metadata_bytes
                       + unique * self.token_metadata_bytes + (self.message_bytes if assignments else 0))


@dataclass(frozen=True)
class Ledger:
    assignments: Matrix
    unique_tokens: Matrix
    dispatch_bytes: Matrix  # [source][target], including separately visible local bytes
    combine_bytes: Matrix  # [target][source]; independently declared dtype/protocol
    protocol: str


def ledger(routes: tuple[TokenRoute, ...], expert_ranks: tuple[int, ...], ranks: int,
           hidden: int, dispatch: Payload, combine: Payload, protocol: str) -> Ledger:
    if not checked(ranks) or not checked(hidden) or not expert_ranks or not protocol:
        raise ValueError("explicit ranks, placement, hidden width and protocol required")
    if any(type(rank) is not int or not 0 <= rank < ranks for rank in expert_ranks):
        raise ValueError("expert placement exceeds rank count")
    assignments = [[0] * ranks for _ in range(ranks)]
    unique = [[set() for _ in range(ranks)] for _ in range(ranks)]
    seen: set[tuple[int, int]] = set()
    for route in routes:
        key = (checked(route.source), checked(route.token))
        if route.source >= ranks or key in seen or not route.experts or len(set(route.experts)) != len(route.experts):
            raise ValueError("duplicate token, invalid source or repeated/empty expert selection")
        seen.add(key)
        for expert in route.experts:
            if checked(expert) >= len(expert_ranks):
                raise ValueError("route references a nonresident expert")
            target = expert_ranks[expert]
            assignments[route.source][target] += 1
            unique[route.source][target].add(route.token)
    a = tuple(tuple(checked(count) for count in row) for row in assignments)
    u = tuple(tuple(len(tokens) for tokens in row) for row in unique)
    outgoing = tuple(tuple(dispatch.bytes(a[r][s], u[r][s], hidden) for s in range(ranks)) for r in range(ranks))
    returning = tuple(tuple(combine.bytes(a[s][r], u[s][r], hidden) for s in range(ranks)) for r in range(ranks))
    return Ledger(a, u, outgoing, returning, protocol)


@dataclass(frozen=True)
class Device:
    identity: str
    dram_bytes_per_ns: float
    workspace_bytes: int


@dataclass(frozen=True)
class Link:
    identity: str
    bytes_per_ns: float
    startup_ns: int = 0
    duplex: Literal["shared", "directional"] = "directional"


@dataclass(frozen=True)
class Topology:
    devices: tuple[Device, ...]
    links: tuple[Link, ...]
    paths: tuple[tuple[int, int, tuple[int, ...]], ...]
    identity: str
    measured: bool = False

    def __post_init__(self) -> None:
        if not self.identity or not self.devices or len({d.identity for d in self.devices}) != len(self.devices):
            raise ValueError("distinct physical node/device identities required")
        if len({link.identity for link in self.links}) != len(self.links):
            raise ValueError("duplicate physical link")
        for link in self.links:
            checked(link.startup_ns)
            if link.duplex not in ("shared", "directional"):
                raise ValueError("invalid duplex policy")
        seen = set()
        for source, target, path in self.paths:
            if (source, target) in seen or source == target or not path or len(set(path)) != len(path):
                raise ValueError("duplicate/local/empty transfer path")
            if any(checked(rank) >= len(self.devices) for rank in (source, target)) or any(checked(i) >= len(self.links) for i in path):
                raise ValueError("path exceeds physical topology")
            seen.add((source, target))
        self.capacities()  # Validate finite rates and buffer capacities once.

    def capacities(self) -> tuple[Capacity, ...]:
        resources = tuple(c for d in self.devices for c in (Capacity(d.identity + "/dram", d.dram_bytes_per_ns),
                          Capacity(d.identity + "/sm", 1), Capacity(d.identity + "/buffer", resident=d.workspace_bytes)))
        return resources + tuple(Capacity(link.identity, link.bytes_per_ns) for link in self.links)

    def traffic(self, matrix: Matrix) -> tuple[tuple[int, ...], int]:
        ranks = len(self.devices)
        if len(matrix) != ranks or any(len(row) != ranks for row in matrix):
            raise ValueError("traffic dimensions differ from topology")
        paths = {(r, s): path for r, s, path in self.paths}
        demand, startup = [0] * len(self.links), 0
        for r, row in enumerate(matrix):
            for s, count in enumerate(row):
                if not checked(count) or r == s:
                    continue  # Local traffic never enters a remote link.
                if (r, s) not in paths:
                    raise ValueError("remote payload has no physical path")
                path = paths[r, s]
                startup = max(startup, checked(sum(self.links[i].startup_ns for i in path)))
                for i in path:
                    demand[i] = checked(demand[i] + count)
        return tuple(demand), startup

    def lower_bound_ns(self, matrix: Matrix) -> int:
        demand, _ = self.traffic(matrix)
        return checked(math.ceil(max((n / link.bytes_per_ns for n, link in zip(demand, self.links)), default=0)))


@dataclass(frozen=True)
class StageCost:
    """Payload/device phase only; protocol startup is supplied by topology."""
    isolated_ns: int
    dram_bytes: tuple[int, ...]
    sm_ns: tuple[int, ...]
    source: str
    qualified: bool = False

    def __post_init__(self) -> None:
        if not checked(self.isolated_ns) or not self.source:
            raise ValueError("positive standalone cost and source required")
        for value in (*self.dram_bytes, *self.sm_ns):
            checked(value)


@dataclass(frozen=True)
class Chunk:
    identity: int
    routes: tuple[TokenRoute, ...]
    dispatch: StageCost
    compute: StageCost
    combine: StageCost
    buffers: tuple[int, ...]


@dataclass(frozen=True)
class Snapshot:
    time_ns: int
    throughput: tuple[float, ...]
    resident: tuple[int, ...]
    generations: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class Report:
    ledgers: tuple[Ledger, ...]
    times: tuple[tuple[int, int, int], ...]
    snapshots: tuple[Snapshot, ...]
    finish_ns: int
    mode: str
    scope: str


def run(chunks: tuple[Chunk, ...], topology: Topology, expert_ranks: tuple[int, ...], hidden: int,
        dispatch: Payload, combine: Payload, protocol: str, bridge: Path, pipeline: bool = False) -> Report:
    if not chunks or len({c.identity for c in chunks}) != len(chunks):
        raise ValueError("nonempty unique chunks required")
    ranks, capacities = len(topology.devices), topology.capacities()
    seen: set[tuple[int, int]] = set()
    ledgers, works = [], []
    for chunk in chunks:
        keys = {(r.source, r.token) for r in chunk.routes}
        if seen & keys:
            raise ValueError("source token was dispatched by multiple chunks")
        seen.update(keys)
        traffic = ledger(chunk.routes, expert_ranks, ranks, hidden, dispatch, combine, protocol)
        ledgers.append(traffic)
        phases = []
        for cost, matrix in ((chunk.dispatch, traffic.dispatch_bytes), (chunk.compute, None), (chunk.combine, traffic.combine_bytes)):
            if len(cost.dram_bytes) != ranks or len(cost.sm_ns) != ranks or len(chunk.buffers) != ranks:
                raise ValueError("per-rank costs/buffers differ from placement")
            links, startup = topology.traffic(matrix) if matrix is not None else ((0,) * len(topology.links), 0)
            if startup:
                phases.append(Phase(startup, (0,) * len(capacities), (0,) * len(capacities)))
            demand = tuple(v for dram, sm in zip(cost.dram_bytes, cost.sm_ns) for v in (dram, sm, 0)) + links
            phases.append(Phase(cost.isolated_ns, demand, (0,) * len(capacities)))
        held = tuple(v for count in chunk.buffers for v in (0, 0, checked(count))) + (0,) * len(topology.links)
        works.append(Work(chunk.identity, 0, tuple(phases), () if pipeline or not works else (works[-1].identity,), held))
    engine = Engine(bridge, capacities)
    snapshots = []
    try:
        for work in works:
            engine.submit(work)
        while True:
            used, resident, generations = [0.] * len(capacities), [0] * len(capacities), []
            for work in works:
                progress = engine.progress(work.identity)
                start, end = engine.times(work.identity)
                generations.append((work.identity, progress.generation, progress.deadline_ns))
                if start >= 0 and end < 0:
                    resident = [a + b for a, b in zip(resident, work.held)]
                if progress.state == 2:
                    phase = work.phases[progress.phase]
                    used = [a + progress.rate * b for a, b in zip(used, phase.demand)]
                    resident = [a + b for a, b in zip(resident, phase.resident)]
            if any(use > c.throughput * (1 + 1e-12) or count > c.resident for c, use, count in zip(capacities, used, resident)):
                raise RuntimeError("resource capacity conservation failed")
            snapshots.append(Snapshot(engine.now, tuple(used), tuple(resident), tuple(generations)))
            next_ns = engine.next_event()
            if next_ns < 0:
                break
            engine.advance(next_ns)
        times = tuple((work.identity, *engine.times(work.identity)) for work in works)
        if any(end < 0 for _, _, end in times):
            raise RuntimeError("EP dependencies/buffers deadlocked")
        qualified = topology.measured and all(cost.qualified for c in chunks for cost in (c.dispatch, c.compute, c.combine))
        return Report(tuple(ledgers), times, tuple(snapshots), engine.now, "chunk_pipeline" if pipeline else "serial_chunks",
                      "qualified_declared_phase_scenario" if qualified else "analytic_topology_scenario")
    finally:
        engine.close()
