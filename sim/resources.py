"""Typed embedding of the existing C++ capacity-sharing engine; EOF teardown."""
from dataclasses import dataclass
import math
from pathlib import Path

from adapters.common.bridge import LineProcess
from .cost import Identity, Profile
from .work import checked


@dataclass(frozen=True)
class Capacity:
    name: str
    throughput: float = 0
    resident: int = 0

    def __post_init__(self) -> None:
        checked(self.resident)
        if not self.name or not math.isfinite(self.throughput) or self.throughput < 0 or not (self.throughput or self.resident):
            raise ValueError("positive throughput or residency capacity required")


@dataclass(frozen=True)
class Phase:
    isolated_ns: int
    demand: tuple[float, ...]
    resident: tuple[int, ...]

    def __post_init__(self) -> None:
        if not checked(self.isolated_ns):
            raise ValueError("positive isolated service required")
        for value in self.resident:
            checked(value)
        if any(not math.isfinite(v) or v < 0 for v in self.demand):
            raise ValueError("nonnegative finite resource demand required")


def profile_phase(profile: Profile, features: tuple[int, ...], identity: Identity,
                  resident: tuple[int, ...]) -> Phase:
    """Charge a qualified discrete combination once, with explicit exclusivity."""
    if profile.identity != identity or not any(resident):
        raise ValueError("matching measurement identity and a reserved combination lane required")
    cost = profile.lookup(features, candidate=False)
    if cost.standalone_ns is None or cost.status != "calibrated_estimate":
        raise ValueError("unqualified or unsupported discrete combination")
    return Phase(cost.standalone_ns, (0,) * len(resident), resident)


@dataclass(frozen=True)
class Work:
    identity: int
    arrival_ns: int
    phases: tuple[Phase, ...]
    dependencies: tuple[int, ...] = ()
    held: tuple[int, ...] = ()
    weight: float = 1


@dataclass(frozen=True)
class Progress:
    state: int
    phase: int
    remaining: float
    rate: float
    generation: int
    deadline_ns: int


class Engine(LineProcess):
    def __init__(self, executable: Path, capacities: tuple[Capacity, ...]):
        if not capacities or len({c.name for c in capacities}) != len(capacities):
            raise ValueError("nonempty unique resources required")
        self.capacities = capacities
        self.identities: set[int] = set()
        self.now = 0
        super().__init__([str(executable), "--resources"])
        fields = [str(len(capacities)), *(str(v) for c in capacities for v in (c.throughput, c.resident))]
        self._ok("capacities " + " ".join(fields))

    def _ok(self, command: str) -> None:
        if self.command(command) != "ok":
            raise RuntimeError("resource embedding rejected command")

    def submit(self, work: Work) -> None:
        count = len(self.capacities)
        held = work.held or (0,) * count
        if not checked(work.identity) or work.identity in self.identities or checked(work.arrival_ns) < self.now or not work.phases:
            raise ValueError("invalid resource identity, arrival or phases")
        if len(held) != count or len(set(work.dependencies)) != len(work.dependencies) or not set(work.dependencies) <= self.identities:
            raise ValueError("held-resource dimensions or dependencies differ")
        if not math.isfinite(work.weight) or work.weight <= 0:
            raise ValueError("positive finite scheduling weight required")
        for phase in work.phases:
            if len(phase.demand) != count or len(phase.resident) != count:
                raise ValueError("phase resource dimensions differ")
            for c, demand, resident, reservation in zip(self.capacities, phase.demand, phase.resident, held):
                if resident + checked(reservation) > c.resident or (demand and not c.throughput):
                    raise ValueError("phase exceeds declared resource support")
        fields = [work.identity, work.arrival_ns, work.weight, len(work.dependencies), *work.dependencies,
                  *held, len(work.phases)]
        for phase in work.phases:
            fields.extend((phase.isolated_ns, *phase.demand, *phase.resident))
        self._ok("work " + " ".join(str(v) for v in fields))
        self.identities.add(work.identity)

    def advance(self, time_ns: int) -> None:
        if checked(time_ns) < self.now:
            raise ValueError("resource clock moved backwards")
        self._ok(f"advance {time_ns}")
        self.now = time_ns

    def next_event(self) -> int:
        return int(self.command("next"))

    def progress(self, identity: int) -> Progress:
        if identity not in self.identities:
            raise ValueError("unknown resource work")
        fields = self.command(f"progress {identity}").split()
        return Progress(int(fields[0]), int(fields[1]), float(fields[2]), float(fields[3]), int(fields[4]), int(fields[5]))

    def times(self, identity: int) -> tuple[int, int]:
        if identity not in self.identities:
            raise ValueError("unknown resource work")
        start, end = self.command(f"status {identity}").split()
        return int(start), int(end)
