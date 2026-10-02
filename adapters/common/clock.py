"""Conservative, single-process actor protocol with registered message delivery."""
from dataclasses import dataclass, replace
import heapq
import time


@dataclass(frozen=True)
class Event:
    producer: str
    sequence: int
    recipient: str
    kind: str
    payload: str
    created_ns: int
    deliver_ns: int
    observed_ns: int | None = None


class Coordinator:
    def __init__(self, actors: tuple[str, ...]):
        if not actors or len(set(actors)) != len(actors):
            raise ValueError("actors must be distinct")
        self.now = 0
        self.bounds: dict[str, int | None] = dict.fromkeys(actors)
        self.sequence = dict.fromkeys(actors, 0)
        self.unregistered: set[tuple[str, int]] = set()
        self.events: list[tuple[int, int, Event]] = []
        self.serial = 0
        self.failed = False

    def resume(self, actor: str) -> None:
        if actor not in self.bounds:
            raise ValueError("unknown actor")
        self.bounds[actor] = None

    def certify(self, actor: str, lower_bound: int) -> None:
        if actor not in self.bounds or lower_bound < self.now:
            raise ValueError("invalid actor lower bound")
        self.bounds[actor] = lower_bound

    def begin_send(self, producer: str) -> tuple[str, int]:
        if self.bounds[producer] is not None:
            raise ValueError("producer must resume before emitting events")
        self.sequence[producer] += 1
        identity = (producer, self.sequence[producer])
        self.unregistered.add(identity)
        return identity

    def register(self, identity: tuple[str, int], recipient: str, kind: str, payload: str,
                 created_ns: int, deliver_ns: int) -> Event:
        if identity not in self.unregistered or recipient not in self.bounds or not self.now <= created_ns <= deliver_ns:
            raise ValueError("unregistered message or invalid causal timestamp")
        self.unregistered.remove(identity)
        event = Event(*identity, recipient, kind, payload, created_ns, deliver_ns)
        self.serial += 1
        heapq.heappush(self.events, (deliver_ns, self.serial, event))
        return event

    def send(self, producer: str, recipient: str, kind: str, payload: str,
             created_ns: int, deliver_ns: int) -> Event:
        return self.register(self.begin_send(producer), recipient, kind, payload, created_ns, deliver_ns)

    def advance(self) -> list[Event]:
        if self.failed:
            raise RuntimeError("protocol_failure")
        if self.unregistered or any(bound is None for bound in self.bounds.values()):
            raise RuntimeError("dependency_blocked: actor or message has no safe certificate")
        bounds = [bound for bound in self.bounds.values() if bound is not None]
        frontier = min(bounds + ([self.events[0][0]] if self.events else []))
        if frontier == self.now and (not self.events or self.events[0][0] != frontier):
            raise RuntimeError("dependency_blocked: certificate renewal required")
        self.now = frontier
        due = []
        while self.events and self.events[0][0] == self.now:
            event = heapq.heappop(self.events)[2]
            self.bounds[event.recipient] = None
            due.append(replace(event, observed_ns=self.now))
        return due


class PacedCoordinator(Coordinator):
    """The same registered-actor protocol, with real-time waits as an oracle."""
    def __init__(self, actors: tuple[str, ...]):
        super().__init__(actors)
        self.started = time.perf_counter_ns()

    def advance(self) -> list[Event]:
        events = super().advance()
        remaining = self.started + self.now - time.perf_counter_ns()
        if remaining > 0:
            time.sleep(remaining / 1e9)
        return events
