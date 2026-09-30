"""Explicit collective rendezvous; no implicit global barrier or data synthesis."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Collective:
    communicator: str
    generation: int
    sequence: int
    kind: str
    dtype: str
    count: int
    ranks: tuple[int, ...]
    graph_replay: int = 0


class CollectiveMatcher:
    def __init__(self) -> None:
        self.operations: dict[tuple[str, int, int, int], tuple[Collective, dict[int, int]]] = {}

    def join(self, operation: Collective, rank: int, ready_ns: int) -> int | None:
        if not operation.communicator or operation.count < 0 or ready_ns < 0 or rank not in operation.ranks or len(set(operation.ranks)) != len(operation.ranks):
            raise ValueError("invalid collective participant")
        identity = (operation.communicator, operation.generation, operation.sequence, operation.graph_replay)
        registered, ranks = self.operations.setdefault(identity, (operation, {}))
        if registered != operation or rank in ranks:
            raise ValueError("collective metadata mismatch or duplicate rank")
        ranks[rank] = ready_ns
        return max(ranks.values()) if len(ranks) == len(operation.ranks) else None
