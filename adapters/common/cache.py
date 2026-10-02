"""Physical KV ownership and delayed router visibility are separate ledgers."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlockIdentity:
    model: str
    tokenizer: str
    adapter: str
    prefix: str
    layout: str


@dataclass
class Block:
    identity: BlockIdentity
    owner: str
    generation: int
    bytes: int
    ready_ns: int
    references: set[str] = field(default_factory=set)


class CacheLedger:
    def __init__(self, capacity: int, fixed_bytes: int = 0):
        if not 0 <= fixed_bytes <= capacity:
            raise ValueError("invalid cache budget")
        self.capacity = capacity
        self.fixed_bytes = fixed_bytes
        self.blocks: dict[str, Block] = {}
        self.graph_pools: dict[str, tuple[int, set[int]]] = {}

    @property
    def used_bytes(self) -> int:
        return self.fixed_bytes + sum(b.bytes for b in self.blocks.values()) + sum(p[0] for p in self.graph_pools.values())

    def allocate(self, allocation: str, block: Block) -> None:
        if allocation in self.blocks or block.bytes <= 0 or block.ready_ns < 0 or block.generation < 0:
            raise ValueError("invalid physical block")
        if self.used_bytes + block.bytes > self.capacity:
            raise MemoryError("KV allocation exceeds available memory")
        self.blocks[allocation] = block

    def acquire(self, allocation: str, request: str, now: int) -> None:
        block = self.blocks[allocation]
        if now < block.ready_ns:
            raise RuntimeError("dependency_blocked: KV transfer not complete")
        block.references.add(request)

    def release(self, allocation: str, request: str) -> None:
        self.blocks[allocation].references.discard(request)

    def evict(self, allocation: str) -> None:
        if self.blocks[allocation].references:
            raise RuntimeError("live physical KV references")
        del self.blocks[allocation]

    def reserve_graph(self, pool: str, bytes: int) -> None:
        if pool in self.graph_pools or bytes <= 0:
            raise ValueError("invalid graph pool")
        if self.used_bytes + bytes > self.capacity:
            raise MemoryError("graph pool exceeds available memory")
        self.graph_pools[pool] = (bytes, set())

    def replay_graph(self, pool: str, invocation: int) -> None:
        pending = self.graph_pools[pool][1]
        if invocation in pending:
            raise ValueError("duplicate graph invocation")
        pending.add(invocation)

    def complete_graph(self, pool: str, invocation: int) -> None:
        self.graph_pools[pool][1].remove(invocation)

    def free_graph(self, pool: str) -> None:
        if self.graph_pools[pool][1]:
            raise RuntimeError("dependency_blocked: graph pool still executing")
        del self.graph_pools[pool]


@dataclass(frozen=True)
class CacheUpdate:
    endpoint: str
    generation: int
    sequence: int
    visible_ns: int
    blocks: frozenset[BlockIdentity]


class CacheVisibility:
    """Full snapshots, ordered within endpoint generations; not delta events."""
    def __init__(self) -> None:
        self.snapshots: dict[str, CacheUpdate] = {}

    def apply(self, update: CacheUpdate, now: int) -> bool:
        if update.visible_ns > now:
            raise RuntimeError("cache update not yet visible")
        previous = self.snapshots.get(update.endpoint)
        if previous and (update.generation, update.sequence) <= (previous.generation, previous.sequence):
            return False
        self.snapshots[update.endpoint] = update
        return True
