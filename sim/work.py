"""Schema-1 descriptors. Routes and state slots are caller-supplied control data."""
from dataclasses import asdict, dataclass, replace
import json
import random

LIMIT = (1 << 63) - 1
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}


def checked(value: int) -> int:
    if type(value) is not int or not 0 <= value <= LIMIT:
        raise ValueError("expected a nonnegative signed-64-bit count/ns/byte value")
    return value


def size(dtype: str) -> int:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype: {dtype}")
    return DTYPE_BYTES[dtype]


@dataclass(frozen=True)
class Execution:
    process: str = "workload"
    context: str = "0"
    instance: str = "0"
    node: str = "0"
    sequence: int = 0
    rank: int = 0
    stream: int = 0


@dataclass(frozen=True)
class MoE:
    n: int
    d: int
    e: int
    k: int
    m: int
    routes: tuple[tuple[int, ...], ...]
    tile_m: int = 1
    shared_m: int = 0
    weight_dtype: str = "bf16"
    activation_dtype: str = "bf16"
    accum_dtype: str = "fp32"
    routing: str = "recorded"

    @property
    def counts(self) -> tuple[int, ...]:
        counts = [0] * self.e
        for row in self.routes:
            for expert in row:
                counts[expert] += 1
        return tuple(counts)


@dataclass(frozen=True)
class TopK:
    n: int
    e: int
    k: int
    dtype: str = "fp32"
    sorted: bool = True
    tie_break: str = "backend_unspecified"
    finite_only: bool = True


@dataclass(frozen=True)
class KDA:
    b: int
    d: int
    h_qk: int
    h_v: int
    d_k: int
    d_v: int
    state_slots: tuple[int, ...]
    seq_lens: tuple[int, ...]
    state_dtype: str = "fp32"
    weight_dtype: str = "bf16"
    activation_dtype: str = "bf16"
    accum_dtype: str = "fp32"
    state_layout: str = "key_value"
    state_capacity: int | None = None
    state_policy: str = "fixed_slots"

    @property
    def capacity(self) -> int:
        return self.b if self.state_capacity is None else self.state_capacity


@dataclass(frozen=True)
class KDAPrefill(KDA):
    chunk_lens: tuple[int, ...] = ()
    chunk_size: int = 64
    conv_width: int = 4


@dataclass(frozen=True)
class Unknown:
    symbol: str
    grid: tuple[int, ...]
    block: tuple[int, ...]
    shared_bytes: int = 0


Spec = MoE | TopK | KDA | KDAPrefill | Unknown


@dataclass(frozen=True)
class Work:
    op_id: str
    spec: Spec
    backend: str
    revision: str
    execution: Execution = Execution()
    deps: tuple[str, ...] = ()
    phase: str = "decode"
    layer: int = 0
    fusion_group: str = ""
    graph_id: str = ""
    trace_id: str = ""
    variants: tuple[tuple[str, str], ...] = ()
    schema: int = 1

    def validate(self) -> None:
        if type(self.schema) is not int or self.schema != 1 or not all(isinstance(value, str) and value for value in (self.op_id, self.backend, self.revision)):
            raise ValueError("schema/identity/backend revision missing or unsupported")
        x, s = self.execution, self.spec
        for value in (x.sequence, x.rank, x.stream, self.layer):
            checked(value)
        if not all((x.process, x.context, x.instance, x.node)):
            raise ValueError("execution identity is incomplete")
        if len(set(self.deps)) != len(self.deps) or self.op_id in self.deps:
            raise ValueError("duplicate or self dependency")
        if len({name for name, _ in self.variants}) != len(self.variants) or any(not name or not variant for name, variant in self.variants):
            raise ValueError("stage variants must have distinct explicit identities")
        if isinstance(s, Unknown):
            if not s.symbol or len(s.grid) != 3 or len(s.block) != 3:
                raise ValueError("unknown kernel must retain launch metadata")
            for value in (*s.grid, *s.block, s.shared_bytes):
                checked(value)
            return
        if self.phase != ("prefill" if isinstance(s, KDAPrefill) else "decode"):
            raise ValueError("phase needs its matching decode or independent prefill descriptor")
        if isinstance(s, (MoE, TopK)):
            for value in (s.n, s.e, s.k):
                checked(value)
            if not 1 <= s.k <= s.e:
                raise ValueError("require 1 <= k <= E")
        if isinstance(s, MoE):
            for value in (s.d, s.m, s.tile_m, s.shared_m):
                checked(value)
            if min(s.d, s.m, s.tile_m) == 0 or len(s.routes) != s.n:
                raise ValueError("invalid MoE dimensions or route row count")
            if s.routing not in ("recorded", "uniform", "hot"):
                raise ValueError("unsupported routing provenance")
            for row in s.routes:
                if len(row) != s.k or len(set(row)) != s.k or any(type(e) is not int or not 0 <= e < s.e for e in row):
                    raise ValueError("routing must select k distinct resident experts")
            for dtype in (s.weight_dtype, s.activation_dtype, s.accum_dtype):
                size(dtype)
        elif isinstance(s, TopK):
            size(s.dtype)
            if type(s.sorted) is not bool or type(s.finite_only) is not bool or not s.tie_break:
                raise ValueError("selection semantics must be explicit")
        else:
            for value in (s.b, s.d, s.h_qk, s.h_v, s.d_k, s.d_v):
                checked(value)
            if min(s.d, s.h_qk, s.h_v, s.d_k, s.d_v) == 0 or s.h_qk != s.h_v:
                raise ValueError("invalid KDA dimensions or unsupported grouped value heads")
            if s.accum_dtype != "fp32" or s.state_layout not in ("key_value", "value_key"):
                raise ValueError("unsupported KDA accumulation or state layout")
            if s.state_policy not in ("fixed_slots", "rotating_slots"):
                raise ValueError("state reuse policy must be explicit")
            if len(s.state_slots) != s.b or len(s.seq_lens) != s.b:
                raise ValueError("one state slot/history length per sequence required")
            for slot in s.state_slots:
                if type(slot) is not int or slot < -1 or slot > LIMIT:
                    raise ValueError("invalid state slot; -1 denotes padding")
            live = [slot for slot in s.state_slots if slot >= 0]
            checked(s.capacity)
            if any(slot >= s.capacity for slot in live):
                raise ValueError("state slot exceeds declared capacity")
            if len(live) != len(set(live)):
                raise ValueError("live state slots must be distinct")
            for length in s.seq_lens:
                checked(length)
            for dtype in (s.state_dtype, s.weight_dtype, s.activation_dtype):
                size(dtype)
            if isinstance(s, KDAPrefill):
                if not checked(s.chunk_size) or not checked(s.conv_width) or len(s.chunk_lens) != s.b:
                    raise ValueError("prefill needs explicit chunk lengths, tile and convolution width")
                for slot, length in zip(s.state_slots, s.chunk_lens):
                    if checked(length) and slot < 0:
                        raise ValueError("padding cannot own prefill tokens")

    def dumps(self) -> str:
        value = asdict(self)
        value["kind"] = type(self.spec).__name__
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def loads(payload: str) -> "Work":
        value = json.loads(payload)
        required = {"op_id", "spec", "backend", "revision", "execution", "deps", "phase", "layer", "fusion_group", "graph_id", "trace_id", "schema", "kind"}
        if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"variants"}:
            raise ValueError("descriptor fields do not match schema 1")
        # These are small, fixed constructor schemas, never executable class names.
        constructors = {"MoE": MoE, "TopK": TopK, "KDA": KDA, "KDAPrefill": KDAPrefill, "Unknown": Unknown}
        if value["kind"] not in constructors:
            raise ValueError("unknown descriptor kind")
        shape = value["spec"]
        for name in ("routes", "state_slots", "seq_lens", "chunk_lens", "grid", "block"):
            if name in shape:
                shape[name] = tuple(tuple(row) for row in shape[name]) if name == "routes" else tuple(shape[name])
        spec = constructors[value.pop("kind")](**value.pop("spec"))
        value["deps"] = tuple(value["deps"])
        value["variants"] = tuple(tuple(pair) for pair in value.get("variants", ()))
        value["execution"] = Execution(**value["execution"])
        work = Work(spec=spec, **value)
        work.validate()
        return work


def routes(n: int, e: int, k: int, seed: int, hot: int | None = None) -> tuple[tuple[int, ...], ...]:
    checked(n)
    if not 1 <= k <= (hot if hot is not None else e) <= e:
        raise ValueError("invalid routing domain")
    rng = random.Random(seed)
    return tuple(tuple(rng.sample(range(e if hot is None else hot), k)) for _ in range(n))


class Registry:
    def __init__(self, works: tuple[Work, ...]):
        self.works = works
        keys: set[Execution] = set()
        ids = {work.op_id for work in works}
        if len(ids) != len(works):
            raise ValueError("duplicate op_id")
        for work in works:
            work.validate()
            if work.execution in keys or any(dep not in ids for dep in work.deps):
                raise ValueError("duplicate execution key or dangling dependency")
            keys.add(work.execution)
        pending = {work.op_id: set(work.deps) for work in works}
        finished: set[str] = set()
        while pending:
            ready = {op for op, deps in pending.items() if deps <= finished}
            if not ready:
                raise ValueError("cyclic logical dependencies")
            finished.update(ready)
            pending = {op: deps for op, deps in pending.items() if op not in ready}


@dataclass(frozen=True)
class Graph:
    graph_id: str
    templates: tuple[Work, ...]
    max_tokens: int
    state_capacity: int

    def bind(self, instance: str, updates: dict[str, Spec]) -> Registry:
        checked(self.max_tokens)
        checked(self.state_capacity)
        if not instance or set(updates) != {work.op_id for work in self.templates}:
            raise ValueError("every graph node needs a fresh binding")
        bound = []
        for template in self.templates:
            spec = updates[template.op_id]
            if type(spec) is not type(template.spec):
                raise ValueError("graph operator kind changed")
            expected = template.spec
            if isinstance(expected, MoE) and isinstance(spec, MoE):
                expected = replace(expected, n=spec.n, routes=spec.routes, routing=spec.routing)
            elif isinstance(expected, TopK) and isinstance(spec, TopK):
                expected = replace(expected, n=spec.n)
            elif isinstance(expected, KDA) and isinstance(spec, KDA):
                expected = replace(expected, b=spec.b, state_slots=spec.state_slots, seq_lens=spec.seq_lens)
                if isinstance(expected, KDAPrefill) and isinstance(spec, KDAPrefill):
                    expected = replace(expected, chunk_lens=spec.chunk_lens)
            if expected != spec:
                raise ValueError("graph static shape/precision/layout changed")
            count = spec.b if isinstance(spec, KDA) else spec.n if isinstance(spec, (MoE, TopK)) else 0
            if count > self.max_tokens:
                raise ValueError("graph shape exceeds captured capacity")
            if isinstance(spec, KDAPrefill) and sum(spec.chunk_lens) > self.max_tokens:
                raise ValueError("prefill tokens exceed captured capacity")
            if isinstance(spec, KDA) and any(slot >= self.state_capacity for slot in spec.state_slots):
                raise ValueError("state slot exceeds captured storage")
            bound.append(replace(template, spec=spec, graph_id=self.graph_id,
                                 execution=replace(template.execution, instance=instance)))
        return Registry(tuple(bound))
