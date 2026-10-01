"""Pure work lowering. Algorithm work, executed padding, traffic and storage differ."""
from dataclasses import dataclass
from .work import KDA, KDAPrefill, MoE, TopK, Unknown, Work, checked, size


@dataclass(frozen=True)
class Stage:
    name: str
    category: str
    family: tuple[str, ...]
    features: tuple[int, ...]
    tc_flops: int = 0
    simt_flops: int | None = 0
    selection_items: int = 0
    logical_bytes: int = 0
    deps: tuple[str, ...] = ()
    dram_bytes: int | None = None


@dataclass(frozen=True)
class Lowered:
    stages: tuple[Stage, ...]
    logical: dict[str, int]
    executed: dict[str, int]
    traffic: dict[str, int | None]
    residency: dict[str, int | None]


def lower(work: Work) -> Lowered:
    work.validate()
    s = work.spec
    stages: list[Stage] = []
    logical: dict[str, int] = {}
    executed: dict[str, int] = {}
    traffic: dict[str, int | None] = {}
    resident: dict[str, int | None] = {}
    variants = dict(work.variants)

    def stage(name: str, category: str, family: tuple[object, ...], features: tuple[int, ...],
              tc: int = 0, simt: int | None = 0, selection: int = 0, byte_count: int = 0) -> None:
        for value in (*features, tc, selection, byte_count):
            checked(value)
        if simt is not None:
            checked(simt)
        identity = tuple(str(item) for item in family)
        if name in variants:
            identity += ("compiled_variant=" + variants[name],)
        stages.append(Stage(name, category, identity, features,
                            tc, simt, selection, byte_count,
                            (stages[-1].name,) if stages else ()))

    if isinstance(s, Unknown):
        stage("unknown", "other", (s.symbol, s.grid, s.block, s.shared_bytes), (1,), simt=None)
    elif isinstance(s, TopK):
        family = (s.e, s.k, s.dtype, s.sorted, s.tie_break, s.finite_only)
        byte_count = checked(s.n * (s.e * size(s.dtype) + s.k * (size(s.dtype) + 8)))
        stage("topk", "other", family, (s.n,), selection=s.n * s.e, byte_count=byte_count)
        logical = {"rows": s.n, "elements": checked(s.n * s.e), "selected": checked(s.n * s.k)}
        executed = {"selection_items": logical["elements"]}
        traffic = {"logical_bytes": byte_count}
        resident = {"scores_bytes": checked(s.n * s.e * size(s.dtype))}
    elif isinstance(s, MoE):
        counts = s.counts
        active = sum(count > 0 for count in counts)
        padded = tuple(((count + s.tile_m - 1) // s.tile_m) * s.tile_m for count in counts)
        assignments, padded_rows = checked(s.n * s.k), checked(sum(padded))
        wa, aa = size(s.weight_dtype), size(s.activation_dtype)
        family = (s.d, s.e, s.k, s.m, s.tile_m, s.shared_m, s.weight_dtype, s.activation_dtype, s.accum_dtype)
        router = checked(2 * s.n * s.d * s.e)
        expert = checked(6 * s.d * s.m * assignments)
        expert_padded = checked(6 * s.d * s.m * padded_rows)
        cold = checked(3 * s.d * s.m * wa * active)
        logical = {"router_flops": router, "expert_flops": expert, "assignments": assignments,
                   "active_experts": active, "shared_flops": checked(6 * s.n * s.d * s.shared_m)}
        executed = {"expert_flops": expert_padded, "padded_rows": padded_rows}
        traffic = {"cold_expert_weight_bytes": cold, "router_weight_bytes": checked(s.d * s.e * wa)}
        resident = {"expert_weight_bytes": checked(3 * s.d * s.m * wa * s.e),
                    "router_weight_bytes": traffic["router_weight_bytes"],
                    "shared_weight_bytes": checked(3 * s.d * s.shared_m * wa)}
        stage("router", "moe", family, (s.n,), tc=router, byte_count=traffic["router_weight_bytes"])
        stage("score", "moe", family, (s.n,), simt=None, byte_count=2 * s.n * s.e * 4)
        stage("topk", "moe", family, (s.n,), selection=s.n * s.e)
        stage("route", "moe", family, (assignments,), simt=None, byte_count=assignments * 8)
        stage("permute", "moe", family, (assignments, active), byte_count=2 * assignments * s.d * aa)
        stage("gate_up", "moe", family, (padded_rows, active), tc=4 * padded_rows * s.d * s.m,
              byte_count=2 * s.d * s.m * wa * active)
        stage("activation", "moe", family, (padded_rows, active), simt=None,
              byte_count=3 * padded_rows * s.m * aa)
        stage("down", "moe", family, (padded_rows, active), tc=2 * padded_rows * s.d * s.m,
              byte_count=s.d * s.m * wa * active)
        stage("combine", "moe", family, (assignments, active), simt=None,
              byte_count=(assignments + s.n) * s.d * aa)
        if s.shared_m:
            stage("shared", "moe", family, (s.n,), tc=logical["shared_flops"], simt=None,
                  byte_count=resident["shared_weight_bytes"])
            stage("output_add", "moe", family, (s.n,), simt=s.n * s.d, byte_count=3 * s.n * s.d * aa)
    else:
        prefill = isinstance(s, KDAPrefill)
        live = sum(slot >= 0 for slot in s.state_slots) if not prefill else sum(length > 0 for length in s.chunk_lens)
        tokens = checked(sum(s.chunk_lens)) if prefill else live
        elements = checked(live * s.h_v * s.d_k * s.d_v)
        state_bytes = checked(elements * size(s.state_dtype))
        qkvo = checked(4 * tokens * s.d * (s.h_qk * s.d_k + s.h_v * s.d_v))
        logical = {"core_flops": checked(7 * tokens * s.h_v * s.d_k * s.d_v), "qkvo_flops": qkvo, "live_sequences": live}
        executed = {"qkvo_tc_flops": qkvo}
        if not prefill:
            executed["core_simt_flops"] = logical["core_flops"]
        traffic = {"logical_state_bytes": checked(2 * state_bytes)}
        resident = {"live_state_bytes": state_bytes,
                    "allocated_state_bytes": checked(s.capacity * s.h_v * s.d_k * s.d_v * size(s.state_dtype)),
                    "qkvo_weight_bytes": checked(2 * s.d * (s.h_qk * s.d_k + s.h_v * s.d_v) * size(s.weight_dtype))}
        family = (s.d, s.h_qk, s.h_v, s.d_k, s.d_v, s.state_dtype,
                  s.weight_dtype, s.activation_dtype, s.accum_dtype, s.state_layout)
        if s.state_policy != "fixed_slots":
            family += (s.state_policy,)
        if prefill:
            lengths = tuple(length for length in s.chunk_lens if length)
            shape = ("uniform", lengths[0]) if lengths and len(set(lengths)) == 1 else ("ragged", tuple(sorted(lengths)))
            family += ("prefill", s.chunk_size, s.conv_width, shape)
            executed["chunks"] = checked(sum((length + s.chunk_size - 1) // s.chunk_size for length in lengths))
            logical["tokens"] = tokens
            resident["conv_state_bytes"] = checked(s.capacity * (s.conv_width - 1)
                                                  * (2 * s.h_qk * s.d_k + s.h_v * s.d_v) * size(s.activation_dtype))
        # QKV and O need separate work: qkv = 2*N*D*(2*Hqk*dk + Hv*dv).
        qkv = checked(2 * tokens * s.d * (2 * s.h_qk * s.d_k + s.h_v * s.d_v))
        stage("qkv", "attn_module", family, (live,), tc=qkv)
        stage("gates", "attn_module", family, (live,), simt=None)
        stage("core", "attn_core", family, (live,), simt=None if prefill else logical["core_flops"],
              byte_count=traffic["logical_state_bytes"])
        stage("out_gate", "attn_module", family, (live,), simt=None)
        stage("o_proj", "attn_module", family, (live,), tc=qkvo - qkv)
    traffic["dram_bytes"] = None  # Cache-dependent: logical accesses are not DRAM counters.
    resident.update(workspace_bytes=None, graph_buffer_bytes=None)
    if set(variants) - {stage.name for stage in stages}:
        raise ValueError("variant references an unknown lowered stage")
    for ledger in (logical, executed, traffic, resident):
        for value in ledger.values():
            if value is not None:
                checked(value)
    return Lowered(tuple(stages), logical, executed, traffic, resident)


def permutation(spec: MoE) -> tuple[tuple[int, int, int], ...]:
    """Stable expert-major (expert, token, slot), used by CPU and native references."""
    return tuple(sorted((expert, token, slot) for token, row in enumerate(spec.routes)
                        for slot, expert in enumerate(row)))


def combine(spec: MoE, values: tuple[tuple[float, ...], ...], weights: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    order = permutation(spec)
    if len(values) != len(order) or len(weights) != spec.n or any(len(row) != spec.k for row in weights):
        raise ValueError("combine shape mismatch")
    width = len(values[0]) if values else spec.d
    if any(len(row) != width for row in values):
        raise ValueError("ragged expert outputs")
    result = [[0.0] * width for _ in range(spec.n)]
    for (_, token, slot), output in zip(order, values):
        for column, value in enumerate(output):
            result[token][column] += value * weights[token][slot]
    return tuple(tuple(row) for row in result)


def kda_reference(state: tuple[tuple[float, ...], ...], q: tuple[float, ...], k: tuple[float, ...],
                  v: tuple[float, ...], alpha: tuple[float, ...], beta: float) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    if not state or len(state) != len(q) or len(k) != len(q) or len(alpha) != len(q) or any(len(row) != len(v) for row in state):
        raise ValueError("KDA reference shape mismatch")
    decayed = [[value * alpha[i] for value in row] for i, row in enumerate(state)]
    residual = [beta * (v[j] - sum(row[j] * k[i] for i, row in enumerate(decayed))) for j in range(len(v))]
    updated = tuple(tuple(value + k[i] * residual[j] for j, value in enumerate(row)) for i, row in enumerate(decayed))
    output = tuple(sum(q[i] * row[j] for i, row in enumerate(updated)) for j in range(len(v)))
    return output, updated


def layer_layout(count: int, full: tuple[int, ...], recurrent: tuple[int, ...], dense_prefix: int) -> tuple[tuple[str, str], ...]:
    """Configuration metadata only; MLA, dense FFN and model output remain unmodeled."""
    checked(count)
    checked(dense_prefix)
    if dense_prefix > count or len(set(full)) != len(full) or len(set(recurrent)) != len(recurrent) or set(full) & set(recurrent) or set(full) | set(recurrent) != set(range(1, count + 1)):
        raise ValueError("hybrid attention layers must partition the model")
    return tuple(("mla_unsupported" if layer in full else "kda", "dense_unsupported" if layer <= dense_prefix else "moe") for layer in range(1, count + 1))
