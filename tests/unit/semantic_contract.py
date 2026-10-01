"""CPU algebra, binding, scheduling and coverage; synthetic timings only."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.cost import AnalyticCost, CalibrationStore, ConstantCost, Cost, Identity, Profile
from sim.models import combine, kda_reference, layer_layout, lower, permutation
from sim.replay import counterfactual, replay
from sim.work import Execution, Graph, KDA, LIMIT, MoE, Registry, TopK, Unknown, Work, routes


def moe() -> MoE:
    return MoE(2, 4, 4, 2, 8, ((0, 1), (1, 2)), tile_m=4)


def kda(b: int = 1) -> KDA:
    return KDA(b, 2304, 32, 32, 128, 128, tuple(range(b)), (2048,) * b)


def work(spec, op: str = "op", sequence: int = 0, stream: int = 0, deps: tuple[str, ...] = ()) -> Work:
    return Work(op, spec, "contract_backend", "revision-1",
                Execution(node=op, sequence=sequence, stream=stream), deps)


class SemanticContract(unittest.TestCase):
    def test_schema_roundtrip_and_units(self):
        for spec in (moe(), kda(), TopK(2, 256, 8), Unknown("opaque", (1, 1, 1), (32, 1, 1))):
            original = work(spec)
            self.assertEqual(Work.loads(original.dumps()), original)
        for spec in (replace(moe(), n=-1), replace(moe(), k=5), replace(moe(), d=0), replace(moe(), weight_dtype="bytes")):
            with self.assertRaises(ValueError):
                lower(work(spec))
        with self.assertRaises(ValueError):
            lower(work(TopK(LIMIT, 2, 1)))
        with self.assertRaises(ValueError):
            replace(work(moe()), schema=2).validate()

    def test_moe_golden_and_hot_counterexample(self):
        cold = lower(work(moe()))
        self.assertEqual(moe().counts, (1, 2, 1, 0))
        self.assertEqual(cold.logical["router_flops"], 64)
        self.assertEqual(cold.logical["expert_flops"], 768)
        self.assertEqual(cold.traffic["cold_expert_weight_bytes"], 576)
        self.assertEqual(cold.residency["expert_weight_bytes"], 768)
        self.assertEqual(cold.executed, {"expert_flops": 2304, "padded_rows": 12})
        hot = lower(work(replace(moe(), routes=((0, 1), (0, 1)), routing="hot")))
        self.assertEqual(hot.logical["expert_flops"], cold.logical["expert_flops"])
        self.assertEqual(hot.traffic["cold_expert_weight_bytes"], 384)
        self.assertEqual(hot.residency, cold.residency)

    def test_routing_and_combine_reference(self):
        spec = moe()
        self.assertEqual(permutation(spec), ((0, 0, 0), (1, 0, 1), (1, 1, 0), (2, 1, 1)))
        self.assertEqual(combine(spec, ((2.,), (4.,), (8.,), (12.,)), ((.25, .75), (.5, .5))), ((3.5,), (10.,)))
        for hot in (None, 2):
            trace = routes(128, 4, 2, 42, hot)
            self.assertEqual(trace, routes(128, 4, 2, 42, hot))
            self.assertEqual(sum(replace(spec, n=128, routes=trace).counts), 256)
        empty = lower(work(replace(spec, n=0, routes=())))
        self.assertEqual(empty.logical["expert_flops"], 0)
        self.assertEqual(empty.executed["padded_rows"], 0)
        for bad in (((0, 0), (1, 2)), ((0, 1), (1, 4))):
            with self.assertRaises(ValueError):
                lower(work(replace(spec, routes=bad)))

    def test_selector_semantics(self):
        for k in (1, 8, 257):
            spec = TopK(4, 257, k, tie_break="backend_unspecified")
            self.assertEqual(lower(work(spec)).logical["selected"], 4 * k)
        with self.assertRaises(ValueError):
            lower(work(TopK(4, 256, 257)))
        self.assertIsNone(AnalyticCost(1e12, 1e11, 1e10, 10).estimate(work(TopK(4, 256, 8))).costs[0].standalone_ns)

    def test_kda_algebra_and_precision(self):
        one, batch = lower(work(kda())), lower(work(kda(64)))
        self.assertEqual(one.residency["live_state_bytes"], 2 * 1024**2)
        self.assertEqual(batch.residency["live_state_bytes"], 128 * 1024**2)
        self.assertEqual(batch.traffic["logical_state_bytes"], 256 * 1024**2)
        self.assertEqual(batch.logical["core_flops"], 234881024)
        self.assertEqual(batch.logical["qkvo_flops"], 4831838208)
        self.assertEqual(lower(work(replace(kda(), seq_lens=(1,)))), one)
        self.assertEqual(lower(work(replace(kda(), state_dtype="bf16"))).residency["live_state_bytes"], 1024**2)
        output, state = kda_reference(((1., 2.), (3., 4.)), (1., 0.), (1., 0.), (5., 6.), (.5, .5), .5)
        self.assertEqual(state, ((2.75, 3.5), (1.5, 2.)))
        self.assertEqual(output, (2.75, 3.5))
        with self.assertRaises(ValueError):
            lower(work(replace(kda(), h_qk=16)))
        padded = lower(work(replace(kda(2), state_slots=(0, -1))))
        self.assertEqual(padded.logical["core_flops"], one.logical["core_flops"])

    def test_graph_fresh_routes_and_slots(self):
        captured = replace(kda(), state_capacity=4, state_policy="rotating_slots")
        templates = (work(moe(), "moe"), work(captured, "kda", 1))
        graph = Graph("g", templates, 4, 4)
        first = graph.bind("r1", {"moe": moe(), "kda": captured})
        second = graph.bind("r2", {"moe": replace(moe(), routes=((0, 1), (0, 1))), "kda": replace(captured, state_slots=(3,))})
        self.assertNotEqual(lower(first.works[0]).logical, lower(second.works[0]).logical)
        self.assertEqual(first.works[1].spec.state_slots, (0,))
        self.assertEqual(second.works[1].spec.state_slots, (3,))
        del graph  # Bound immutable descriptors own their data.
        self.assertEqual(first.works[0].spec.counts, (1, 2, 1, 0))
        for updates in ({"moe": moe()}, {"moe": replace(moe(), d=8), "kda": kda()}, {"moe": moe(), "kda": replace(kda(), state_slots=(4,))}):
            with self.assertRaises(ValueError):
                Graph("g", templates, 4, 4).bind("bad", updates)

    def test_dependencies_cycles_and_reproducibility(self):
        ops = (work(TopK(1, 4, 2), "a", stream=1), work(TopK(1, 4, 2), "b", stream=2, deps=("a",)))
        registry = Registry(ops)
        report = replay(registry, ConstantCost(30), {"b": 100})
        self.assertEqual([(i.start_ns, i.end_ns) for i in report.timeline], [(0, 30), (100, 130)])
        self.assertEqual(report.host_gap_ns, 70)
        self.assertEqual(len({replay(registry, ConstantCost(30), {"b": 100}).digest() for _ in range(100)}), 1)
        with self.assertRaises(ValueError):
            replay(Registry((replace(ops[0], deps=("b",)), ops[1])), ConstantCost(30))
        with self.assertRaises(ValueError):
            Registry((replace(ops[0], deps=("missing",)),))
        with self.assertRaises(ValueError):
            Registry((ops[0], replace(ops[0], op_id="duplicate")))

    def test_fusion_one_physical_charge(self):
        ops = tuple(replace(work(TopK(1, 4, 2), str(i), i), fusion_group="physical") for i in range(2))
        report = replay(Registry(ops), ConstantCost(30), fused={"physical": Cost(50, "synthetic", "fusion_fixture")})
        self.assertEqual(report.makespan_ns, 50)
        self.assertEqual(report.accounting()["fused_shared_ns"], 50)
        self.assertEqual(len(report.timeline), 1)
        self.assertEqual(len(report.estimates), 2)
        with self.assertRaises(ValueError):
            replay(Registry(ops), ConstantCost(30))

    def test_profiles_modes_and_coverage(self):
        spec = TopK(4, 256, 8)
        op = work(spec)
        identity = Identity("GPU-contract", "driver", op.backend, op.revision, "graph", "measured-steady")
        store = CalibrationStore(identity, ())
        with self.assertRaises(ValueError):
            replay(Registry((op,)), store)
        self.assertIsNone(replay(Registry((op,)), store, strict=False).makespan_ns)
        compat = replay(Registry((op,)), CalibrationStore(identity, (), mode="compat"))
        self.assertEqual((compat.unknown_count, compat.status), (1, "compat"))
        stage = lower(op).stages[0]
        profile = Profile("p", identity, stage.name, stage.family, (((1,), 20), ((8,), 90)), (10., 10.), (1,), (8,), True)
        calibrated = replay(Registry((op,)), CalibrationStore(identity, (profile,)))
        self.assertEqual((calibrated.makespan_ns, calibrated.unknown_count), (50, 0))
        with self.assertRaises(ValueError):
            CalibrationStore(replace(identity, hardware="another_gpu"), (profile,))
        with self.assertRaises(ValueError):
            replace(profile, schema=2)
        with self.assertRaises(ValueError):
            replace(profile, coefficients=(-1., 10.))
        with self.assertRaises(ValueError):
            replay(Registry((work(replace(spec, n=16)),)), CalibrationStore(identity, (profile,)))

    def test_compiled_variant_separates_interpolation(self):
        op = replace(work(moe()), variants=(("route", "compiled-path-a"),))
        other = replace(op, variants=(("route", "compiled-path-b"),))
        left = next(stage for stage in lower(op).stages if stage.name == "route")
        right = next(stage for stage in lower(other).stages if stage.name == "route")
        self.assertNotEqual(left.family, right.family)
        identity = Identity("GPU-contract", "driver", op.backend, op.revision, "graph", "measured-steady")
        profile = Profile("route-a", identity, left.name, left.family, ((left.features, 30),))
        costs = CalibrationStore(identity, (profile,)).estimate(other).costs
        self.assertIsNone(costs[3].standalone_ns)
        with self.assertRaises(ValueError):
            Cost(-1, "synthetic", "invalid")

    def test_amdahl_serial_only(self):
        ops = (work(TopK(1, 4, 2), "other"), work(TopK(1, 4, 2), "attn", 1))
        report = replay(Registry(ops), ConstantCost(30), {"attn": 70})
        report = replace(report, timeline=(report.timeline[0], replace(report.timeline[1], category="attn_core")))
        self.assertAlmostEqual(counterfactual(report, "attn_core", 2), 1.1764705882352942)
        self.assertEqual(counterfactual(report, "other", 2), 1.)  # Saved work is absorbed by host release.

    def test_hybrid_layout_is_partial(self):
        full = (4, 8, 12, 16, 20, 24, 27)
        recurrent = tuple(i for i in range(1, 28) if i not in full)
        layout = layer_layout(27, full, recurrent, 1)
        self.assertEqual(sum(attn == "kda" for attn, _ in layout), 20)
        self.assertEqual(sum(attn == "mla_unsupported" for attn, _ in layout), 7)
        self.assertEqual(layout[0], ("kda", "dense_unsupported"))
        with self.assertRaises(ValueError):
            layer_layout(27, full, recurrent + (4,), 1)


if __name__ == "__main__":
    unittest.main()
