"""T14/T15: exact EP payloads, link/SM sharing and final-use buffer release."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.ep import Chunk, Device, Link, Payload, StageCost, TokenRoute, Topology, ledger, run
from sim.cost import Identity, Profile
from sim.resources import Capacity, Engine, Phase, Work, profile_phase

BRIDGE = Path(sys.argv.pop(1))
PROTOCOL = "explicit_contract_protocol"
PAYLOAD = Payload("bf16", "unique_tokens")
TOPOLOGY = Topology((Device("node/card0", 100, 64), Device("node/card1", 100, 64)),
                    (Link("shared-bus", 1, duplex="shared"),), ((0, 1, (0,)), (1, 0, (0,))), "synthetic_half_duplex")


def chunks() -> tuple[Chunk, ...]:
    transfer = StageCost(100, (100, 100), (0, 0), "synthetic_transfer")
    return tuple(Chunk(i + 1, (TokenRoute(i, 0, (1 - i,)),), transfer,
                       StageCost(40, (0, 0), (40, 0) if i else (0, 40), "synthetic_compute"), transfer, (16, 16)) for i in range(2))


class EpContract(unittest.TestCase):
    def test_discrete_combination_charged_once_without_bandwidth_division(self):
        identity = Identity("synthetic_contract", "none", "combined_event_fixture", "1", "graph", "measured-steady")
        profile = Profile("synthetic_pair", identity, "pair", ("fixture",), (((1,), 100), ((3,), 200)),
                          (50, 50), (1,), (3,), qualified=True)
        engine = Engine(BRIDGE, (Capacity("combination_lane", resident=1),))
        try:
            for work_id, feature in ((1, 1), (2, 2)):
                engine.submit(Work(work_id, 0, (profile_phase(profile, (feature,), identity, (1,)),)))
            engine.advance(100)
            self.assertEqual(engine.times(1), (0, 100))
            self.assertEqual(engine.times(2), (100, -1))
            engine.advance(250)
            self.assertEqual(engine.times(2), (100, 250))
        finally:
            engine.close()
        for source, features, expected in ((replace(profile, qualified=False), (1,), identity),
                                           (profile, (4,), identity),
                                           (profile, (1,), replace(identity, revision="2"))):
            with self.assertRaises(ValueError):
                profile_phase(source, features, expected, (1,))

    def test_assignments_unique_tokens_independent_combine_and_local_bytes(self):
        routes = (TokenRoute(0, 0, (1, 2)), TokenRoute(0, 1, (0,)))
        result = ledger(routes, (0, 1, 1), 2, 4,
                        Payload("bf16", "unique_tokens", assignment_metadata_bytes=4), Payload("fp32", "assignments"), PROTOCOL)
        self.assertEqual(result.assignments, ((1, 2), (0, 0)))
        self.assertEqual(result.unique_tokens, ((1, 1), (0, 0)))
        self.assertEqual(result.dispatch_bytes, ((12, 16), (0, 0)))
        self.assertEqual(result.combine_bytes, ((16, 0), (32, 0)))
        self.assertEqual(TOPOLOGY.traffic(result.dispatch_bytes)[0], (16,))
        self.assertEqual(TOPOLOGY.lower_bound_ns(result.combine_bytes), 32)
        reduced = ledger(routes, (0, 1, 1), 2, 4, PAYLOAD, Payload("fp32", "unique_tokens"), PROTOCOL)
        self.assertEqual(reduced.combine_bytes[1][0], 16)
        for bad in ((TokenRoute(0, 0, (1, 1)),), (TokenRoute(0, 0, (1,)),) * 2):
            with self.assertRaises(ValueError):
                ledger(bad, (0, 1, 1), 2, 4, PAYLOAD, PAYLOAD, PROTOCOL)

    def test_saturated_links_and_sm_never_grant_each_operation_full_capacity(self):
        serial = run(chunks(), TOPOLOGY, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE)
        pipeline = run(chunks(), TOPOLOGY, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE, pipeline=True)
        self.assertEqual((serial.finish_ns, pipeline.finish_ns), (480, 440))
        self.assertEqual(pipeline.scope, "analytic_topology_scenario")
        self.assertTrue(all(s.throughput[-1] <= 1 for s in pipeline.snapshots))
        self.assertTrue(any(s.throughput[-1] == 1 for s in pipeline.snapshots))
        full = replace(TOPOLOGY, links=(Link("forward", 1), Link("reverse", 1)),
                       paths=((0, 1, (0,)), (1, 0, (1,))))
        duplex = run(chunks(), full, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE, pipeline=True)
        self.assertEqual(duplex.finish_ns, 240)

    def test_buffers_hold_until_combine_and_limit_pipeline_admission(self):
        limited = replace(TOPOLOGY, devices=tuple(replace(d, workspace_bytes=16) for d in TOPOLOGY.devices))
        result = run(chunks(), limited, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE, pipeline=True)
        self.assertEqual(result.times, ((1, 0, 240), (2, 240, 480)))
        self.assertTrue(all(s.resident[2] <= 16 and s.resident[5] <= 16 for s in result.snapshots))
        self.assertTrue(all(s.resident[2] == 16 for s in result.snapshots if s.time_ns < 480))

    def test_slow_rank_barrier_startup_and_counterfactual_reschedule(self):
        slow = (replace(chunks()[0], compute=replace(chunks()[0].compute, isolated_ns=100)),)
        delayed = run(slow, TOPOLOGY, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE)
        self.assertEqual(delayed.finish_ns, 300)
        startup = replace(TOPOLOGY, links=(replace(TOPOLOGY.links[0], startup_ns=5),))
        self.assertEqual(run(slow, startup, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE).finish_ns, 310)
        faster = tuple(replace(c, compute=replace(c.compute, isolated_ns=20, sm_ns=tuple(n // 2 for n in c.compute.sm_ns))) for c in chunks())
        result = run(faster, TOPOLOGY, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE, pipeline=True)
        self.assertEqual(result.finish_ns, 420)

    def test_resource_changes_recalculate_remaining_work_and_generation(self):
        engine = Engine(BRIDGE, (Capacity("dram", 1),))
        try:
            engine.submit(Work(1, 0, (Phase(100, (100,), (0,)),)))
            initial = engine.progress(1)
            engine.advance(20)
            engine.submit(Work(2, 20, (Phase(40, (40,), (0,)),), weight=2.5))
            changed = engine.progress(1)
            self.assertGreater(changed.generation, initial.generation)
            self.assertNotEqual(changed.deadline_ns, initial.deadline_ns)
            self.assertAlmostEqual(changed.remaining, .8)
            self.assertAlmostEqual(changed.rate * 100 + engine.progress(2).rate * 40, 1)
            engine.advance(100)
            self.assertEqual(engine.times(2), (20, 100))
            self.assertEqual(engine.times(1)[1], -1)
            engine.advance(140)
            self.assertEqual(engine.times(1), (0, 140))
        finally:
            engine.close()

    def test_invalid_placement_path_buffer_and_repeatability(self):
        with self.assertRaises(ValueError):
            ledger((TokenRoute(0, 0, (2,)),), (0, 1), 2, 50, PAYLOAD, PAYLOAD, PROTOCOL)
        with self.assertRaises(ValueError):
            replace(TOPOLOGY, paths=()).traffic(((0, 1), (0, 0)))
        too_big = (replace(chunks()[0], buffers=(65, 65)),)
        with self.assertRaises(ValueError):
            run(too_big, TOPOLOGY, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE)
        reference = run(chunks(), TOPOLOGY, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE, pipeline=True)
        for _ in range(10):
            self.assertEqual(run(chunks(), TOPOLOGY, (0, 1), 50, PAYLOAD, PAYLOAD, PROTOCOL, BRIDGE, pipeline=True), reference)


if __name__ == "__main__":
    unittest.main()
