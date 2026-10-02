"""T13: request/token conservation, physical admission and cancellation lifetime."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.cost import Cost
from sim.serving import Batch, DeviceMemory, PageLayout, Policy, Request, SCOPE, percentiles, run


class FixtureCost:
    scope = SCOPE
    def estimate(self, batch: Batch) -> Cost:
        ns = 10 + sum(item.tokens for item in batch.items) if batch.phase == "prefill" else 7
        return Cost(ns, "synthetic", "request_contract")


DEVICE = DeviceMemory("node/card", 100, 60, (PageLayout(4, 10, "explicit_latent_layout"),),
                      state_slots=2, state_bytes_per_slot=2, conv_bytes_per_slot=1,
                      workspace_bytes=8, graph_bytes=2, communication_bytes=2, allocator_bytes=2)


class ServingContract(unittest.TestCase):
    def test_chunks_decode_priority_and_arrival_during_execution(self):
        requests = (Request("a", 0, 5, 2), Request("b", 2, 3, 1))
        report = run(requests, (replace(DEVICE, capacity_bytes=200),), Policy(2, 3), FixtureCost())
        self.assertEqual([step.batch.phase for step in report.batches], ["prefill", "prefill", "decode", "prefill"])
        self.assertEqual([out.prompt_processed for out in report.outcomes], [5, 3])
        self.assertEqual([out.tokens_ns for out in report.outcomes], [[26, 33], [45]])
        self.assertEqual(report.outcomes[1].admitted_ns, 2)
        self.assertEqual(sum(item.tokens for step in report.batches if step.batch.phase == "prefill" for item in step.batch.items), 8)
        self.assertEqual(report.metrics()["emitted_tokens"], 3)
        self.assertEqual(report.metrics()["itl"], {"samples": 1, "p50_ns": 7, "p95_ns": None, "p99_ns": None})

    def test_each_card_capacity_and_fcfs_head_of_line(self):
        requests = (Request("a", 0, 4, 1), Request("b", 0, 8, 1), Request("c", 0, 4, 1), Request("oversize", 0, 12, 1))
        cards = (replace(DEVICE, device="node/card0", capacity_bytes=200), replace(DEVICE, device="node/card1"))
        report = run(requests, cards, Policy(4, 32), FixtureCost())
        self.assertEqual([out.admitted_ns for out in report.outcomes], [0, 14, 32, None])
        self.assertEqual([out.state for out in report.outcomes], ["completed"] * 3 + ["rejected"])
        self.assertEqual(report.peak_bytes, {"node/card0": 100, "node/card1": 100})
        self.assertTrue(all(80 <= event.used_bytes <= 100 for event in report.memory))
        self.assertEqual(report.metrics()["itl"]["p50_ns"], None)
        self.assertEqual(report.metrics()["completed"] + report.metrics()["rejected"], len(requests))

    def test_cancelled_inflight_request_retains_pages_and_slot(self):
        card = replace(DEVICE, state_slots=1, capacity_bytes=200)
        for cancel_ns in (6, 14):
            report = run((Request("a", 0, 4, 4, cancel_ns), Request("b", 7, 4, 1)),
                         (card,), Policy(2, 32), FixtureCost())
            a, b = report.outcomes
            self.assertEqual((a.state, a.terminal_ns, a.tokens_ns), ("cancelled", 14, []))
            self.assertEqual((a.prompt_processed, a.discarded_tokens), (4, 1))
            self.assertEqual(b.admitted_ns, 14)
            releases = [event.time_ns for event in report.memory if event.request_id == "a" and event.action == "release"]
            self.assertEqual(releases, [14])
            reservations = [event for event in report.memory if event.action == "reserve"]
            self.assertEqual([event.state_slot for event in reservations], [0, 0])

    def test_preallocated_kv_pool_is_not_charged_again_per_request(self):
        card = replace(DEVICE, capacity_bytes=100, kv_pool_bytes=20)
        report = run((Request("a", 0, 4, 1), Request("b", 0, 8, 1)), (card,), Policy(2, 32), FixtureCost())
        self.assertEqual(report.peak_bytes, {card.device: 100})
        self.assertTrue(all(event.used_bytes == 100 for event in report.memory))
        self.assertEqual([out.admitted_ns for out in report.outcomes], [0, 14])
        self.assertEqual(report.memory[-1].reserved_kv_bytes, 0)

    def test_waiting_cancel_and_late_cancel_do_not_reserve_or_extend_drain(self):
        report = run((Request("a", 0, 4, 1, 100), Request("b", 0, 4, 1, 0)),
                     (DEVICE,), Policy(1, 32), FixtureCost())
        self.assertEqual(report.finish_ns, 14)
        self.assertEqual(report.outcomes[1].state, "cancelled")
        self.assertFalse(any(event.request_id == "b" for event in report.memory))

    def test_unsupported_prefill_is_not_decode_times_tokens(self):
        class DecodeOnly(FixtureCost):
            def estimate(self, batch: Batch) -> Cost:
                return Cost(None, "unsupported", "", "no independent prefill profile") if batch.phase == "prefill" else super().estimate(batch)
        with self.assertRaisesRegex(ValueError, "no independent prefill profile"):
            run((Request("a", 0, 4, 2),), (DEVICE,), Policy(1, 32), DecodeOnly())
        with self.assertRaises(ValueError):
            Request("a", 5, 4, 2, 4)
        with self.assertRaises(ValueError):
            replace(DEVICE, weight_bytes=200)

    def test_percentile_population_and_repeatable_conservation(self):
        self.assertIsNone(percentiles(list(range(99)))["p99_ns"])
        self.assertEqual(percentiles(list(range(100)))["p99_ns"], 98.01)
        requests = tuple(Request(str(i), i * 3, 1 + i % 8, 1 + i % 4, i * 3 + 5 if i % 5 == 0 else None) for i in range(30))
        reference = run(requests, (DEVICE,), Policy(2, 5), FixtureCost())
        self.assertEqual(sum(out.state in ("completed", "cancelled", "rejected") for out in reference.outcomes), len(requests))
        for out in reference.outcomes:
            self.assertEqual(len(out.tokens_ns), out.request.output_len if out.state == "completed" else len(out.tokens_ns))
            self.assertTrue(all(t >= out.request.arrival_ns for t in out.tokens_ns))
        for _ in range(100):
            self.assertEqual(asdict(run(requests, (DEVICE,), Policy(2, 5), FixtureCost())), asdict(reference))


if __name__ == "__main__":
    unittest.main()
