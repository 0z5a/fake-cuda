"""Safe horizons, delayed delivery, collective generations and control provenance."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapters.common.clock import Coordinator
from adapters.common.collectives import Collective, CollectiveMatcher
from adapters.common.control import ControlLedger, ControlValue, DataKind


class Contracts(unittest.TestCase):
    def test_safe_horizon_and_arrival(self) -> None:
        c = Coordinator(("request", "device", "scheduler"))
        c.send("request", "scheduler", "arrival", "req", 5, 5)
        c.send("device", "scheduler", "token", "req", 10, 12)
        c.certify("request", 20)
        c.certify("device", 20)
        with self.assertRaisesRegex(RuntimeError, "dependency_blocked"):
            c.advance()
        c.certify("scheduler", 20)
        arrival = c.advance()[0]
        self.assertEqual(c.now, 5)
        self.assertEqual(arrival.observed_ns, 5)
        with self.assertRaisesRegex(RuntimeError, "dependency_blocked"):
            c.advance()
        c.certify("scheduler", 20)
        token = c.advance()[0]
        self.assertEqual((token.created_ns, token.deliver_ns, token.observed_ns), (10, 12, 12))

    def test_unregistered_transport_and_failure(self) -> None:
        c = Coordinator(("sender", "receiver"))
        identity = c.begin_send("sender")
        c.certify("sender", 100)
        c.certify("receiver", 100)
        with self.assertRaisesRegex(RuntimeError, "dependency_blocked"):
            c.advance()
        c.register(identity, "receiver", "message", "data", 1, 2)
        self.assertEqual(c.advance()[0].deliver_ns, 2)
        c.failed = True
        with self.assertRaisesRegex(RuntimeError, "protocol_failure"):
            c.advance()

    def test_collective_identity(self) -> None:
        match = CollectiveMatcher()
        first = Collective("comm", 0, 1, "all_reduce", "fp16", 1024, (0, 1), 1)
        self.assertIsNone(match.join(first, 0, 10))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            match.join(replace(first, count=2048), 1, 30)
        self.assertEqual(match.join(first, 1, 30), 30)
        second = replace(first, graph_replay=2)
        self.assertIsNone(match.join(second, 1, 40))
        self.assertEqual(match.join(second, 0, 50), 50)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            match.join(second, 0, 50)

    def test_control_producer_chain(self) -> None:
        ledger = ControlLedger()
        ledger.produce("compute", ControlValue("kernel", DataKind.COMPUTE))
        ledger.produce("token", ControlValue("D2H", DataKind.UNKNOWN, parents=("compute",)))
        with self.assertRaisesRegex(ValueError, "token <- D2H.*compute <- kernel"):
            ledger.read("token", "sampler")
        ledger.produce("fixed", ControlValue("fixed_length_token_oracle", DataKind.HOST_VALID, (100,)))
        self.assertEqual(ledger.read("fixed", "scheduler"), (100,))


if __name__ == "__main__":
    unittest.main()
