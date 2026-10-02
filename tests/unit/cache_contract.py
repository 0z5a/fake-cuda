from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.common.cache import BlockIdentity, Block, CacheLedger, CacheUpdate, CacheVisibility


class Contracts(unittest.TestCase):
    def test_physical_references_and_graph_budget(self) -> None:
        ledger = CacheLedger(1000, 200)
        identity = BlockIdentity("m", "t", "a", "prefix", "tp2-rank0-fp16")
        ledger.allocate("physical0", Block(identity, "endpoint0", 0, 100, 10))
        with self.assertRaisesRegex(RuntimeError, "transfer"):
            ledger.acquire("physical0", "request0", 9)
        ledger.acquire("physical0", "request0", 10)
        ledger.acquire("physical0", "request1", 10)
        self.assertEqual(ledger.used_bytes, 300)
        ledger.reserve_graph("pool", 600)
        with self.assertRaises(MemoryError):
            ledger.allocate("physical1", Block(identity, "endpoint0", 1, 101, 10))
        ledger.replay_graph("pool", 1)
        with self.assertRaisesRegex(RuntimeError, "executing"):
            ledger.free_graph("pool")
        ledger.complete_graph("pool", 1)
        ledger.free_graph("pool")
        with self.assertRaisesRegex(RuntimeError, "references"):
            ledger.evict("physical0")
        ledger.release("physical0", "request0")
        ledger.release("physical0", "request1")
        ledger.evict("physical0")
        self.assertEqual(ledger.used_bytes, 200)

    def test_delayed_visibility_and_generation(self) -> None:
        visibility = CacheVisibility()
        identity = BlockIdentity("m", "t", "a", "prefix", "tp2-rank0-fp16")
        update = CacheUpdate("endpoint", 0, 2, 20, frozenset((identity,)))
        with self.assertRaisesRegex(RuntimeError, "not yet visible"):
            visibility.apply(update, 19)
        self.assertTrue(visibility.apply(update, 20))
        self.assertFalse(visibility.apply(CacheUpdate("endpoint", 0, 1, 10, frozenset()), 21))
        self.assertTrue(visibility.apply(CacheUpdate("endpoint", 1, 0, 22, frozenset()), 22))
        self.assertFalse(visibility.apply(update, 23))


if __name__ == "__main__":
    unittest.main()
