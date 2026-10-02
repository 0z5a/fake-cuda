"""Exercise native Dynamo snapshot selection and P/D reservation compensation."""
import asyncio
import os
from pathlib import Path
import sys

os.environ["DYN_ROUTER_USE_KV_EVENTS"] = "false"
os.environ["DYN_ROUTER_ASSUME_KV_REUSE"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adapters.dynamo.native import NativeRouter, PlanBook, SelectionServiceError


def counts(router: NativeRouter) -> tuple[int, int, int]:
    rows = [row for partition in router.service.loads() for row in partition["loads"]]
    return tuple(sum(row[key] for row in rows) for key in
                 ("active_requests", "potential_prefill_tokens", "potential_decode_blocks"))


async def main() -> None:
    prefill, decode = NativeRouter(), NativeRouter()
    plans = PlanBook(prefill, decode)
    try:
        await prefill.worker(1)
        # The second native selector has no eligible worker: real admission
        # fails after the first side booked, and compensation clears its load.
        try:
            await plans.reserve("0", 0, 64, 8, 0, 100)
        except SelectionServiceError:
            pass
        else:
            raise AssertionError("native decode selector should reject an empty pool")
        assert plans.plans[("0", 0)].closed and counts(prefill) == (0, 0, 0)
        await decode.worker(2)
        plan = await plans.reserve("0", 1, 64, 8, 0, 100)
        assert plan.endpoints == (1, 2)
        assert counts(prefill) == (1, 64, 4) and counts(decode) == (1, 0, 4)
        assert await plans.prefill_complete("0", 1)
        assert not await plans.prefill_complete("0", 1)
        assert counts(prefill) == (0, 0, 0) and counts(decode) == (1, 0, 4)
        assert await plans.close("0", 1, "completed")
        assert not await plans.close("0", 1, "late_cancel")
        await plans.reserve("1", 0, 64, 8, 10, 20)
        assert await plans.close("1", 0, "cancelled")
        assert not await plans.prefill_complete("1", 0)
        await plans.reserve("2", 0, 64, 8, 20, 20)
        await plans.expire(39)
        assert not plans.plans[("2", 0)].closed
        await plans.expire(40)
        assert plans.plans[("2", 0)].closed
        assert not await plans.close("2", 0, "late_completed")
        assert counts(prefill) == counts(decode) == (0, 0, 0)
        # Snapshot and shadow replay use the same real selector; neither books.
        request = {"model_name": "model", "routing_group": "default", "token_ids": [100] * 64}
        snapshots = [await prefill.service.select(request) for _ in range(3)]
        assert all(row["worker_id"] == 1 and row["overlap"]["gpu"] == 0 for row in snapshots)
        assert counts(prefill) == (0, 0, 0)
        print("PASS native Dynamo 1.5.0 R03/R04: partial failure compensation, independent P/D release, cancel, virtual lease, late/duplicate callbacks; all native loads zero")
        print("PASS snapshot/shadow: native eligibility and selection, no bookings; KV-event and prefix-reuse policies explicitly disabled")
    finally:
        await prefill.close()
        await decode.close()


if __name__ == "__main__":
    asyncio.run(main())
