"""Dynamo's native selection and load tracker, with adapter-owned deduplication."""
from dataclasses import dataclass
from importlib.metadata import version

from dynamo.llm import SelectionService
from dynamo._core import SelectionServiceError


class NativeRouter:
    def __init__(self) -> None:
        if version("ai-dynamo-runtime") != "1.5.0":
            raise ValueError("qualified Dynamo runtime is 1.5.0")
        self.service = SelectionService(indexer_threads=1)
        self.active: dict[str, bool] = {}
        self.used: set[str] = set()

    async def worker(self, identity: int) -> None:
        await self.service.upsert_worker({"worker_id": identity, "model_name": "model",
            "routing_group": "default", "endpoint": f"http://worker{identity}",
            "block_size": 16, "data_parallel_start_rank": 0, "data_parallel_size": 1})

    async def book(self, identity: str, tokens: list[int], output: int) -> int:
        if identity in self.used or not tokens or output <= 0:
            raise ValueError("reused identity or invalid request budget")
        result = await self.service.select_and_reserve({"selection_id": identity,
            "model_name": "model", "routing_group": "default",
            "token_ids": tokens,
            "expected_output_tokens": output})
        self.active[identity] = False
        self.used.add(identity)
        return result["worker_id"]

    async def prefill_complete(self, identity: str) -> bool:
        if identity not in self.active or self.active[identity]:
            return False
        await self.service.prefill_complete(identity)
        self.active[identity] = True
        return True

    async def release(self, identity: str) -> bool:
        if identity not in self.active:
            return False
        await self.service.free_reservation(identity)
        del self.active[identity]
        return True

    async def close(self) -> None:
        for identity in list(self.active):
            await self.release(identity)
        self.service.shutdown()


@dataclass
class Plan:
    identity: str
    attempt: int
    expires_ns: int
    endpoints: tuple[int, int] | None = None
    closed: bool = False


class PlanBook:
    """Explicit virtual leases above two independent native P/D selection cores."""
    def __init__(self, prefill: NativeRouter, decode: NativeRouter):
        self.prefill, self.decode = prefill, decode
        self.plans: dict[tuple[str, int], Plan] = {}
        self.events: list[tuple[str, int, str]] = []

    async def reserve(self, identity: str, attempt: int, prompt: int, output: int,
                      now_ns: int, lease_ns: int) -> Plan:
        key = (identity, attempt)
        if key in self.plans or attempt < 0 or now_ns < 0 or lease_ns <= 0:
            raise ValueError("invalid or reused plan attempt")
        plan = Plan(identity, attempt, now_ns + lease_ns)
        self.plans[key] = plan
        prefix = f"{identity}/{attempt}"
        try:
            p = await self.prefill.book(prefix + "/p", [100] * prompt, output)
            d = await self.decode.book(prefix + "/d", [100] * prompt, output)
            # Decode owns KV capacity but does not execute prefill itself.
            await self.decode.prefill_complete(prefix + "/d")
        except SelectionServiceError:
            await self.prefill.release(prefix + "/p")
            await self.decode.release(prefix + "/d")
            plan.closed = True
            self.events.append((*key, "compensated"))
            raise
        plan.endpoints = (p, d)
        self.events.append((*key, "reserved"))
        return plan

    async def prefill_complete(self, identity: str, attempt: int) -> bool:
        plan = self.plans[(identity, attempt)]
        if plan.closed:
            return False
        released = await self.prefill.release(f"{identity}/{attempt}/p")
        if released:
            self.events.append((identity, attempt, "prefill_released"))
        return released

    async def close(self, identity: str, attempt: int, reason: str) -> bool:
        plan = self.plans[(identity, attempt)]
        if plan.closed:
            return False
        await self.prefill.release(f"{identity}/{attempt}/p")
        await self.decode.release(f"{identity}/{attempt}/d")
        plan.closed = True
        self.events.append((identity, attempt, reason))
        return True

    async def expire(self, now_ns: int) -> None:
        for plan in self.plans.values():
            if not plan.closed and plan.expires_ns <= now_ns:
                await self.close(plan.identity, plan.attempt, "expired")
