"""Native llm-d SchedulerProfile embedded in the shared endpoint loop."""
from pathlib import Path

from adapters.llmd_epp.client import NativeEpp
from adapters.vllm.costs import StepCost
from adapters.vllm.routing import Decision, Route, run as endpoint_loop
from adapters.vllm.runner import Workload, ServingResult
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig


class EppPolicy(NativeEpp):
    def choose(self, request: str, tokens: list[int], output: int, queues: list[int]) -> Decision:
        endpoint, scores = self.select(queues)
        return Decision(endpoint, scores)

    def feedback(self, prefills: list[str], finished: list[str]) -> None:
        # This queue-only profile consumes delivered queue snapshots.
        pass


def run(workload: Workload, configurations: list[tuple[VllmConfig, KVCacheConfig, int, int]],
        model: StepCost, scales: tuple[float, ...], metrics_delay_ns: int,
        bridge_path: Path, epp_path: Path) -> tuple[ServingResult, list[Route]]:
    return endpoint_loop(workload, configurations, model, scales, metrics_delay_ns,
                         bridge_path, EppPolicy(epp_path))
