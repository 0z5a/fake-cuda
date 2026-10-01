"""One original Scheduler and an explicit fixed-length control oracle."""
import statistics

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from adapters.common.control import ControlLedger, ControlValue, DataKind
from adapters.vllm.costs import Shape


def make_scheduler(configuration: tuple[VllmConfig, KVCacheConfig, int, int], *, allow_chunked: bool = False) -> Scheduler:
    config, kv_config, block_size, hash_size = configuration
    if config.scheduler_config.async_scheduling or config.cache_config.enable_prefix_caching or config.speculative_config is not None or (config.scheduler_config.enable_chunked_prefill and not allow_chunked):
        raise ValueError("unsupported scheduler configuration")
    config.cache_config.num_gpu_blocks = kv_config.num_blocks
    config.cache_config.block_size = block_size
    return Scheduler(config, kv_config, StructuredOutputManager(config), block_size, hash_size)


def prompt_tokens(identity: str, prompt: int) -> list[int]:
    return [100 + int(identity) % 8] * prompt


def add_request(scheduler: Scheduler, identity: str, prompt: int, output: int, arrival: int) -> None:
    scheduler.add_request(Request(identity, prompt_tokens(identity, prompt),
        SamplingParams(temperature=0, ignore_eos=True, max_tokens=output, detokenize=False),
        None, arrival_time=arrival / 1e9))


def scheduled_shape(scheduler: Scheduler, out: SchedulerOutput) -> Shape:
    prefill = [scheduler.requests[rid].num_computed_tokens - count < scheduler.requests[rid].num_prompt_tokens
               for rid, count in out.num_scheduled_tokens.items()]
    if not prefill or any(prefill) != all(prefill):
        raise ValueError("empty or mixed steps are outside calibration")
    contexts = [scheduler.requests[rid].num_computed_tokens for rid in out.num_scheduled_tokens]
    if len(set(contexts)) != 1:
        raise ValueError("heterogeneous contexts are outside homogeneous-step calibration")
    return Shape("prefill" if all(prefill) else "decode", len(prefill), statistics.mean(contexts))


def complete_step(scheduler: Scheduler, out: SchedulerOutput, ledger: ControlLedger,
                  step_identity: str) -> list[tuple[str, list[int], bool]]:
    ids = list(out.num_scheduled_tokens)
    sampled = []
    for rid in ids:
        request = scheduler.requests[rid]
        identity = f"{step_identity}/{rid}"
        value = (100,) if request.num_computed_tokens >= request.num_tokens else ()
        ledger.produce(identity, ControlValue("fixed_length_token_oracle", DataKind.HOST_VALID, value))
        sampled.append(list(ledger.read(identity, "vllm.Scheduler.update_from_output")))
    runner = ModelRunnerOutput(ids, {rid: i for i, rid in enumerate(ids)}, sampled)
    result = scheduler.update_from_output(out, runner)
    return [(o.request_id, o.new_token_ids, o.finished) for batch in result.values() for o in batch.outputs]
