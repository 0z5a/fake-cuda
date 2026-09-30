from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    model_config_sha256: str
    hardware_profile_sha256: str
    backend: str
    backend_version: str
    dtype: str
    attention_backend: str
    graph_mode: str
    tp: int
    pp: int
    attention_dp: int
    cp: int
    moe_tp: int
    moe_ep: int
    kv_block_size: int
    scope: str
    aisimulate_version: str = "0.12.0"
