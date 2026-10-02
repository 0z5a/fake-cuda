"""Host control reads require a declared producer, never a buffer-size guess."""
from dataclasses import dataclass
from enum import Enum


class DataKind(Enum):
    HOST_VALID = "host-backed/valid"
    COMPUTE = "virtual-compute"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ControlValue:
    producer: str
    kind: DataKind
    values: tuple[int, ...] = ()
    parents: tuple[str, ...] = ()


class ControlLedger:
    def __init__(self) -> None:
        self.values: dict[str, ControlValue] = {}
        self.reads: list[tuple[str, str]] = []

    def produce(self, identity: str, value: ControlValue) -> None:
        if identity in self.values or not value.producer or any(parent not in self.values for parent in value.parents):
            raise ValueError("invalid control producer")
        self.values[identity] = value

    def read(self, identity: str, consumer: str) -> tuple[int, ...]:
        value = self.values[identity]
        if value.kind is not DataKind.HOST_VALID:
            chain = []
            pending = [identity]
            while pending:
                item = pending.pop()
                source = self.values[item]
                chain.append(f"{item} <- {source.producer} ({source.kind.value})")
                pending.extend(source.parents)
            raise ValueError("unverified host control read: " + "; ".join(chain))
        self.reads.append((identity, consumer))
        return value.values
