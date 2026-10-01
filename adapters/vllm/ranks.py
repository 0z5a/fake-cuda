"""Original schedulers behind a synchronous, timestamped rank protocol."""
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
import time
from typing import Literal

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.core.sched.output import SchedulerOutput

from adapters.common.control import ControlLedger
from adapters.vllm.costs import Shape
from adapters.vllm.worker import add_request, complete_step, make_scheduler, scheduled_shape

Configuration = tuple[VllmConfig, KVCacheConfig, int, int]
Operation = Literal["reset", "arrive", "schedule", "complete"]


@dataclass(frozen=True)
class Command:
    sequence: int
    epoch: int
    virtual_ns: int
    operation: Operation
    arrivals: tuple[tuple[str, int, int, int], ...] = ()
    configuration: Configuration | None = None


@dataclass(frozen=True)
class Reply:
    sequence: int
    epoch: int
    virtual_ns: int
    pid: int
    unfinished: int
    waiting: int
    control_reads: int
    shape: Shape | None = None
    token_counts: dict[str, int] | None = None
    visible: tuple[tuple[str, tuple[int, ...], bool], ...] = ()


@dataclass(frozen=True)
class Failure:
    message: str


class LocalRank:
    def __init__(self, configuration: Configuration):
        self.scheduler = make_scheduler(configuration)
        self.ledger = ControlLedger()
        self.scheduled: SchedulerOutput | None = None

    def command(self, command: Command) -> Reply:
        shape = None
        counts = None
        visible = ()
        if command.operation == "reset":
            assert command.configuration is not None
            self.scheduler = make_scheduler(command.configuration)
            self.ledger = ControlLedger()
            self.scheduled = None
        elif command.operation == "arrive":
            for identity, prompt, output, arrival in command.arrivals:
                add_request(self.scheduler, identity, prompt, output, arrival)
        elif command.operation == "schedule":
            if self.scheduled is not None:
                raise ValueError("rank already has an in-flight step")
            self.scheduled = self.scheduler.schedule()
            shape = scheduled_shape(self.scheduler, self.scheduled)
            counts = dict(self.scheduled.num_scheduled_tokens)
        elif command.operation == "complete":
            if self.scheduled is None:
                raise ValueError("rank has no in-flight step")
            outputs = complete_step(self.scheduler, self.scheduled, self.ledger, str(command.sequence))
            visible = tuple((rid, tuple(tokens), finished) for rid, tokens, finished in outputs)
            self.scheduled = None
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU rank initialized CUDA")
        return Reply(command.sequence, command.epoch, command.virtual_ns, os.getpid(),
                     self.scheduler.get_num_unfinished_requests(), len(self.scheduler.waiting),
                     len(self.ledger.reads), shape, counts, visible)


def rank_main(connection: Connection, configuration: Configuration, delay_s: float) -> None:
    sequence = 0
    epoch = 0
    virtual_ns = 0
    try:
        rank = LocalRank(configuration)
        while True:
            command = connection.recv()
            if not isinstance(command, Command) or command.sequence != sequence + 1:
                raise ValueError("invalid rank command sequence or timestamp")
            if command.operation == "reset":
                if command.epoch != epoch + 1 or command.virtual_ns != 0:
                    raise ValueError("invalid rank run epoch")
                epoch, virtual_ns = command.epoch, 0
            elif command.epoch != epoch or command.virtual_ns < virtual_ns:
                raise ValueError("invalid rank run epoch or timestamp")
            sequence, virtual_ns = command.sequence, command.virtual_ns
            reply = rank.command(command)
            if delay_s:
                time.sleep(delay_s)
            connection.send(reply)
    except EOFError:
        pass
    except Exception as error:
        # Surface failures at the process boundary so the parent cannot silently
        # certify a failed rank or wait for a reply that will never arrive.
        connection.send(Failure(f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


class RankGroup:
    def __init__(self, configurations: list[Configuration], processes: bool = False, delay_s: float = 0):
        if not configurations or delay_s < 0:
            raise ValueError("invalid rank group")
        self.sequence = 0
        self.epoch = 0
        self.size = len(configurations)
        self.locals = [] if processes else [LocalRank(config) for config in configurations]
        self.connections: list[Connection] = []
        self.processes: list[mp.Process] = []
        self.pids: set[int] = set()
        if processes:
            context = mp.get_context("spawn")
            for configuration in configurations:
                parent, child = context.Pipe()
                process = context.Process(target=rank_main, args=(child, configuration, delay_s))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)

    def command(self, operation: Operation, virtual_ns: int,
                arrivals: tuple[tuple[str, int, int, int], ...] = (),
                configurations: list[Configuration] | None = None) -> list[Reply]:
        if configurations is not None and len(configurations) != self.size:
            raise ValueError("configuration/rank count mismatch")
        self.sequence += 1
        if operation == "reset":
            self.epoch += 1
        commands = [Command(self.sequence, self.epoch, virtual_ns, operation, arrivals,
                            configurations[i] if configurations is not None else None)
                    for i in range(self.size)]
        if self.locals:
            replies = [rank.command(command) for rank, command in zip(self.locals, commands)]
        else:
            for connection, command in zip(self.connections, commands):
                connection.send(command)
            replies = []
            for connection in self.connections:
                reply = connection.recv()
                if isinstance(reply, Failure):
                    raise RuntimeError("protocol_failure: " + reply.message)
                if not isinstance(reply, Reply):
                    raise RuntimeError("protocol_failure: invalid rank response")
                replies.append(reply)
        for reply in replies:
            if reply.sequence != self.sequence or reply.epoch != self.epoch or reply.virtual_ns != virtual_ns:
                raise RuntimeError("protocol_failure: rank reply identity mismatch")
            self.pids.add(reply.pid)
        return replies

    def close(self) -> None:
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            process.join()
            if process.exitcode:
                raise RuntimeError(f"CPU rank exited with {process.exitcode}")
