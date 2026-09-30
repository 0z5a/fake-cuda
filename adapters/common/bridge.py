"""Own a ResourceEngine subprocess and let it exit naturally on EOF."""
from pathlib import Path
import subprocess


class LineProcess:
    def __init__(self, arguments: list[str]):
        self.process = subprocess.Popen(arguments, text=True,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        assert self.process.stdin is not None and self.process.stdout is not None

    def command(self, text: str) -> str:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline().strip()
        if not line:
            raise RuntimeError("embedding subprocess exited without a response")
        return line

    def close(self) -> None:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.close()
        self.process.stdout.close()
        if self.process.wait():
            raise RuntimeError("embedding subprocess failed")


class ResourceBridge(LineProcess):
    def __init__(self, executable: Path, ranks: int):
        super().__init__([str(executable), str(ranks)])

    def submit(self, identity: int, rank: int, arrival: int, cost: int) -> None:
        if self.command(f"submit {identity} {rank} {arrival} {cost}") != "ok":
            raise RuntimeError("ResourceEngine submission failed")

    def advance(self, time: int) -> None:
        if self.command(f"advance {time}") != "ok":
            raise RuntimeError("ResourceEngine advance failed")

    def next_event(self) -> int:
        return int(self.command("next"))

    def completion(self, identity: int) -> int:
        return int(self.command(f"status {identity}").split()[1])
