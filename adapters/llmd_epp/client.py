import json
from pathlib import Path

from adapters.common.bridge import LineProcess


class NativeEpp(LineProcess):
    def __init__(self, executable: Path):
        super().__init__([str(executable)])

    def select(self, queues: list[int]) -> tuple[int, dict[str, float]]:
        snapshot = [{"id": str(i), "queue": queue, "enabled": True} for i, queue in enumerate(queues)]
        response = json.loads(self.command(json.dumps(snapshot)))
        return int(response["picked"]), response["scores"]
