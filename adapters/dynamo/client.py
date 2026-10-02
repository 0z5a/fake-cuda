import json
from pathlib import Path

from adapters.common.bridge import LineProcess
from adapters.vllm.routing import Decision


class DynamoPolicy(LineProcess):
    def __init__(self, python: Path, workers: int):
        super().__init__([str(python), str(Path(__file__).with_name("main.py")), "--workers", str(workers)])
        self.final_loads: list[dict] = []

    def choose(self, request: str, tokens: list[int], output: int, queues: list[int]) -> Decision:
        response = json.loads(self.command(json.dumps({"operation": "choose", "request": request,
                                                       "tokens": tokens, "output": output})))
        return Decision(response["endpoint"], {}, tuple(response["loads"]))

    def feedback(self, prefills: list[str], finished: list[str]) -> None:
        response = json.loads(self.command(json.dumps({"operation": "feedback", "prefills": prefills, "finished": finished})))
        if not response["ok"]:
            raise RuntimeError("native Dynamo rejected completion feedback")

    def close(self) -> None:
        self.final_loads = json.loads(self.command(json.dumps({"operation": "loads"})))["loads"]
        super().close()
