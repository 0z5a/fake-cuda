from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from adapters.common.bridge import LineProcess
from adapters.vllm.costs import Shape, UnsupportedStep
from adapters.aisimulate.identity import Identity


class AisStepCost(LineProcess):
    scope = "whole_step_including_host_and_communication"

    def __init__(self, python: Path, calibration: Path, identity: Identity):
        executable = Path(__file__).with_name("main.py")
        digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
        super().__init__([str(python), str(executable), "--calibration", str(calibration),
                          "--sha256", digest, "--identity", json.dumps(asdict(identity))])
        self.cache: dict[Shape, int] = {}

    def predict(self, shape: Shape) -> int:
        if shape not in self.cache:
            response = self.command(f"{shape.phase} {shape.batch} {shape.context}")
            if response.startswith("unsupported "):
                raise UnsupportedStep(response.removeprefix("unsupported "))
            self.cache[shape] = int(response)
        return self.cache[shape]
