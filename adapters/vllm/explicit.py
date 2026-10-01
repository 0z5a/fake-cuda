"""Use frozen whole-model costs in the independent explicit-request simulator."""
from sim.cost import Cost
from sim.serving import Batch, SCOPE
from .costs import Shape, StepCost, UnsupportedStep


class UniformStepCost:
    scope = SCOPE

    def __init__(self, model: StepCost, source: str):
        if model.scope != SCOPE or not source:
            raise ValueError("whole-step source required")
        self.model, self.source = model, source

    def estimate(self, batch: Batch) -> Cost:
        contexts = {item.context for item in batch.items}
        if not batch.items or len(contexts) != 1:
            raise UnsupportedStep("heterogeneous contexts need independent calibration")
        if batch.phase == "prefill" and any(item.tokens != item.context for item in batch.items):
            raise UnsupportedStep("chunked continuation is outside unchunked whole-step calibration")
        shape = Shape(batch.phase, len(batch.items), contexts.pop())
        return Cost(self.model.predict(shape), "calibrated_estimate", self.source)
