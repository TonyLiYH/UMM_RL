"""A minimal, framework-neutral AdamW optimizer over :class:`Parameter`.

Implements exactly the decoupled-weight-decay AdamW update
(`Loshchilov & Hutter, 2019 <https://arxiv.org/abs/1711.05101>`_), matching
``torch.optim.AdamW``'s (non-amsgrad) formula term for term, so that the
"AdamW moment summaries" this task must record (``exp_avg``/``exp_avg_sq``
per block, plus ``step``) mean exactly what they would in a torch-based
system: this is a genuine AdamW state, not a stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import Parameter


@dataclass
class _State:
    step: int = 0
    exp_avg: np.ndarray | None = None
    exp_avg_sq: np.ndarray | None = None


@dataclass
class AdamWOptimizer:
    parameters: tuple[Parameter, ...]
    lr: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    state: dict[int, _State] = field(default_factory=dict, init=False)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad = np.zeros_like(parameter.value)

    def step(self) -> None:
        beta1, beta2 = self.betas
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            key = id(parameter)
            state = self.state.setdefault(key, _State())
            if state.exp_avg is None:
                state.exp_avg = np.zeros_like(parameter.value)
                state.exp_avg_sq = np.zeros_like(parameter.value)
            state.step += 1

            grad = parameter.grad
            if self.weight_decay != 0.0:
                parameter.value = parameter.value * (1.0 - self.lr * self.weight_decay)

            state.exp_avg = beta1 * state.exp_avg + (1.0 - beta1) * grad
            state.exp_avg_sq = beta2 * state.exp_avg_sq + (1.0 - beta2) * (grad * grad)

            bias_correction1 = 1.0 - beta1**state.step
            bias_correction2 = 1.0 - beta2**state.step
            step_size = self.lr / bias_correction1
            denom = np.sqrt(state.exp_avg_sq) / np.sqrt(bias_correction2) + self.eps

            parameter.value = parameter.value - step_size * state.exp_avg / denom

    def moment_state(self, parameter: Parameter) -> dict[str, object]:
        """Read-only view of a parameter's optimizer state, keyed the same
        shape as a torch ``optimizer.state[parameter]`` dict would be
        (``exp_avg``, ``exp_avg_sq``, ``step``), or ``{}`` if uninitialized.
        """
        state = self.state.get(id(parameter))
        if state is None or state.exp_avg is None:
            return {}
        return {"exp_avg": state.exp_avg, "exp_avg_sq": state.exp_avg_sq, "step": state.step}
