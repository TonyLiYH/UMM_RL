"""The framework-neutral parameter primitive.

This instrumentation layer deliberately depends on nothing beyond ``numpy``
(already a declared project dependency; see ``pyproject.toml``, outside this
task's ``allowed_paths``) rather than any specific deep-learning framework.
:class:`Parameter` mirrors the one property every framework's parameter
object has that this task's measurements actually need: a mutable value
array plus a mutable, possibly-``None`` gradient array of the same shape.
Object identity (``id(parameter)``) is used throughout the recorder as the
stable key for a parameter, exactly as ``torch.nn.Parameter`` identity would
be used against an ``optimizer.state`` dict in a torch-based system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Parameter:
    """A named, mutable value array with an accumulating gradient slot."""

    value: np.ndarray
    grad: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        self.value = np.asarray(self.value, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def numel(self) -> int:
        return int(self.value.size)

    def zero_grad(self) -> None:
        self.grad = None

    def accumulate_grad(self, contribution: np.ndarray) -> None:
        contribution = contribution.reshape(self.value.shape)
        self.grad = contribution.copy() if self.grad is None else self.grad + contribution
