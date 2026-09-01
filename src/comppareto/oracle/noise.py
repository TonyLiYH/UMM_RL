"""Stochastic private-gradient noise models.

Matches ``docs/theory/oracle-spec.md`` section 8 ("Noise models"). Noise is
realized deterministically given a seeded ``numpy.random.Generator`` so that
finite-response trajectories are exactly reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

NoiseKind = Literal["gaussian", "block_correlated"]


@dataclass(frozen=True)
class NoiseModel:
    kind: NoiseKind
    sigma: float
    rho: float = 0.0
    sub_block_width: int | None = None

    def __post_init__(self) -> None:
        if self.sigma < 0 or not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite and non-negative")
        if self.kind == "block_correlated":
            if not (0.0 <= self.rho < 1.0):
                raise ValueError("rho must be in [0, 1) for block_correlated noise")
            if self.sub_block_width is not None and self.sub_block_width <= 0:
                raise ValueError("sub_block_width must be positive")
        elif self.kind != "gaussian":
            raise ValueError(f"unknown noise kind {self.kind!r}")

    def covariance(self, dim: int) -> FloatArray:
        if self.kind == "gaussian":
            return (self.sigma**2) * np.eye(dim)
        sub_width = self.sub_block_width or max(1, -(-dim // 4))
        group = np.arange(dim) // sub_width
        same_group = (group[:, None] == group[None, :]).astype(np.float64)
        return (self.sigma**2) * ((1.0 - self.rho) * np.eye(dim) + self.rho * same_group)

    def sample(self, rng: np.random.Generator, steps: int, dim: int) -> FloatArray:
        """Draw ``zeta_{0..steps-1}`` i.i.d. from ``N(0, covariance(dim))``."""

        cov = self.covariance(dim)
        if dim == 0:
            return np.zeros((steps, 0))
        chol = np.linalg.cholesky(cov + 1e-15 * np.eye(dim))
        eps = rng.standard_normal((steps, dim))
        return eps @ chol.T
