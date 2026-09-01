"""Stability-regime control: choose the step size ``eta`` to realize a target
spectral radius of the per-step state Jacobian.

Matches ``docs/theory/oracle-spec.md`` section 8: "stable" needs
``rho < 1``; "deliberately unstable" needs ``rho`` in the open interval
``(1.05, 2.0)``. For SGD the spectral radius of ``M_i = I - eta*C_i`` has a
closed form in terms of ``C_i``'s extreme eigenvalues, used exactly. For
momentum the augmented transition matrix ``A_i`` has no equally simple
closed form as a function of the whole spectrum of ``C_i``, so ``eta`` is
found numerically (bisection/bounded minimization on the true spectral
radius, computed by direct eigendecomposition of ``A_i``); the realized
value is always measured and recorded rather than assumed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar

from comppareto.oracle.momentum import momentum_transition_jacobian
from comppareto.oracle.tasks import OracleTask

FloatArray = NDArray[np.float64]

StabilityRegime = Literal["stable", "unstable"]

UNSTABLE_RHO_RANGE = (1.05, 2.0)


def spectral_radius(matrix: FloatArray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def sgd_eta_for_regime(
    eigs_c: FloatArray, regime: StabilityRegime, rng: np.random.Generator
) -> tuple[float, float]:
    """Exact closed-form ``eta`` and realized spectral radius of ``M_i = I - eta*C_i``."""

    lam_min = float(np.min(eigs_c))
    lam_max = float(np.max(eigs_c))
    if regime == "stable":
        eta = 2.0 / (lam_min + lam_max)
        realized = (lam_max - lam_min) / (lam_max + lam_min)
        return eta, realized
    if regime == "unstable":
        target_rho = float(rng.uniform(*UNSTABLE_RHO_RANGE))
        eta = (1.0 + target_rho) / lam_max
        return eta, target_rho
    raise ValueError(f"unknown stability regime {regime!r}")


def momentum_eta_for_regime(
    task: OracleTask, beta: float, regime: StabilityRegime, rng: np.random.Generator
) -> tuple[float, float]:
    """Numeric ``eta`` and measured realized spectral radius of ``A_i``."""

    c = task.private_curvature
    lam_max = float(np.max(np.linalg.eigvalsh(c)))

    def rho(eta: float) -> float:
        return spectral_radius(momentum_transition_jacobian(task, eta, beta))

    if regime == "stable":
        result = minimize_scalar(rho, bounds=(1e-8, 2.0 / lam_max), method="bounded")
        eta = float(result.x)
        return eta, float(result.fun)
    if regime == "unstable":
        target_rho = float(rng.uniform(*UNSTABLE_RHO_RANGE))
        optimum = minimize_scalar(rho, bounds=(1e-8, 2.0 / lam_max), method="bounded")
        lo = float(optimum.x)
        hi = lo
        step = max(lo, 1e-6)
        while rho(hi) < target_rho and hi < 1e6 / lam_max:
            hi += step
            step *= 2.0
        if rho(hi) < target_rho:
            # Target unreachable in the search range; report the best available point.
            return hi, rho(hi)
        eta = brentq(lambda e: rho(e) - target_rho, lo, hi, xtol=1e-12)
        return float(eta), float(rho(eta))
    raise ValueError(f"unknown stability regime {regime!r}")
