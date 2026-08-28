"""Independent high-accuracy common-descent/Pareto reference over lifted task gradients.

Matches ``docs/theory/oracle-spec.md`` section 6's rerun-response gradients,
lifted into the shared theta-space via each task's selector ``P_i``
(:mod:`comppareto.oracle.selectors`). Given the ``m`` lifted gradients
``g_1,...,g_m`` in ``R^P``, this module finds the minimum-norm point in their
convex hull -- the classical multiple-gradient common-descent direction
(Desideri 2012 / Sener & Koltun 2018) -- by two independent methods:

(a) :func:`min_norm_point_active_set` -- exact active-set enumeration. For
    ``m<=8`` this enumerates every candidate active subset, solves the
    linear KKT system on that subset exactly, and checks feasibility and
    optimality, giving the exact global optimum of the convex QP;
(b) :func:`min_norm_point_frank_wolfe` -- an iterative conditional-gradient
    solver with a closed-form exact line search (the objective is
    quadratic), converged to a tight tolerance, as an independent numerical
    cross-check of (a) that does not solve any linear system.

Both report the same optimality certificate: the KKT residual
``max(0, mu - min_i g_i . w)`` (should be ~0 at the optimum) and the
active-set consistency ``max_i in active |g_i . w - mu|``, where ``w`` is the
candidate min-norm point and ``mu = ||w||^2``.

Deliberately independent of ``src/comppareto/quadratic.py`` (outside this
task's ``allowed_paths``), matching the established convention in
``selectors.py``/``tasks.py`` of self-contained reimplementation.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lift_gradient(selector: FloatArray, local_gradient: FloatArray) -> FloatArray:
    """``P_i^T g_i``: lift a task-local gradient into the shared theta-space."""

    return selector.T @ local_gradient


@dataclass(frozen=True)
class ParetoReference:
    """A candidate min-norm point in the convex hull of the lifted gradients."""

    weights: FloatArray
    combined_gradient: FloatArray
    direction: FloatArray
    objective: float
    stationary: bool
    active_set: tuple[int, ...]
    kkt_residual: float
    active_consistency_residual: float
    method: str


def _solve_equality_subproblem(gram: FloatArray, active: tuple[int, ...]) -> tuple[FloatArray, float] | None:
    """Solve ``Gram_active @ lambda = mu * 1``, ``sum(lambda) = 1`` exactly.

    Returns ``(lambda_active, mu)`` or ``None`` if the augmented system is
    singular (degenerate Gram submatrix).
    """

    k = len(active)
    sub = gram[np.ix_(active, active)]
    a = np.zeros((k + 1, k + 1))
    a[:k, :k] = sub
    a[:k, k] = -1.0
    a[k, :k] = 1.0
    rhs = np.zeros(k + 1)
    rhs[k] = 1.0
    try:
        sol = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        return None
    return sol[:k], float(sol[k])


def min_norm_point_active_set(
    gradients: Sequence[FloatArray], *, tolerance: float = 1e-10
) -> ParetoReference:
    """Exact global minimum-norm point in ``conv(gradients)`` via active-set enumeration.

    Feasible for the oracle's ``m<=8`` task count: enumerates all ``2^m-1``
    non-empty subsets, solves the equality-constrained sub-problem on each
    exactly, and keeps the feasible (``lambda>=0``) candidate satisfying the
    KKT optimality certificate with the smallest objective.
    """

    g = np.stack([np.asarray(v, dtype=np.float64) for v in gradients])
    m = g.shape[0]
    gram = g @ g.T

    best: tuple[float, FloatArray, FloatArray, float, float, tuple[int, ...]] | None = None
    for size in range(1, m + 1):
        for active in itertools.combinations(range(m), size):
            solved = _solve_equality_subproblem(gram, active)
            if solved is None:
                continue
            lam_active, _mu = solved
            if np.any(lam_active < -tolerance):
                continue
            lam = np.zeros(m)
            lam[list(active)] = np.clip(lam_active, 0.0, None)
            total = lam.sum()
            if total <= 0:
                continue
            lam /= total
            w = lam @ g
            objective = float(w @ w)
            proj = g @ w
            kkt_residual = max(0.0, objective - float(np.min(proj)))
            active_mask = lam > tolerance
            active_consistency = (
                float(np.max(np.abs(proj[active_mask] - objective))) if active_mask.any() else 0.0
            )
            if kkt_residual <= tolerance and active_consistency <= tolerance:
                if best is None or objective < best[0]:
                    best = (
                        objective,
                        lam,
                        w,
                        kkt_residual,
                        active_consistency,
                        tuple(int(i) for i in np.nonzero(active_mask)[0]),
                    )

    if best is None:
        # The full-support subset (active = every task) is always tried above
        # and always feasible for a convex hull membership problem, so this
        # branch is unreachable for a correct Gram matrix; kept as an explicit
        # guard rather than silently returning a wrong answer.
        raise RuntimeError("active-set enumeration found no KKT-feasible candidate")

    objective, lam, w, kkt_residual, active_consistency, active_set = best
    stationary = objective <= tolerance
    direction = np.zeros_like(w) if stationary else -w
    return ParetoReference(
        weights=lam,
        combined_gradient=w,
        direction=direction,
        objective=objective,
        stationary=stationary,
        active_set=active_set,
        kkt_residual=kkt_residual,
        active_consistency_residual=active_consistency,
        method="active_set_enumeration",
    )


def min_norm_point_frank_wolfe(
    gradients: Sequence[FloatArray], *, iterations: int = 20000, tolerance: float = 1e-13
) -> ParetoReference:
    """Independent iterative cross-check via conditional gradient (Frank-Wolfe).

    Distinct code path from :func:`min_norm_point_active_set`: never forms or
    solves a linear system. At each iteration, moves toward the vertex
    ``e_s`` with the most negative correlation ``g_s . w``, using the
    closed-form exact line search available because ``||lambda @ g||^2`` is
    quadratic in the step length.
    """

    g = np.stack([np.asarray(v, dtype=np.float64) for v in gradients])
    m = g.shape[0]
    lam = np.full(m, 1.0 / m)
    w = lam @ g
    for _ in range(iterations):
        proj = g @ w
        s = int(np.argmin(proj))
        gap = float(w @ w - proj[s])
        if gap <= tolerance:
            break
        gs = g[s]
        denom = float((gs - w) @ (gs - w))
        if denom <= tolerance:
            break
        t = float(np.clip(gap / denom, 0.0, 1.0))
        lam = (1.0 - t) * lam
        lam[s] += t
        w = (1.0 - t) * w + t * gs
        if t <= tolerance:
            break

    objective = float(w @ w)
    proj = g @ w
    kkt_residual = max(0.0, objective - float(np.min(proj)))
    active_mask = lam > 1e-9
    active_consistency = (
        float(np.max(np.abs(proj[active_mask] - objective))) if active_mask.any() else 0.0
    )
    stationary = objective <= 1e-10
    direction = np.zeros_like(w) if stationary else -w
    return ParetoReference(
        weights=lam,
        combined_gradient=w,
        direction=direction,
        objective=objective,
        stationary=stationary,
        active_set=tuple(int(i) for i in np.nonzero(active_mask)[0]),
        kkt_residual=kkt_residual,
        active_consistency_residual=active_consistency,
        method="frank_wolfe",
    )


def case_pareto_reference(selectors: Sequence[FloatArray], local_gradients: Sequence[FloatArray]) -> dict:
    """Lift each task's local gradient and cross-check both min-norm-point solvers.

    Returns a plain-dict summary (weights, combined direction, objective,
    KKT residuals for both methods, and the disagreement between them)
    suitable for direct inclusion in a run manifest.
    """

    lifted = [lift_gradient(p, grad) for p, grad in zip(selectors, local_gradients)]
    exact = min_norm_point_active_set(lifted)
    iterative = min_norm_point_frank_wolfe(lifted)
    cross_check_error = float(np.linalg.norm(exact.combined_gradient - iterative.combined_gradient))
    return {
        "active_set": {
            "weights": exact.weights.tolist(),
            "combined_gradient": exact.combined_gradient.tolist(),
            "direction": exact.direction.tolist(),
            "objective": exact.objective,
            "stationary": exact.stationary,
            "active_set": list(exact.active_set),
            "kkt_residual": exact.kkt_residual,
            "active_consistency_residual": exact.active_consistency_residual,
        },
        "frank_wolfe": {
            "weights": iterative.weights.tolist(),
            "combined_gradient": iterative.combined_gradient.tolist(),
            "direction": iterative.direction.tolist(),
            "objective": iterative.objective,
            "stationary": iterative.stationary,
            "active_set": list(iterative.active_set),
            "kkt_residual": iterative.kkt_residual,
            "active_consistency_residual": iterative.active_consistency_residual,
        },
        "cross_check_error": cross_check_error,
    }
