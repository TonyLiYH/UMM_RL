"""Analytic counterexamples: positive private blocks, indefinite joint curvature.

Every case below is constructed so that:

* ``h_phiphi + mu * I`` (the regularized private curvature) is positive
  definite -- required by ``QuadraticTask`` and always satisfied here;
* ``h_xx`` (the shared/local curvature block) is *also* positive definite
  on its own;
* the Schur complement ``h_xx - h_xphi @ private_curvature^-1 @ h_xphi.T``
  (the true effective/joint curvature) is indefinite, with an eigenvalue
  and eigenvector known in closed form by construction.

The construction is the isotropic rank-k perturbation
``h_xx = a * I``, ``M = h_xphi @ private_curvature^-1 @ h_xphi.T``, where the
columns of ``h_xphi`` are mutually orthogonal vectors ``c_j`` and
``private_curvature`` is diagonal with entries ``p_j``. Then
``M = sum_j (1/p_j) c_j c_j^T`` is diagonalized by the ``c_j`` directions
with eigenvalue ``||c_j||^2 / p_j`` (and 0 on the orthogonal complement), so
``schur = a * I - M`` has closed-form eigenvalues ``a - ||c_j||^2 / p_j``
along ``c_j / ||c_j||`` and ``a`` elsewhere. All eigenvalues below were
cross-checked numerically against ``np.linalg.eigh(task.schur())`` in
``tests/test_trust_region_guard.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .quadratic import QuadraticTask

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Counterexample:
    """One analytically-constructed acceptance test case."""

    name: str
    category: str  # "unsafe" | "safe" | "reducible"
    description: str
    task: QuadraticTask
    step: FloatArray
    radius: float
    analytic_min_eigenvalue: float
    analytic_negative_direction: FloatArray | None
    expected_verdict: str  # "accept" | "reject" | "reduce"


def build_counterexample_suite() -> list[Counterexample]:
    cases: list[Counterexample] = []

    # --- Case 1: pure curvature, zero gradient, 2D shared / 1D private ---
    # h_xx = I_2 (PD), h_xphi = [3, 4]^T, h_phiphi = 0, mu = 1
    # => private_curvature = [1] (PD). c = [3,4], ||c||^2 = 25, p = 1.
    # schur eigenvalues: 1 - 25/1 = -24 along [0.6, 0.8]; +1 along [-0.8, 0.6].
    task1 = QuadraticTask(
        local_gradient=np.zeros(2),
        h_xx=np.eye(2),
        h_xphi=np.array([[3.0], [4.0]]),
        h_phiphi=np.array([[0.0]]),
        mu=1.0,
        selector=np.eye(2),
    )
    cases.append(
        Counterexample(
            name="pure_curvature_2d",
            category="unsafe",
            description=(
                "Zero gradient; step is the unit negative-curvature direction of "
                "schur ([0.6, 0.8], eigenvalue -24). Compensated model predicts a "
                "large decrease (-12) by assuming the private coordinate "
                "instantaneously re-optimizes; frozen/measured change is +0.5 "
                "(a genuine increase)."
            ),
            task=task1,
            step=np.array([0.6, 0.8]),
            radius=1.0,
            analytic_min_eigenvalue=-24.0,
            analytic_negative_direction=np.array([0.6, 0.8]),
            expected_verdict="reject",
        )
    )

    # --- Case 2: safe control, 2D, schur stays positive definite ---
    # h_xphi = [1, 0]^T, mu = 4 (p = 4). ||c||^2/p = 1/4 < a = 1 => schur PD.
    task_safe = QuadraticTask(
        local_gradient=np.array([-1.0, 0.0]),
        h_xx=np.eye(2),
        h_xphi=np.array([[1.0], [0.0]]),
        h_phiphi=np.array([[0.0]]),
        mu=4.0,
        selector=np.eye(2),
    )
    cases.append(
        Counterexample(
            name="safe_control_2d",
            category="safe",
            description=(
                "Control case: schur stays positive definite (eigenvalues "
                "0.75, 1.0). Predicted (-0.406) and measured (-0.375) both show "
                "a genuine decrease; the contract must accept, not just reject "
                "everything that touches a private/coupled block."
            ),
            task=task_safe,
            step=np.array([0.5, 0.0]),
            radius=0.5,
            analytic_min_eigenvalue=0.75,
            analytic_negative_direction=None,
            expected_verdict="accept",
        )
    )

    # --- Case 3a/3b: 3D shared, 2D private, TWO known negative directions ---
    # h_xx = I_3, columns c1=[3,4,0], c2=[0,0,5] orthogonal, private_curvature = I_2.
    # schur eigenvalues: -24 along [0.6,0.8,0]; -24 along [0,0,1]; +1 along [0.8,-0.6,0].
    task3 = QuadraticTask(
        local_gradient=np.zeros(3),
        h_xx=np.eye(3),
        h_xphi=np.array([[3.0, 0.0], [4.0, 0.0], [0.0, 5.0]]),
        h_phiphi=np.zeros((2, 2)),
        mu=1.0,
        selector=np.eye(3),
    )
    cases.append(
        Counterexample(
            name="two_negative_directions_dir_a",
            category="unsafe",
            description=(
                "3D shared / 2D private task with two independent negative-"
                "curvature directions (schur eigenvalues -24, -24, +1). Step "
                "along the first negative direction [0.6, 0.8, 0]: predicted "
                "-12, measured +0.5."
            ),
            task=task3,
            step=np.array([0.6, 0.8, 0.0]),
            radius=1.0,
            analytic_min_eigenvalue=-24.0,
            analytic_negative_direction=np.array([0.6, 0.8, 0.0]),
            expected_verdict="reject",
        )
    )
    cases.append(
        Counterexample(
            name="two_negative_directions_dir_b",
            category="unsafe",
            description=(
                "Same task as two_negative_directions_dir_a; step along the "
                "second, independent negative direction [0, 0, 1]: predicted "
                "-12, measured +0.5. Confirms both analytic negative-curvature "
                "directions individually produce a false accept under the "
                "naive (predicted-only) rule."
            ),
            task=task3,
            step=np.array([0.0, 0.0, 1.0]),
            radius=1.0,
            analytic_min_eigenvalue=-24.0,
            analytic_negative_direction=np.array([0.0, 0.0, 1.0]),
            expected_verdict="reject",
        )
    )

    # --- Case 4: reducible -- mixed safe + unsafe step, reduction recovers a
    # genuinely improving step ---
    # Reuse task1's schur eigenbasis: v1=[0.6,0.8] (eigenvalue -24, unsafe),
    # v2=[-0.8,0.6] (eigenvalue +1, safe). Gradient favors v2 strongly.
    v1 = np.array([0.6, 0.8])
    v2 = np.array([-0.8, 0.6])
    gradient5 = -5.0 * v2
    task5 = QuadraticTask(
        local_gradient=gradient5,
        h_xx=np.eye(2),
        h_xphi=np.array([[3.0], [4.0]]),
        h_phiphi=np.array([[0.0]]),
        mu=1.0,
        selector=np.eye(2),
    )
    unsafe_mixed_step = 3.2 * v1 + 1.0 * v2
    cases.append(
        Counterexample(
            name="reducible_mixed_step",
            category="reducible",
            description=(
                "Step mixes a favorable safe-direction component (1.0 * v2, "
                "gradient-aligned) with a large unsafe-direction component "
                "(3.2 * v1). Predicted change is a wildly optimistic -127.38 "
                "(exploiting the -24 eigenvalue); measured/frozen change is "
                "+0.62 (a genuine increase) -- a false accept under the naive "
                "rule. Projecting out the v1 component yields a reduced step "
                "(1.0 * v2) whose predicted and measured changes agree exactly "
                "(-4.5) and are both negative: a genuine, smaller improvement "
                "survives the reduction."
            ),
            task=task5,
            step=unsafe_mixed_step,
            radius=4.0,
            analytic_min_eigenvalue=-24.0,
            analytic_negative_direction=v1,
            expected_verdict="reduce",
        )
    )

    # --- Case 5: indefinite schur, but a strongly favorable gradient makes
    # the step genuinely safe (contract must not be overly conservative) ---
    gradient6 = -5.0 * v1
    task6 = QuadraticTask(
        local_gradient=gradient6,
        h_xx=np.eye(2),
        h_xphi=np.array([[3.0], [4.0]]),
        h_phiphi=np.array([[0.0]]),
        mu=1.0,
        selector=np.eye(2),
    )
    cases.append(
        Counterexample(
            name="indefinite_but_genuinely_safe",
            category="safe",
            description=(
                "Same indefinite schur as pure_curvature_2d, but the step "
                "(the unconstrained Newton step on h_xx alone, [3, 4]) is "
                "strongly gradient-favored: predicted -325, measured -12.5. "
                "Both signals agree the step helps, so the contract must "
                "accept despite the indefinite effective curvature -- "
                "rejecting everything indefinite would be overly conservative."
            ),
            task=task6,
            step=np.array([3.0, 4.0]),
            radius=6.0,
            analytic_min_eigenvalue=-24.0,
            analytic_negative_direction=v1,
            expected_verdict="accept",
        )
    )

    return cases
