"""Measured-objective acceptance contract for indefinite joint curvature.

Motivation (T130)
------------------
``QuadraticTask`` only enforces that the *regularized private* curvature
block (``h_phiphi + mu * I``) is positive definite. It never checks the
shared block ``h_xx`` and it never checks the *effective* (Schur-complement)
curvature that governs how a step in the shared coordinate actually changes
the objective once the private coordinate is allowed to respond.

It is therefore possible to construct tasks where:

* the private curvature is positive definite (required, always true), and
* the shared block ``h_xx`` is *also* positive definite,

while the Schur complement ``task.schur()`` -- the true joint/effective
curvature -- has a negative eigenvalue. ``trust_region_optimum`` already
refuses to operate directly on an indefinite Hessian (it raises
``CurvatureError``), which is the correct behaviour *provided the caller
remembers to pass the Schur complement rather than ``h_xx`` alone*. A
hypothesized implementation that instead reasons "private curvature is
positive, therefore this must be safe" and trusts the compensated
(Schur-based) predicted change without ever taking a fresh, independent
measurement is exactly the "silent convex projection" failure mode the T130
research claim warns about.

This module implements a defense-in-depth acceptance contract that does not
rely solely on an eigenvalue sign check: it additionally requires a *fresh
measured* objective change -- the exact quadratic value evaluated with the
private coordinate held at a concrete, currently-realized value (by default
zero / "not yet re-optimized"), as opposed to the compensated model's
optimistic assumption that the private coordinate has already jumped to its
analytic optimum. A step is only accepted when both signals agree that the
objective genuinely decreases; when they disagree the contract attempts a
safe reduction (removing the component of the step along known
negative-curvature directions of the Schur complement) before falling back
to outright rejection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .quadratic import FloatArray, QuadraticTask, _array

_REDUCTION_NORM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SchurCurvature:
    """Eigen-decomposition of a task's effective (Schur complement) curvature."""

    eigenvalues: FloatArray
    eigenvectors: FloatArray
    min_eigenvalue: float
    tolerance: float
    indefinite: bool
    negative_directions: FloatArray


def schur_curvature(task: QuadraticTask, *, tolerance: float = 1e-9) -> SchurCurvature:
    """Eigen-decompose ``task.schur()`` and flag negative-curvature directions."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    matrix = task.schur()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    min_eigenvalue = float(np.min(eigenvalues))
    indefinite = min_eigenvalue < -tolerance
    negative_mask = eigenvalues < -tolerance
    negative_directions = eigenvectors[:, negative_mask].T.copy()
    return SchurCurvature(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        min_eigenvalue=min_eigenvalue,
        tolerance=tolerance,
        indefinite=indefinite,
        negative_directions=negative_directions,
    )


def frozen_measured_change(
    task: QuadraticTask,
    global_step: NDArray[np.floating],
    *,
    private_step: NDArray[np.floating] | None = None,
) -> float:
    """Exact objective change with the private coordinate held fixed.

    This models a *realistic* measurement: the private parameters have not
    (yet) jumped to the analytic optimum that the compensated/Schur model
    assumes; by default they are held at zero (their value before the
    step). This is the "fresh measured" counterpart to
    ``QuadraticTask.compensated_change``, which always assumes instantaneous,
    ideal private re-optimization.
    """

    private_dim = task.h_phiphi.shape[0]
    private = (
        np.zeros(private_dim)
        if private_step is None
        else _array(private_step, name="private_step")
    )
    return task.direct_change(global_step, private)


def project_out_negative_curvature(
    curvature: SchurCurvature,
    global_step: NDArray[np.floating],
) -> FloatArray:
    """Remove the components of ``global_step`` along unsafe eigen-directions."""

    step = _array(global_step, name="global_step")
    reduced = step.copy()
    for direction in curvature.negative_directions:
        reduced = reduced - float(direction @ reduced) * direction
    return reduced


@dataclass(frozen=True)
class StepAcceptanceRecord:
    """Result of applying the measured-objective acceptance contract to one step."""

    step: FloatArray
    predicted_change: float
    measured_change: float
    naive_accept: bool
    corrected_accept: bool
    false_accept: bool
    indefinite: bool
    min_eigenvalue: float
    verdict: str
    reduced_step: FloatArray | None
    reduced_predicted_change: float | None
    reduced_measured_change: float | None
    reduced_accept: bool | None


def evaluate_step(
    task: QuadraticTask,
    global_step: NDArray[np.floating],
    *,
    tolerance: float = 1e-9,
    attempt_reduction: bool = True,
) -> StepAcceptanceRecord:
    """Apply the corrected (measured-objective) acceptance contract to a step.

    ``naive_accept`` mirrors a rule that trusts only the compensated
    (Schur-based) *predicted* change -- i.e. "positive private curvature,
    plus a model that says this decreases, is proof enough". ``corrected_accept``
    additionally requires the freshly measured (private-frozen) change to
    also show a genuine decrease. When the two disagree, this is a
    ``false_accept``: the naive rule would let the step through even though
    a fresh measurement shows it does not actually help. When that happens
    and the task's effective curvature is indefinite, the contract attempts
    to reduce the step by projecting out the offending negative-curvature
    components before declaring an outright rejection.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    step = _array(global_step, name="global_step")
    curvature = schur_curvature(task, tolerance=tolerance)

    predicted = task.compensated_change(step)
    measured = frozen_measured_change(task, step)
    naive_accept = predicted < -tolerance
    corrected_accept = naive_accept and measured < -tolerance
    false_accept = naive_accept and not corrected_accept

    reduced_step: FloatArray | None = None
    reduced_predicted: float | None = None
    reduced_measured: float | None = None
    reduced_accept: bool | None = None

    if corrected_accept:
        verdict = "accept"
    else:
        verdict = "reject"
        if attempt_reduction and curvature.indefinite:
            candidate = project_out_negative_curvature(curvature, step)
            if np.linalg.norm(candidate) > _REDUCTION_NORM_TOLERANCE:
                reduced_step = candidate
                reduced_predicted = task.compensated_change(candidate)
                reduced_measured = frozen_measured_change(task, candidate)
                reduced_accept = (
                    reduced_predicted < -tolerance and reduced_measured < -tolerance
                )
                if reduced_accept:
                    verdict = "reduce"

    return StepAcceptanceRecord(
        step=step,
        predicted_change=predicted,
        measured_change=measured,
        naive_accept=naive_accept,
        corrected_accept=corrected_accept,
        false_accept=false_accept,
        indefinite=curvature.indefinite,
        min_eigenvalue=curvature.min_eigenvalue,
        verdict=verdict,
        reduced_step=reduced_step,
        reduced_predicted_change=reduced_predicted,
        reduced_measured_change=reduced_measured,
        reduced_accept=reduced_accept,
    )


def naive_predicted_only_accept(
    task: QuadraticTask,
    global_step: NDArray[np.floating],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Baseline rule: accept whenever the compensated model predicts descent.

    This is the rule implicitly justified by "positive private curvature
    alone is insufficient" -- it never independently measures the objective
    change and never inspects the Schur complement's eigenvalues; it simply
    trusts the compensated model's sign. It is provided so its false-accept
    rate can be measured against ``evaluate_step``'s corrected contract.
    """

    step = _array(global_step, name="global_step")
    predicted = task.compensated_change(step)
    return predicted < -tolerance
