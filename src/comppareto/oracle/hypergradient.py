"""Rerun/commit gradients and the exact quadratic loss-change identity.

Matches ``docs/theory/oracle-spec.md`` section 6. Optimizer-agnostic: takes
the task-local sensitivity ``Z_K`` (``Z_K^phi`` for momentum) and the affine
remainder ``r_K = phi_K(x_i=0)`` as inputs, so the same formulas serve both
SGD and momentum.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from comppareto.oracle.tasks import OracleTask

FloatArray = NDArray[np.float64]


def quadratic_model(task: OracleTask, z_k: FloatArray, r_k: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``(hat_g_i, Q_i^K)``: the exact quadratic model of ``F_i^{K,rerun}`` in ``x_i``."""

    h_xx = task.h_xx
    h_xphi = task.h_xphi
    h_phix = task.h_phix
    h_phiphi = task.h_phiphi
    q = h_xx + h_xphi @ z_k + z_k.T @ h_phix + z_k.T @ h_phiphi @ z_k
    g_hat = task.a_meta + h_xphi @ r_k + z_k.T @ task.b_meta + z_k.T @ (h_phiphi @ r_k)
    return g_hat, q


def rerun_gradient(task: OracleTask, x_i: FloatArray, phi_k: FloatArray, z_k: FloatArray) -> FloatArray:
    """``grad F_i^{K,rerun}`` at the current operating point (analytic, via ``Z_K``).

    Independent of the hand-coded reverse-mode adjoint recursion in
    ``sgd.py``/``momentum.py``: this evaluates the closed-form §6 expression
    directly from ``Z_K`` rather than backward-accumulating over the
    trajectory.
    """

    grad_x = task.meta_gradient_x(x_i, phi_k)
    grad_phi = task.meta_gradient_phi(x_i, phi_k)
    return grad_x + z_k.T @ grad_phi


def commit_gradient(task: OracleTask, x_i: FloatArray, phi_k: FloatArray) -> FloatArray:
    """``grad F_i^{K,commit}`` at ``x_i' = x_i``: no private-compensation term."""

    return task.meta_gradient_x(x_i, phi_k)


def compensation_gap(task: OracleTask, x_i: FloatArray, phi_k: FloatArray, z_k: FloatArray) -> FloatArray:
    """``grad F_i^{K,rerun} - grad F_i^{K,commit} = Z_K^T grad_phi ell_i^meta``."""

    return z_k.T @ task.meta_gradient_phi(x_i, phi_k)


def exact_loss_change(gradient_at_point: FloatArray, q: FloatArray, local_step: FloatArray) -> float:
    """``Delta F_i^{K,rerun}(d) = grad^T (P_i d) + 1/2 (P_i d)^T Q (P_i d)``, zero truncation error.

    ``gradient_at_point`` must be the gradient at the *current* operating
    point (:func:`rerun_gradient`), not ``hat_g`` from :func:`quadratic_model`
    (which is anchored at ``x_i=0`` via ``r_K`` and differs from the point
    gradient by ``Q @ x_i``).
    """

    return float(gradient_at_point @ local_step + 0.5 * local_step @ (q @ local_step))
