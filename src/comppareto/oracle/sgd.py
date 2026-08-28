"""Exact K-step SGD private response: state, sensitivity, and Jacobians.

Matches ``docs/theory/oracle-spec.md`` section 4. Three independent code
paths are provided so cross-checks are meaningful:

- :func:`sgd_unroll` is the literal step-by-step simulation (the
  "independently unrolled state" reference);
- :func:`sgd_closed_form_state` / :func:`sgd_sensitivity` are the analytic
  closed forms, built from matrix powers and a geometric-series identity,
  never from the step-by-step loop;
- :func:`sgd_reverse_mode_gradient` is an independently implemented
  reverse-mode differentiation over the literal unroll's stored trajectory
  (the hand-coded reverse-mode reference), independent of the forward
  sensitivity recursion above.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from comppareto.oracle.tasks import OracleTask

FloatArray = NDArray[np.float64]


def sgd_state_jacobian(task: OracleTask, eta: float) -> FloatArray:
    """``M_i = I - eta_i * C_i``: the per-step state Jacobian ``J_k`` (constant in k)."""

    return np.eye(task.private_dim) - eta * task.private_curvature


def sgd_input_jacobian(task: OracleTask, eta: float) -> FloatArray:
    """``B_k = -eta_i * H_phix``: the per-step input Jacobian (constant in k)."""

    return -eta * task.h_phix


def sgd_unroll(
    task: OracleTask, x_i: FloatArray, phi_0: FloatArray, eta: float, noise: FloatArray
) -> FloatArray:
    """Literal step-by-step simulation; returns the trajectory ``phi_0..phi_K``."""

    steps = noise.shape[0]
    trajectory = np.empty((steps + 1, task.private_dim))
    trajectory[0] = phi_0
    phi = phi_0
    for k in range(steps):
        grad = task.train_private_gradient(x_i, phi)
        phi = phi - eta * (grad + noise[k])
        trajectory[k + 1] = phi
    return trajectory


def sgd_closed_form_state(
    task: OracleTask, x_i: FloatArray, phi_0: FloatArray, eta: float, noise: FloatArray
) -> FloatArray:
    """Analytic closed-form ``phi_K``, via matrix powers, independent of :func:`sgd_unroll`."""

    steps = noise.shape[0]
    d = task.private_dim
    m = sgd_state_jacobian(task, eta)
    c = task.private_curvature
    const = task.private_affine_term(x_i)
    m_k = np.linalg.matrix_power(m, steps)

    # sum_{l=0}^{K-1} M^l = (I - M^K)(I - M)^{-1} = (I - M^K)(eta*C)^{-1}; the eta
    # cancels against the recursion's leading -eta, leaving -(I-M^K) C^{-1} const.
    deterministic = m_k @ phi_0 - np.linalg.solve(c, (np.eye(d) - m_k) @ const)

    noise_term = np.zeros(d)
    for j in range(steps):
        power = steps - 1 - j
        noise_term += np.linalg.matrix_power(m, power) @ noise[j]
    return deterministic - eta * noise_term


def sgd_sensitivity(task: OracleTask, eta: float, steps: int) -> FloatArray:
    """Analytic closed-form ``Z_K = d phi_K / d x_i``."""

    d = task.private_dim
    m = sgd_state_jacobian(task, eta)
    c = task.private_curvature
    m_k = np.linalg.matrix_power(m, steps)
    return -np.linalg.solve(c, (np.eye(d) - m_k)) @ task.h_phix


def sgd_sensitivity_trajectory(task: OracleTask, eta: float, steps: int) -> FloatArray:
    """The full sensitivity trajectory ``Z_0..Z_K``, shape ``(steps+1, d, p_i)``.

    Forward recursion ``Z_0=0``, ``Z_{k+1} = M_i Z_k + B_k`` (``M_i`` from
    :func:`sgd_state_jacobian`, ``B_k`` from :func:`sgd_input_jacobian`),
    independent of :func:`sgd_sensitivity`'s closed-form geometric-series
    shortcut: the two must agree at ``k=steps`` as a cross-check.
    """

    m = sgd_state_jacobian(task, eta)
    b = sgd_input_jacobian(task, eta)
    trajectory = np.zeros((steps + 1, task.private_dim, task.shared_dim))
    for k in range(steps):
        trajectory[k + 1] = m @ trajectory[k] + b
    return trajectory


def sgd_reverse_mode_gradient(
    task: OracleTask, x_i: FloatArray, trajectory: FloatArray, eta: float
) -> FloatArray:
    """Hand-coded reverse-mode ``d F_i^{K,rerun} / d x_i`` over the stored trajectory.

    Independent of :func:`sgd_sensitivity`: forward-store (via
    :func:`sgd_unroll`) then backward-accumulate, rather than a forward
    sensitivity recursion.
    """

    steps = trajectory.shape[0] - 1
    phi_k = trajectory[-1]
    m = sgd_state_jacobian(task, eta)
    b = sgd_input_jacobian(task, eta)

    grad_x = task.meta_gradient_x(x_i, phi_k).copy()
    adjoint = task.meta_gradient_phi(x_i, phi_k)
    for _ in range(steps):
        grad_x = grad_x + b.T @ adjoint
        adjoint = m.T @ adjoint
    return grad_x


def sgd_commit_gradient(task: OracleTask, x_i: FloatArray, phi_k: FloatArray) -> FloatArray:
    """``d F_i^{K,commit} / d x_i'`` at ``x_i' = x_i``: no private-compensation term."""

    return task.meta_gradient_x(x_i, phi_k)
