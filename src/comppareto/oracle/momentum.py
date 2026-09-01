"""Exact K-step momentum private response: augmented state, sensitivity, Jacobians.

Matches ``docs/theory/oracle-spec.md`` section 5. Mirrors the three
independent code paths in ``sgd.py`` (literal unroll, analytic closed form
via matrix powers, independently implemented reverse-mode differentiation),
lifted to the augmented state ``u_k = (phi_k, v_k)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from comppareto.oracle.tasks import OracleTask

FloatArray = NDArray[np.float64]


def momentum_transition_jacobian(task: OracleTask, eta: float, beta: float) -> FloatArray:
    """``A_i`` block matrix: the per-step augmented-state Jacobian ``J_k`` (constant in k)."""

    d = task.private_dim
    c = task.private_curvature
    top = np.hstack([np.eye(d) - eta * c, -eta * beta * np.eye(d)])
    bottom = np.hstack([c, beta * np.eye(d)])
    return np.vstack([top, bottom])


def momentum_input_jacobian(task: OracleTask, eta: float) -> FloatArray:
    """``B_k``: the per-step augmented-state input Jacobian (constant in k)."""

    h_phix = task.h_phix
    return np.vstack([-eta * h_phix, h_phix])


def _affine_offset(task: OracleTask, x_i: FloatArray, eta: float) -> FloatArray:
    const = task.private_affine_term(x_i)
    return np.concatenate([-eta * const, const])


def momentum_unroll(
    task: OracleTask,
    x_i: FloatArray,
    phi_0: FloatArray,
    v_0: FloatArray,
    eta: float,
    beta: float,
    noise: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Literal step-by-step simulation; returns ``(phi_0..phi_K, v_0..v_K)``."""

    steps = noise.shape[0]
    d = task.private_dim
    phi_traj = np.empty((steps + 1, d))
    v_traj = np.empty((steps + 1, d))
    phi_traj[0] = phi_0
    v_traj[0] = v_0
    phi, v = phi_0, v_0
    for k in range(steps):
        grad = task.train_private_gradient(x_i, phi)
        v = beta * v + grad + noise[k]
        phi = phi - eta * v
        phi_traj[k + 1] = phi
        v_traj[k + 1] = v
    return phi_traj, v_traj


def momentum_closed_form_state(
    task: OracleTask,
    x_i: FloatArray,
    phi_0: FloatArray,
    v_0: FloatArray,
    eta: float,
    beta: float,
    noise: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Analytic closed-form ``(phi_K, v_K)`` via matrix powers, independent of the unroll.

    Accumulated by direct repeated matrix multiplication rather than a
    matrix-inverse geometric-series shortcut, since ``A_i`` need not be
    invertible or well-conditioned in the deliberately unstable regime.
    """

    steps = noise.shape[0]
    d = task.private_dim
    a = momentum_transition_jacobian(task, eta, beta)
    b = _affine_offset(task, x_i, eta)
    u0 = np.concatenate([phi_0, v_0])
    u = np.linalg.matrix_power(a, steps) @ u0
    for j in range(steps):
        power = steps - 1 - j
        a_power = np.linalg.matrix_power(a, power)
        n_j = np.concatenate([-eta * noise[j], noise[j]])
        u = u + a_power @ (b + n_j)
    return u[:d], u[d:]


def momentum_sensitivity(task: OracleTask, eta: float, beta: float, steps: int) -> FloatArray:
    """Analytic closed-form ``W_K = d(phi_K, v_K) / d x_i``, shape ``(2d, p_i)``.

    Top ``d`` rows are ``Z_K^phi``; bottom ``d`` rows are the momentum-buffer
    sensitivity. Computed as ``sum_{l=0}^{K-1} A_i^l @ B_k``, using explicit
    per-term matrix powers rather than an incremental recursion.
    """

    a = momentum_transition_jacobian(task, eta, beta)
    b = momentum_input_jacobian(task, eta)
    w = np.zeros((a.shape[0], b.shape[1]))
    for l in range(steps):
        w = w + np.linalg.matrix_power(a, l) @ b
    return w


def momentum_sensitivity_trajectory(task: OracleTask, eta: float, beta: float, steps: int) -> FloatArray:
    """The full sensitivity trajectory ``W_0..W_K``, shape ``(steps+1, 2d, p_i)``.

    Forward recursion ``W_0=0``, ``W_{k+1} = A_i W_k + B_k`` (``A_i`` from
    :func:`momentum_transition_jacobian`, ``B_k`` from
    :func:`momentum_input_jacobian`), independent of
    :func:`momentum_sensitivity`'s explicit-matrix-power sum: the two must
    agree at ``k=steps`` as a cross-check.
    """

    a = momentum_transition_jacobian(task, eta, beta)
    b = momentum_input_jacobian(task, eta)
    trajectory = np.zeros((steps + 1, 2 * task.private_dim, task.shared_dim))
    for k in range(steps):
        trajectory[k + 1] = a @ trajectory[k] + b
    return trajectory


def momentum_reverse_mode_gradient(
    task: OracleTask,
    x_i: FloatArray,
    phi_trajectory: FloatArray,
    eta: float,
    beta: float,
) -> FloatArray:
    """Hand-coded reverse-mode ``d F_i^{K,rerun} / d x_i`` over the stored trajectory.

    The meta loss depends only on ``phi_K``, not ``v_K``, so the seed adjoint
    on the augmented state has a zero momentum-buffer block.
    """

    steps = phi_trajectory.shape[0] - 1
    d = task.private_dim
    phi_k = phi_trajectory[-1]
    a = momentum_transition_jacobian(task, eta, beta)
    b = momentum_input_jacobian(task, eta)

    grad_x = task.meta_gradient_x(x_i, phi_k).copy()
    adjoint = np.concatenate([task.meta_gradient_phi(x_i, phi_k), np.zeros(d)])
    for _ in range(steps):
        grad_x = grad_x + b.T @ adjoint
        adjoint = a.T @ adjoint
    return grad_x


def momentum_commit_gradient(task: OracleTask, x_i: FloatArray, phi_k: FloatArray) -> FloatArray:
    """``d F_i^{K,commit} / d x_i'`` at ``x_i' = x_i``: no private-compensation term."""

    return task.meta_gradient_x(x_i, phi_k)
