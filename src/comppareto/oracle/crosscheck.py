"""Numerical cross-checks against the §9 pass/fail-gate tolerances.

Every comparison routes through :func:`compare`, which implements the
near-zero handling rule stated in ``docs/theory/oracle-spec.md`` sections
8-9: absolute tolerance ``1e-11`` when the reference norm is ``<=1e-10``,
relative tolerance otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from comppareto.oracle import hypergradient as hg
from comppareto.oracle import momentum as mom
from comppareto.oracle import sgd
from comppareto.oracle.tasks import OracleTask

FloatArray = NDArray[np.float64]

NEAR_ZERO_NORM = 1e-10
NEAR_ZERO_ABS_TOL = 1e-11
FD_STEPS: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
FD_REL_TOL = 1e-6


@dataclass(frozen=True)
class CheckResult:
    name: str
    error: float
    tolerance: float
    mode: str  # "relative" | "absolute"
    passed: bool


def compare(name: str, reference: FloatArray | float, value: FloatArray | float, rel_tol: float) -> CheckResult:
    ref = np.atleast_1d(np.asarray(reference, dtype=np.float64))
    val = np.atleast_1d(np.asarray(value, dtype=np.float64))
    diff_norm = float(np.linalg.norm(val - ref))
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm <= NEAR_ZERO_NORM:
        return CheckResult(name, diff_norm, NEAR_ZERO_ABS_TOL, "absolute", diff_norm <= NEAR_ZERO_ABS_TOL)
    error = diff_norm / ref_norm
    return CheckResult(name, error, rel_tol, "relative", error <= rel_tol)


@dataclass(frozen=True)
class CaseChecks:
    state: CheckResult
    hypergradient: CheckResult
    loss_change: CheckResult
    finite_difference: tuple[CheckResult, ...]

    @property
    def all_passed(self) -> bool:
        gate = (self.state.passed, self.hypergradient.passed, self.loss_change.passed)
        return all(gate) and all(c.passed for c in self.finite_difference if c.mode == "absolute" or c.error <= FD_REL_TOL)


def _fd_directional(loss_fn, x_i: FloatArray, direction: FloatArray, h: float) -> float:
    return float((loss_fn(x_i + h * direction) - loss_fn(x_i - h * direction)) / (2.0 * h))


def check_sgd_case(
    task: OracleTask,
    x_i: FloatArray,
    phi_0: FloatArray,
    eta: float,
    noise: FloatArray,
    probe_direction: FloatArray,
    step: FloatArray,
) -> CaseChecks:
    steps = noise.shape[0]

    trajectory = sgd.sgd_unroll(task, x_i, phi_0, eta, noise)
    phi_k_unroll = trajectory[-1]
    phi_k_closed = sgd.sgd_closed_form_state(task, x_i, phi_0, eta, noise)
    state_check = compare("sgd_state", phi_k_unroll, phi_k_closed, 1e-10)

    z_k = sgd.sgd_sensitivity(task, eta, steps)
    r_k = sgd.sgd_closed_form_state(task, np.zeros_like(x_i), phi_0, eta, noise)

    analytic_grad = hg.rerun_gradient(task, x_i, phi_k_closed, z_k)
    reverse_mode_grad = sgd.sgd_reverse_mode_gradient(task, x_i, trajectory, eta)
    hypergrad_check = compare("sgd_hypergradient", reverse_mode_grad, analytic_grad, 1e-9)

    _, q = hg.quadratic_model(task, z_k, r_k)
    direct_loss = lambda xi: task.meta_loss(  # noqa: E731
        xi, sgd.sgd_closed_form_state(task, xi, phi_0, eta, noise)
    )
    base_loss = direct_loss(x_i)
    direct_delta = direct_loss(x_i + step) - base_loss
    # Spec §6: Delta F(d) = (grad F at x_i)^T d + 1/2 d^T Q d, i.e. the gradient
    # at the *current* point, not the quadratic model's global coefficient g_hat
    # (which is anchored at x_i=0 via r_K and differs from the point gradient by Q@x_i).
    exact_delta = hg.exact_loss_change(analytic_grad, q, step)
    loss_change_check = compare("sgd_loss_change", exact_delta, direct_delta, 1e-9)

    fd_checks = []
    analytic_directional = float(analytic_grad @ probe_direction)
    for h in FD_STEPS:
        fd_value = _fd_directional(direct_loss, x_i, probe_direction, h)
        fd_checks.append(compare(f"sgd_fd_h={h:g}", analytic_directional, fd_value, FD_REL_TOL))

    return CaseChecks(state_check, hypergrad_check, loss_change_check, tuple(fd_checks))


def check_momentum_case(
    task: OracleTask,
    x_i: FloatArray,
    phi_0: FloatArray,
    v_0: FloatArray,
    eta: float,
    beta: float,
    noise: FloatArray,
    probe_direction: FloatArray,
    step: FloatArray,
) -> CaseChecks:
    steps = noise.shape[0]

    phi_traj, v_traj = mom.momentum_unroll(task, x_i, phi_0, v_0, eta, beta, noise)
    phi_k_unroll = phi_traj[-1]
    phi_k_closed, v_k_closed = mom.momentum_closed_form_state(task, x_i, phi_0, v_0, eta, beta, noise)
    state_err_phi = compare("momentum_state_phi", phi_k_unroll, phi_k_closed, 1e-10)
    state_err_v = compare("momentum_state_v", v_traj[-1], v_k_closed, 1e-10)
    state_check = state_err_phi if state_err_phi.error >= state_err_v.error else state_err_v

    w_k = mom.momentum_sensitivity(task, eta, beta, steps)
    d = task.private_dim
    z_k_phi = w_k[:d]
    zero_x = np.zeros_like(x_i)
    r_k_phi, _ = mom.momentum_closed_form_state(task, zero_x, phi_0, v_0, eta, beta, noise)

    analytic_grad = hg.rerun_gradient(task, x_i, phi_k_closed, z_k_phi)
    reverse_mode_grad = mom.momentum_reverse_mode_gradient(task, x_i, phi_traj, eta, beta)
    hypergrad_check = compare("momentum_hypergradient", reverse_mode_grad, analytic_grad, 1e-9)

    _, q = hg.quadratic_model(task, z_k_phi, r_k_phi)

    def direct_loss(xi: FloatArray) -> float:
        phi_k, _ = mom.momentum_closed_form_state(task, xi, phi_0, v_0, eta, beta, noise)
        return task.meta_loss(xi, phi_k)

    base_loss = direct_loss(x_i)
    direct_delta = direct_loss(x_i + step) - base_loss
    exact_delta = hg.exact_loss_change(analytic_grad, q, step)
    loss_change_check = compare("momentum_loss_change", exact_delta, direct_delta, 1e-9)

    fd_checks = []
    analytic_directional = float(analytic_grad @ probe_direction)
    for h in FD_STEPS:
        fd_value = _fd_directional(direct_loss, x_i, probe_direction, h)
        fd_checks.append(compare(f"momentum_fd_h={h:g}", analytic_directional, fd_value, FD_REL_TOL))

    return CaseChecks(state_check, hypergrad_check, loss_change_check, tuple(fd_checks))
