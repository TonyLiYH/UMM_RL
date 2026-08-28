from __future__ import annotations

import numpy as np
import pytest

from comppareto.oracle.noise import NoiseModel
from comppareto.oracle.momentum import (
    momentum_closed_form_state,
    momentum_commit_gradient,
    momentum_input_jacobian,
    momentum_reverse_mode_gradient,
    momentum_sensitivity,
    momentum_sensitivity_trajectory,
    momentum_transition_jacobian,
    momentum_unroll,
)
from comppareto.oracle.hypergradient import exact_loss_change, quadratic_model, rerun_gradient
from _helpers import make_task

STATE_REL_TOL = 1e-10
HYPERGRAD_REL_TOL = 1e-9
LOSS_CHANGE_REL_TOL = 1e-9
FD_STEPS = (1e-2, 1e-4, 1e-6)
FD_REL_TOL = 1e-6


@pytest.mark.parametrize("p_i,d_i,steps,beta", [(3, 4, 5, 0.9), (5, 2, 1, 0.5), (2, 8, 10, 0.9)])
def test_momentum_closed_form_matches_unroll(
    rng: np.random.Generator, p_i: int, d_i: int, steps: int, beta: float
) -> None:
    task = make_task(rng, p_i, d_i)
    noise_model = NoiseModel(kind="gaussian", sigma=0.05)
    noise = noise_model.sample(rng, steps, d_i)
    x_i = rng.standard_normal(p_i)
    phi_0 = rng.standard_normal(d_i)
    v_0 = rng.standard_normal(d_i)
    eta = 0.05

    phi_traj, v_traj = momentum_unroll(task, x_i, phi_0, v_0, eta, beta, noise)
    phi_closed, v_closed = momentum_closed_form_state(task, x_i, phi_0, v_0, eta, beta, noise)

    phi_rel_err = np.linalg.norm(phi_traj[-1] - phi_closed) / np.linalg.norm(phi_traj[-1])
    v_rel_err = np.linalg.norm(v_traj[-1] - v_closed) / np.linalg.norm(v_traj[-1])
    assert phi_rel_err <= STATE_REL_TOL
    assert v_rel_err <= STATE_REL_TOL


def test_momentum_hypergradient_analytic_matches_reverse_mode(rng: np.random.Generator) -> None:
    task = make_task(rng, 4, 6)
    noise_model = NoiseModel(kind="block_correlated", sigma=0.1, rho=0.4, sub_block_width=2)
    steps = 6
    noise = noise_model.sample(rng, steps, 6)
    eta, beta = 0.05, 0.9
    x_i = rng.standard_normal(4)
    phi_0 = rng.standard_normal(6)
    v_0 = rng.standard_normal(6)

    phi_traj, _ = momentum_unroll(task, x_i, phi_0, v_0, eta, beta, noise)
    w_k = momentum_sensitivity(task, eta, beta, steps)
    z_k_phi = w_k[: task.private_dim]
    analytic = rerun_gradient(task, x_i, phi_traj[-1], z_k_phi)
    reverse_mode = momentum_reverse_mode_gradient(task, x_i, phi_traj, eta, beta)

    rel_err = np.linalg.norm(reverse_mode - analytic) / np.linalg.norm(analytic)
    assert rel_err <= HYPERGRAD_REL_TOL


@pytest.mark.parametrize("h", FD_STEPS)
def test_momentum_finite_difference_matches_analytic_directional_derivative(
    rng: np.random.Generator, h: float
) -> None:
    task = make_task(rng, 3, 5)
    noise_model = NoiseModel(kind="gaussian", sigma=0.05)
    steps = 4
    noise = noise_model.sample(rng, steps, 5)
    eta, beta = 0.05, 0.9
    x_i = rng.standard_normal(3)
    phi_0 = rng.standard_normal(5)
    v_0 = rng.standard_normal(5)
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)

    def loss(xi: np.ndarray) -> float:
        phi_k, _ = momentum_closed_form_state(task, xi, phi_0, v_0, eta, beta, noise)
        return task.meta_loss(xi, phi_k)

    d = task.private_dim
    w_k = momentum_sensitivity(task, eta, beta, steps)
    z_k_phi = w_k[:d]
    phi_k, _ = momentum_closed_form_state(task, x_i, phi_0, v_0, eta, beta, noise)
    analytic_directional = float(rerun_gradient(task, x_i, phi_k, z_k_phi) @ direction)
    fd_directional = (loss(x_i + h * direction) - loss(x_i - h * direction)) / (2 * h)

    rel_err = abs(fd_directional - analytic_directional) / abs(analytic_directional)
    assert rel_err <= FD_REL_TOL


def test_momentum_exact_quadratic_loss_change_identity(rng: np.random.Generator) -> None:
    task = make_task(rng, 4, 6)
    noise_model = NoiseModel(kind="gaussian", sigma=0.05)
    steps = 5
    noise = noise_model.sample(rng, steps, 6)
    eta, beta = 0.05, 0.9
    x_i = rng.standard_normal(4)
    phi_0 = rng.standard_normal(6)
    v_0 = rng.standard_normal(6)
    step = 0.02 * rng.standard_normal(4)
    d = task.private_dim

    def loss(xi: np.ndarray) -> float:
        phi_k, _ = momentum_closed_form_state(task, xi, phi_0, v_0, eta, beta, noise)
        return task.meta_loss(xi, phi_k)

    w_k = momentum_sensitivity(task, eta, beta, steps)
    z_k_phi = w_k[:d]
    r_k_phi, _ = momentum_closed_form_state(task, np.zeros(4), phi_0, v_0, eta, beta, noise)
    phi_k, _ = momentum_closed_form_state(task, x_i, phi_0, v_0, eta, beta, noise)
    g_hat, q = quadratic_model(task, z_k_phi, r_k_phi)
    grad_at_point = rerun_gradient(task, x_i, phi_k, z_k_phi)

    exact_delta = exact_loss_change(grad_at_point, q, step)
    direct_delta = loss(x_i + step) - loss(x_i)

    rel_err = abs(exact_delta - direct_delta) / abs(direct_delta)
    assert rel_err <= LOSS_CHANGE_REL_TOL


def test_momentum_rerun_commit_decomposition_matches_compensation_gap(rng: np.random.Generator) -> None:
    task = make_task(rng, 3, 4)
    noise_model = NoiseModel(kind="gaussian", sigma=0.05)
    steps = 5
    noise = noise_model.sample(rng, steps, 4)
    eta, beta = 0.05, 0.9
    x_i = rng.standard_normal(3)
    phi_0 = rng.standard_normal(4)
    v_0 = rng.standard_normal(4)
    d = task.private_dim

    w_k = momentum_sensitivity(task, eta, beta, steps)
    z_k_phi = w_k[:d]
    phi_k, _ = momentum_closed_form_state(task, x_i, phi_0, v_0, eta, beta, noise)
    rerun_grad = rerun_gradient(task, x_i, phi_k, z_k_phi)
    commit_grad = momentum_commit_gradient(task, x_i, phi_k)
    gap = z_k_phi.T @ task.meta_gradient_phi(x_i, phi_k)

    rel_err = np.linalg.norm((rerun_grad - commit_grad) - gap) / np.linalg.norm(gap)
    assert rel_err <= 1e-10


@pytest.mark.parametrize("beta", [0.5, 0.9])
def test_momentum_transition_jacobian_stable_contraction(rng: np.random.Generator, beta: float) -> None:
    task = make_task(rng, 2, 3)
    lam_max = np.max(np.linalg.eigvalsh(task.private_curvature))
    eta = 0.5 / lam_max
    from scipy.optimize import minimize_scalar

    def rho(e: float) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(momentum_transition_jacobian(task, e, beta)))))

    result = minimize_scalar(rho, bounds=(1e-8, 2.0 / lam_max), method="bounded")
    assert result.fun < 1.0


@pytest.mark.parametrize("p_i,d_i,steps,beta", [(3, 4, 5, 0.9), (5, 2, 1, 0.5), (2, 8, 10, 0.9)])
def test_momentum_sensitivity_trajectory_final_step_matches_closed_form(
    rng: np.random.Generator, p_i: int, d_i: int, steps: int, beta: float
) -> None:
    task = make_task(rng, p_i, d_i)
    eta = 0.05

    trajectory = momentum_sensitivity_trajectory(task, eta, beta, steps)
    closed_form = momentum_sensitivity(task, eta, beta, steps)

    assert trajectory.shape == (steps + 1, 2 * d_i, p_i)
    assert np.allclose(trajectory[0], np.zeros((2 * d_i, p_i)))
    rel_err = np.linalg.norm(trajectory[-1] - closed_form) / np.linalg.norm(closed_form)
    assert rel_err <= STATE_REL_TOL


def test_momentum_sensitivity_trajectory_obeys_one_step_recursion(rng: np.random.Generator) -> None:
    task = make_task(rng, 4, 6)
    eta, beta = 0.05, 0.9
    a = momentum_transition_jacobian(task, eta, beta)
    b = momentum_input_jacobian(task, eta)

    trajectory = momentum_sensitivity_trajectory(task, eta, beta, 6)
    for k in range(6):
        rel_err = np.linalg.norm(trajectory[k + 1] - (a @ trajectory[k] + b)) / max(
            np.linalg.norm(trajectory[k + 1]), 1e-30
        )
        assert rel_err <= 1e-12
