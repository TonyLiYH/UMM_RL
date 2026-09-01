from __future__ import annotations

import numpy as np
import pytest

from comppareto.oracle.noise import NoiseModel
from comppareto.oracle.sgd import (
    sgd_closed_form_state,
    sgd_commit_gradient,
    sgd_input_jacobian,
    sgd_reverse_mode_gradient,
    sgd_sensitivity,
    sgd_sensitivity_trajectory,
    sgd_state_jacobian,
    sgd_unroll,
)
from comppareto.oracle.hypergradient import exact_loss_change, quadratic_model, rerun_gradient
from _helpers import make_task

STATE_REL_TOL = 1e-10
HYPERGRAD_REL_TOL = 1e-9
LOSS_CHANGE_REL_TOL = 1e-9
FD_STEPS = (1e-2, 1e-4, 1e-6)
FD_REL_TOL = 1e-6


@pytest.mark.parametrize("p_i,d_i,steps", [(3, 4, 5), (5, 2, 1), (2, 8, 10)])
def test_sgd_closed_form_matches_unroll(rng: np.random.Generator, p_i: int, d_i: int, steps: int) -> None:
    task = make_task(rng, p_i, d_i)
    noise_model = NoiseModel(kind="gaussian", sigma=0.05)
    noise = noise_model.sample(rng, steps, d_i)
    x_i = rng.standard_normal(p_i)
    phi_0 = rng.standard_normal(d_i)

    trajectory = sgd_unroll(task, x_i, phi_0, 0.05, noise)
    closed = sgd_closed_form_state(task, x_i, phi_0, 0.05, noise)

    rel_err = np.linalg.norm(trajectory[-1] - closed) / np.linalg.norm(trajectory[-1])
    assert rel_err <= STATE_REL_TOL


def test_sgd_hypergradient_analytic_matches_reverse_mode(rng: np.random.Generator) -> None:
    task = make_task(rng, 4, 6)
    noise_model = NoiseModel(kind="block_correlated", sigma=0.1, rho=0.4, sub_block_width=2)
    steps = 6
    noise = noise_model.sample(rng, steps, 6)
    eta = 0.05
    x_i = rng.standard_normal(4)
    phi_0 = rng.standard_normal(6)

    trajectory = sgd_unroll(task, x_i, phi_0, eta, noise)
    z_k = sgd_sensitivity(task, eta, steps)
    analytic = rerun_gradient(task, x_i, trajectory[-1], z_k)
    reverse_mode = sgd_reverse_mode_gradient(task, x_i, trajectory, eta)

    rel_err = np.linalg.norm(reverse_mode - analytic) / np.linalg.norm(analytic)
    assert rel_err <= HYPERGRAD_REL_TOL


@pytest.mark.parametrize("h", FD_STEPS)
def test_sgd_finite_difference_matches_analytic_directional_derivative(
    rng: np.random.Generator, h: float
) -> None:
    task = make_task(rng, 3, 5)
    noise_model = NoiseModel(kind="gaussian", sigma=0.05)
    steps = 4
    noise = noise_model.sample(rng, steps, 5)
    eta = 0.05
    x_i = rng.standard_normal(3)
    phi_0 = rng.standard_normal(5)
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)

    def loss(xi: np.ndarray) -> float:
        phi_k = sgd_closed_form_state(task, xi, phi_0, eta, noise)
        return task.meta_loss(xi, phi_k)

    z_k = sgd_sensitivity(task, eta, steps)
    phi_k = sgd_closed_form_state(task, x_i, phi_0, eta, noise)
    analytic_directional = float(rerun_gradient(task, x_i, phi_k, z_k) @ direction)
    fd_directional = (loss(x_i + h * direction) - loss(x_i - h * direction)) / (2 * h)

    rel_err = abs(fd_directional - analytic_directional) / abs(analytic_directional)
    assert rel_err <= FD_REL_TOL


def test_sgd_exact_quadratic_loss_change_identity(rng: np.random.Generator) -> None:
    task = make_task(rng, 4, 6)
    noise_model = NoiseModel(kind="gaussian", sigma=0.05)
    steps = 5
    noise = noise_model.sample(rng, steps, 6)
    eta = 0.05
    x_i = rng.standard_normal(4)
    phi_0 = rng.standard_normal(6)
    step = 0.02 * rng.standard_normal(4)

    def loss(xi: np.ndarray) -> float:
        phi_k = sgd_closed_form_state(task, xi, phi_0, eta, noise)
        return task.meta_loss(xi, phi_k)

    z_k = sgd_sensitivity(task, eta, steps)
    r_k = sgd_closed_form_state(task, np.zeros(4), phi_0, eta, noise)
    phi_k = sgd_closed_form_state(task, x_i, phi_0, eta, noise)
    g_hat, q = quadratic_model(task, z_k, r_k)
    grad_at_point = rerun_gradient(task, x_i, phi_k, z_k)

    exact_delta = exact_loss_change(grad_at_point, q, step)
    direct_delta = loss(x_i + step) - loss(x_i)

    rel_err = abs(exact_delta - direct_delta) / abs(direct_delta)
    assert rel_err <= LOSS_CHANGE_REL_TOL


def test_sgd_rerun_commit_decomposition_matches_compensation_gap(rng: np.random.Generator) -> None:
    task = make_task(rng, 3, 4)
    noise_model = NoiseModel(kind="gaussian", sigma=0.05)
    steps = 5
    noise = noise_model.sample(rng, steps, 4)
    eta = 0.05
    x_i = rng.standard_normal(3)
    phi_0 = rng.standard_normal(4)

    z_k = sgd_sensitivity(task, eta, steps)
    phi_k = sgd_closed_form_state(task, x_i, phi_0, eta, noise)
    rerun_grad = rerun_gradient(task, x_i, phi_k, z_k)
    commit_grad = sgd_commit_gradient(task, x_i, phi_k)
    gap = z_k.T @ task.meta_gradient_phi(x_i, phi_k)

    rel_err = np.linalg.norm((rerun_grad - commit_grad) - gap) / np.linalg.norm(gap)
    assert rel_err <= 1e-10


def test_sgd_state_jacobian_matches_stable_contraction(rng: np.random.Generator) -> None:
    task = make_task(rng, 2, 3)
    eta = 0.5 / np.max(np.linalg.eigvalsh(task.private_curvature))
    m = sgd_state_jacobian(task, eta)
    assert np.max(np.abs(np.linalg.eigvals(m))) < 1.0


@pytest.mark.parametrize("p_i,d_i,steps", [(3, 4, 5), (5, 2, 1), (2, 8, 10)])
def test_sgd_sensitivity_trajectory_final_step_matches_closed_form(
    rng: np.random.Generator, p_i: int, d_i: int, steps: int
) -> None:
    task = make_task(rng, p_i, d_i)
    eta = 0.05

    trajectory = sgd_sensitivity_trajectory(task, eta, steps)
    closed_form = sgd_sensitivity(task, eta, steps)

    assert trajectory.shape == (steps + 1, d_i, p_i)
    assert np.allclose(trajectory[0], np.zeros((d_i, p_i)))
    rel_err = np.linalg.norm(trajectory[-1] - closed_form) / np.linalg.norm(closed_form)
    assert rel_err <= STATE_REL_TOL


def test_sgd_sensitivity_trajectory_obeys_one_step_recursion(rng: np.random.Generator) -> None:
    task = make_task(rng, 4, 6)
    eta = 0.05
    m = sgd_state_jacobian(task, eta)
    b = sgd_input_jacobian(task, eta)

    trajectory = sgd_sensitivity_trajectory(task, eta, 6)
    for k in range(6):
        rel_err = np.linalg.norm(trajectory[k + 1] - (m @ trajectory[k] + b)) / max(
            np.linalg.norm(trajectory[k + 1]), 1e-30
        )
        assert rel_err <= 1e-12
