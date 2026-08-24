from __future__ import annotations

import numpy as np
import pytest

from comppareto.quadratic import (
    CurvatureError,
    QuadraticTask,
    common_descent_two,
    negotiate_retained_gain,
    retained_gain,
    trust_region_optimum,
)


def make_task() -> QuadraticTask:
    return QuadraticTask(
        local_gradient=np.array([0.4, -0.2]),
        h_xx=np.array([[3.0, 0.4], [0.4, 2.0]]),
        h_xphi=np.array([[0.5], [-0.3]]),
        h_phiphi=np.array([[1.2]]),
        mu=0.8,
        selector=np.eye(2),
    )


def test_schur_change_matches_direct_private_minimum() -> None:
    task = make_task()
    step = np.array([0.1, -0.2])

    private_step = task.private_response(step)
    direct = task.direct_change(step, private_step)

    assert task.compensated_change(step) == pytest.approx(direct, abs=1e-12)
    assert direct <= task.direct_change(step, np.zeros(1))


def test_block_selector_lifts_local_gradient_once() -> None:
    selector = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    task = QuadraticTask(
        local_gradient=np.array([2.0, -3.0]),
        h_xx=np.eye(2),
        h_xphi=np.zeros((2, 1)),
        h_phiphi=np.eye(1),
        mu=0.5,
        selector=selector,
    )

    np.testing.assert_allclose(task.lifted_gradient(), np.array([2.0, 0.0, -3.0]))


@pytest.mark.parametrize(
    "selector",
    [
        np.array([[0.5, 0.5]]),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 1.0]]),
    ],
)
def test_selector_rejects_non_selection_matrices(selector: np.ndarray) -> None:
    with pytest.raises(ValueError, match="selector"):
        QuadraticTask(
            local_gradient=np.ones(selector.shape[0]),
            h_xx=np.eye(selector.shape[0]),
            h_xphi=np.zeros((selector.shape[0], 1)),
            h_phiphi=np.eye(1),
            mu=0.5,
            selector=selector,
        )


def test_common_descent_two_gives_descent_for_both_gradients() -> None:
    first = np.array([1.0, 0.0])
    second = np.array([0.0, 1.0])

    result = common_descent_two(first, second, metric=np.eye(2))

    assert not result.stationary
    assert first @ result.direction < 0
    assert second @ result.direction < 0
    assert first @ result.direction == pytest.approx(-result.margin)
    assert second @ result.direction == pytest.approx(-result.margin)


def test_common_descent_two_detects_pareto_stationary_segment() -> None:
    result = common_descent_two(
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        metric=np.eye(2),
    )

    assert result.stationary
    np.testing.assert_allclose(result.direction, np.zeros(2), atol=1e-12)


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_conditional_loss_rescaling_preserves_schur_and_retained_gain(scale: float) -> None:
    task = make_task()
    scaled = task.scaled(scale)
    step = np.array([0.1, -0.2])

    np.testing.assert_allclose(scaled.schur(), scale * task.schur(), rtol=1e-12)
    change = task.compensated_change(step)
    attainable = 0.7
    epsilon = 1e-6
    assert retained_gain(scale * change, scale * attainable, scale * epsilon) == pytest.approx(
        retained_gain(change, attainable, epsilon)
    )


def test_indefinite_regularized_private_curvature_is_rejected() -> None:
    with pytest.raises(CurvatureError, match="positive definite"):
        QuadraticTask(
            local_gradient=np.array([1.0]),
            h_xx=np.eye(1),
            h_xphi=np.ones((1, 1)),
            h_phiphi=np.array([[-2.0]]),
            mu=0.5,
            selector=np.eye(1),
        )


def test_trust_region_optimum_solves_active_boundary() -> None:
    result = trust_region_optimum(
        gradient=np.array([-2.0, 0.0]),
        hessian=np.eye(2),
        metric=np.eye(2),
        radius=0.5,
    )

    np.testing.assert_allclose(result.step, np.array([0.5, 0.0]), atol=1e-9)
    assert result.attainable_gain == pytest.approx(0.875, abs=1e-9)
    assert result.boundary_active


def test_retained_gain_negotiation_is_invariant_to_independent_task_scaling() -> None:
    selector = np.eye(2)
    first = QuadraticTask(
        local_gradient=np.array([-2.0, 0.0]),
        h_xx=np.eye(2),
        h_xphi=np.zeros((2, 1)),
        h_phiphi=np.eye(1),
        mu=0.5,
        selector=selector,
    )
    second = QuadraticTask(
        local_gradient=np.array([0.0, -1.0]),
        h_xx=np.eye(2),
        h_xphi=np.zeros((2, 1)),
        h_phiphi=np.eye(1),
        mu=0.5,
        selector=selector,
    )

    base = negotiate_retained_gain(
        [first, second], metric=np.eye(2), radius=0.5, epsilons=[1e-8, 2e-8]
    )
    scaled = negotiate_retained_gain(
        [first.scaled(1e3), second.scaled(1e-3)],
        metric=np.eye(2),
        radius=0.5,
        epsilons=[1e-5, 2e-11],
    )

    np.testing.assert_allclose(scaled.step, base.step, atol=1e-7)
    assert scaled.tau == pytest.approx(base.tau, abs=1e-7)
