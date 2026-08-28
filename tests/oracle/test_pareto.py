from __future__ import annotations

import numpy as np
import pytest

from comppareto.oracle.pareto import (
    case_pareto_reference,
    lift_gradient,
    min_norm_point_active_set,
    min_norm_point_frank_wolfe,
)
from comppareto.oracle.selectors import BlockLayout, build_incidence, build_task_selectors

KKT_TOL = 1e-8
FW_KKT_TOL = 1e-6
CROSS_CHECK_REL_TOL = 1e-4


def test_lift_gradient_is_selector_transpose(rng: np.random.Generator) -> None:
    selector = np.eye(3, 5)
    local_grad = rng.standard_normal(3)
    lifted = lift_gradient(selector, local_grad)
    assert np.allclose(lifted, selector.T @ local_grad)


def test_min_norm_point_stationary_when_origin_in_hull() -> None:
    angles = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    gradients = [np.array([np.cos(a), np.sin(a)]) for a in angles]

    exact = min_norm_point_active_set(gradients)
    iterative = min_norm_point_frank_wolfe(gradients)

    assert exact.stationary
    assert exact.objective <= 1e-10
    assert np.allclose(exact.combined_gradient, 0.0, atol=1e-8)
    assert iterative.objective <= 1e-8


def test_min_norm_point_single_gradient_is_itself() -> None:
    g = np.array([2.0, -1.0, 0.5])
    exact = min_norm_point_active_set([g])
    assert np.allclose(exact.combined_gradient, g)
    assert exact.weights.tolist() == [1.0]
    assert not exact.stationary


@pytest.mark.parametrize("num_tasks", [2, 3, 4])
def test_disjoint_family_matches_orthogonal_closed_form(rng: np.random.Generator, num_tasks: int) -> None:
    # Disjoint task selectors touch non-overlapping blocks, so the lifted
    # gradients are pairwise orthogonal; the min-norm point in their convex
    # hull then has the closed form lambda_i ~ 1/||g_i||^2 (all weights
    # strictly positive, so the KKT system reduces to this single formula).
    layout = BlockLayout(tuple([3] * 9))
    inc = build_incidence("disjoint", num_tasks, layout.num_blocks, rng)
    selectors = build_task_selectors(layout, inc)
    local_grads = [rng.standard_normal(sel.shape[0]) + 0.5 for sel in selectors]
    lifted = [lift_gradient(p, g) for p, g in zip(selectors, local_grads)]

    norms_sq = np.array([float(v @ v) for v in lifted])
    expected_lambda = (1.0 / norms_sq) / np.sum(1.0 / norms_sq)
    expected_objective = 1.0 / np.sum(1.0 / norms_sq)

    exact = min_norm_point_active_set(lifted)

    assert np.allclose(exact.weights, expected_lambda, atol=1e-8)
    assert abs(exact.objective - expected_objective) <= 1e-8
    assert exact.kkt_residual <= KKT_TOL


def test_full_overlap_two_task_matches_segment_projection(rng: np.random.Generator) -> None:
    # With full overlap and two tasks the lifted gradients live in the same
    # subspace; the min-norm point of a two-point convex hull is the
    # closed-form projection of the origin onto the segment [g0, g1].
    layout = BlockLayout(tuple([3] * 6))
    inc = build_incidence("full_overlap", 2, layout.num_blocks, rng)
    selectors = build_task_selectors(layout, inc)
    g0 = rng.standard_normal(selectors[0].shape[0])
    g1 = rng.standard_normal(selectors[0].shape[0])
    lifted = [lift_gradient(selectors[0], g0), lift_gradient(selectors[1], g1)]

    diff = lifted[1] - lifted[0]
    t = float(np.clip(-lifted[0] @ diff / (diff @ diff), 0.0, 1.0))
    expected = lifted[0] + t * diff

    exact = min_norm_point_active_set(lifted)

    assert np.linalg.norm(exact.combined_gradient - expected) <= 1e-8
    assert exact.kkt_residual <= KKT_TOL


@pytest.mark.parametrize(
    "family,num_tasks",
    [
        ("disjoint", 3),
        ("partial", 4),
        ("full_overlap", 3),
        ("star", 4),
        ("chain", 5),
        ("random_sparse", 4),
    ],
)
def test_active_set_and_frank_wolfe_agree_across_families(
    rng: np.random.Generator, family: str, num_tasks: int
) -> None:
    layout = BlockLayout(tuple([3] * 9))
    inc = build_incidence(family, num_tasks, layout.num_blocks, rng)
    selectors = build_task_selectors(layout, inc)
    local_grads = [rng.standard_normal(sel.shape[0]) for sel in selectors]
    lifted = [lift_gradient(p, g) for p, g in zip(selectors, local_grads)]

    exact = min_norm_point_active_set(lifted)
    iterative = min_norm_point_frank_wolfe(lifted)

    assert exact.kkt_residual <= KKT_TOL
    assert exact.active_consistency_residual <= KKT_TOL
    assert iterative.kkt_residual <= FW_KKT_TOL

    scale = max(np.linalg.norm(exact.combined_gradient), 1e-12)
    rel_err = np.linalg.norm(exact.combined_gradient - iterative.combined_gradient) / scale
    assert rel_err <= CROSS_CHECK_REL_TOL


def test_case_pareto_reference_reports_small_cross_check_error(rng: np.random.Generator) -> None:
    layout = BlockLayout(tuple([3] * 9))
    inc = build_incidence("partial", 4, layout.num_blocks, rng)
    selectors = build_task_selectors(layout, inc)
    local_grads = [rng.standard_normal(sel.shape[0]) for sel in selectors]

    summary = case_pareto_reference(selectors, local_grads)

    assert summary["active_set"]["kkt_residual"] <= KKT_TOL
    assert summary["frank_wolfe"]["kkt_residual"] <= FW_KKT_TOL
    scale = max(np.linalg.norm(summary["active_set"]["combined_gradient"]), 1e-12)
    assert summary["cross_check_error"] / scale <= CROSS_CHECK_REL_TOL
