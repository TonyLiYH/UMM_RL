from __future__ import annotations

import numpy as np
import pytest

from comppareto.overlap.regimes import REGIMES
from comppareto.overlap.verify import (
    DEFAULT_TOLERANCES,
    generate_case,
    independent_gather,
    independent_lift_gradient,
    independent_lift_hessian,
    independent_private_optimum,
    verify_case,
)

CONFIG_SEED = 424242

# Frozen protocol: ">=100 seeded cases per overlap regime, dimensions 2-32".
SEEDS_PER_REGIME = 100


@pytest.mark.parametrize("regime", REGIMES)
def test_generate_case_covers_full_dimension_range(regime: str) -> None:
    dims = {generate_case(CONFIG_SEED, i, regime).global_dim for i in range(SEEDS_PER_REGIME)}
    assert min(dims) == 2
    assert max(dims) == 32
    assert dims == set(range(2, 33))


@pytest.mark.parametrize("regime", REGIMES)
def test_generate_case_is_deterministic(regime: str) -> None:
    first = generate_case(CONFIG_SEED, 17, regime)
    second = generate_case(CONFIG_SEED, 17, regime)
    assert first.supports == second.supports
    assert first.global_dim == second.global_dim
    np.testing.assert_array_equal(first.global_step, second.global_step)
    for task_a, task_b in zip(first.tasks, second.tasks):
        np.testing.assert_array_equal(task_a.local_gradient, task_b.local_gradient)
        np.testing.assert_array_equal(task_a.h_xx, task_b.h_xx)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("case_index", range(SEEDS_PER_REGIME))
def test_frozen_protocol_sweep_has_zero_unexplained_mismatches(
    regime: str, case_index: int
) -> None:
    """The pass/fail gate itself: 100 seeded cases/regime, dims 2-32,
    zero unexplained selector or objective mismatches."""

    case = generate_case(CONFIG_SEED, case_index, regime)
    assert 2 <= case.global_dim <= 32
    result = verify_case(case)
    assert result.all_passed, (case_index, regime, case.global_dim, result)


def test_independent_lift_gradient_matches_selector_formula() -> None:
    case = generate_case(CONFIG_SEED, 3, "partial")
    task = case.tasks[0]
    support = case.supports[0]
    lifted = independent_lift_gradient(support, case.global_dim, task.local_gradient)
    np.testing.assert_allclose(lifted, task.lifted_gradient(), atol=1e-12)


def test_independent_lift_hessian_matches_selector_formula() -> None:
    case = generate_case(CONFIG_SEED, 3, "partial")
    task = case.tasks[0]
    support = case.supports[0]
    lifted = independent_lift_hessian(support, case.global_dim, task.h_xx)
    np.testing.assert_allclose(lifted, task.selector.T @ task.h_xx @ task.selector, atol=1e-12)


def test_independent_gather_matches_selector_matmul() -> None:
    case = generate_case(CONFIG_SEED, 3, "full")
    support = case.supports[0]
    task = case.tasks[0]
    gathered = independent_gather(support, case.global_step)
    np.testing.assert_allclose(gathered, task.selector @ case.global_step, atol=1e-12)


def test_independent_private_optimum_matches_closed_form_solve() -> None:
    case = generate_case(CONFIG_SEED, 5, "disjoint")
    task = case.tasks[0]
    support = case.supports[0]
    # QuadraticTask.private_response takes the *global* step and applies the
    # selector internally; independent_private_optimum takes the already
    # gathered *local* step directly.
    closed_form = task.private_response(case.global_step)
    local_step = independent_gather(support, case.global_step)
    solved = independent_private_optimum(task, local_step)
    assert solved.success
    np.testing.assert_allclose(solved.x, closed_form, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("regime", REGIMES)
def test_safe_set_invariance_is_vacuous_under_full_overlap(regime: str) -> None:
    for case_index in range(20):
        case = generate_case(CONFIG_SEED, case_index, regime)
        result = verify_case(case)
        if regime == "full":
            assert all(pair.invariance is None for pair in result.safe_set_pairs)
        elif regime == "disjoint" and case.num_tasks >= 2:
            # every pair has a fully private complement under disjoint
            assert all(pair.invariance is not None for pair in result.safe_set_pairs)


def test_safe_set_invariance_is_realized_for_partial_overlap_when_private_coords_exist() -> None:
    saw_non_vacuous = False
    for case_index in range(SEEDS_PER_REGIME):
        case = generate_case(CONFIG_SEED, case_index, "partial")
        result = verify_case(case)
        for pair in result.safe_set_pairs:
            if pair.invariance is not None:
                saw_non_vacuous = True
                assert pair.invariance.passed
    assert saw_non_vacuous


def test_perturbing_a_shared_coordinate_can_change_the_other_tasks_objective() -> None:
    """Sanity check that the safe-set invariance check is non-trivial: a
    perturbation supported on a *shared* coordinate (not S_i \\ S_j) is not
    guaranteed to leave task j's compensated_change unchanged, so the check
    in comppareto.overlap.verify is actually discriminating, not vacuously
    true by construction for every perturbation."""

    found_a_case_where_shared_perturbation_matters = False
    rng = np.random.default_rng(99)
    for case_index in range(50):
        case = generate_case(CONFIG_SEED, case_index, "partial")
        for i in range(case.num_tasks):
            for j in range(case.num_tasks):
                if i == j:
                    continue
                shared = set(case.supports[i]) & set(case.supports[j])
                if not shared:
                    continue
                delta = np.zeros(case.global_dim)
                idx = sorted(shared)
                delta[idx] = rng.standard_normal(len(idx))
                task_j = case.tasks[j]
                before = task_j.compensated_change(case.global_step)
                after = task_j.compensated_change(case.global_step + delta)
                if abs(after - before) > DEFAULT_TOLERANCES["safe_set"]:
                    found_a_case_where_shared_perturbation_matters = True
    assert found_a_case_where_shared_perturbation_matters
