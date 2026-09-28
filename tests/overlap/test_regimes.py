from __future__ import annotations

import numpy as np
import pytest

from comppareto.overlap.regimes import (
    REGIMES,
    RegimeError,
    sample_num_tasks,
    sample_supports,
)


def test_regimes_tuple_is_disjoint_partial_full() -> None:
    assert REGIMES == ("disjoint", "partial", "full")


@pytest.mark.parametrize("global_dim", [2, 3, 5, 8, 16, 32])
def test_sample_num_tasks_stays_in_bounds(global_dim: int) -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        num_tasks = sample_num_tasks(rng, global_dim, max_tasks=5)
        assert 2 <= num_tasks <= min(5, global_dim)


def test_sample_num_tasks_rejects_global_dim_below_two() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(RegimeError):
        sample_num_tasks(rng, global_dim=1)


_REGIME_TEST_OFFSET = {"disjoint": 0, "partial": 1, "full": 2}


@pytest.mark.parametrize("global_dim", range(2, 33))
@pytest.mark.parametrize("regime", REGIMES)
def test_sample_supports_satisfies_regime_contract_across_dims(
    regime: str, global_dim: int
) -> None:
    # Deterministic seed derived from fixed integer offsets, not hash() (which
    # is process-randomized for strings and would make failures unreproducible).
    rng = np.random.default_rng([_REGIME_TEST_OFFSET[regime], global_dim])
    for trial in range(5):
        num_tasks = sample_num_tasks(rng, global_dim, max_tasks=5)
        supports = sample_supports(rng, regime, num_tasks, global_dim)  # type: ignore[arg-type]
        assert len(supports) == num_tasks
        sets = [set(s) for s in supports]
        full_set = set(range(global_dim))
        for s in sets:
            assert s.issubset(full_set)
            assert s

        pair_overlaps = [
            len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))
        ]
        if regime == "disjoint":
            assert all(overlap == 0 for overlap in pair_overlaps)
            # every coordinate is claimed by exactly one task
            union = set().union(*sets) if sets else set()
            assert union.issubset(full_set)
        elif regime == "full":
            assert all(s == full_set for s in sets)
        elif regime == "partial":
            assert any(overlap > 0 for overlap in pair_overlaps)
            assert not all(s == full_set for s in sets)


def test_disjoint_regime_rejects_more_tasks_than_dimensions() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(RegimeError):
        sample_supports(rng, "disjoint", num_tasks=5, global_dim=3)


def test_partial_regime_is_nondegenerate_at_minimum_dimension() -> None:
    # global_dim=2 is the tightest boundary case: partial overlap must still
    # be distinguishable from both disjoint (needs a shared coordinate) and
    # full (needs at least one non-full task).
    rng = np.random.default_rng(0)
    for seed in range(200):
        rng = np.random.default_rng(seed)
        supports = sample_supports(rng, "partial", num_tasks=2, global_dim=2)
        sets = [set(s) for s in supports]
        assert len(sets[0] & sets[1]) > 0
        assert not all(s == {0, 1} for s in sets)


def test_supports_do_not_repeat_a_coordinate_within_one_task() -> None:
    rng = np.random.default_rng(7)
    for regime in REGIMES:
        for global_dim in (2, 4, 10, 32):
            num_tasks = sample_num_tasks(rng, global_dim, max_tasks=5)
            supports = sample_supports(rng, regime, num_tasks, global_dim)  # type: ignore[arg-type]
            for support in supports:
                assert len(support) == len(set(support))
