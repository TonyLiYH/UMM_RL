from __future__ import annotations

import numpy as np
import pytest

from comppareto.oracle.selectors import (
    GRAPH_FAMILIES,
    BlockLayout,
    SelectorError,
    build_incidence,
    build_selector,
    build_task_selectors,
    validate_selector,
)


def test_block_layout_slices() -> None:
    layout = BlockLayout((2, 3, 1))
    assert layout.num_blocks == 3
    assert layout.total_dim == 6
    assert layout.block_slice(0) == slice(0, 2)
    assert layout.block_slice(1) == slice(2, 5)
    assert layout.block_slice(2) == slice(5, 6)


def test_block_layout_rejects_empty_or_nonpositive() -> None:
    with pytest.raises(SelectorError):
        BlockLayout(())
    with pytest.raises(SelectorError):
        BlockLayout((2, 0, 1))


def test_build_selector_single_lift_no_duplicate() -> None:
    layout = BlockLayout((2, 2, 2))
    selector = build_selector(layout, [0, 2])
    assert selector.shape == (4, 6)
    validate_selector(selector)
    assert np.array_equal(np.sum(selector, axis=1), np.ones(4))
    assert np.all(np.sum(selector, axis=0) <= 1.0)


def test_validate_selector_rejects_violations() -> None:
    with pytest.raises(SelectorError):
        validate_selector(np.array([[0.5, 0.5]]))
    with pytest.raises(SelectorError):
        validate_selector(np.array([[1.0, 1.0]]))
    with pytest.raises(SelectorError):
        validate_selector(np.array([[1.0, 0.0], [1.0, 0.0]]))


@pytest.mark.parametrize("family", GRAPH_FAMILIES)
def test_build_incidence_matches_family_property(family: str) -> None:
    rng = np.random.default_rng(1)
    inc = build_incidence(family, num_tasks=4, num_blocks=8, rng=rng)
    col_counts = inc.sum(axis=0)
    assert np.all(inc.sum(axis=1) >= 1)
    if family == "disjoint":
        assert np.all(col_counts <= 1)
    elif family == "full_overlap":
        assert np.all(col_counts == 4)
    elif family == "partial":
        assert np.any(col_counts >= 2)
        assert np.any(col_counts == 1)


def test_build_incidence_rejects_out_of_range_sizes() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(SelectorError):
        build_incidence("disjoint", num_tasks=1, num_blocks=8, rng=rng)
    with pytest.raises(SelectorError):
        build_incidence("disjoint", num_tasks=4, num_blocks=2, rng=rng)


def test_build_task_selectors_all_valid() -> None:
    rng = np.random.default_rng(2)
    layout = BlockLayout((2,) * 8)
    inc = build_incidence("star", num_tasks=4, num_blocks=8, rng=rng)
    selectors = build_task_selectors(layout, inc)
    assert len(selectors) == 4
    for selector in selectors:
        validate_selector(selector)
