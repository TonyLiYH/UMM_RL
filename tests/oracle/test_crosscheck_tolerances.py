from __future__ import annotations

import numpy as np

from comppareto.oracle.crosscheck import NEAR_ZERO_ABS_TOL, NEAR_ZERO_NORM, compare


def test_compare_uses_relative_error_above_near_zero_threshold() -> None:
    ref = np.array([1.0, 2.0, 3.0])
    val = ref * (1 + 1e-11)
    result = compare("x", ref, val, rel_tol=1e-9)
    assert result.mode == "relative"
    assert result.passed


def test_compare_switches_to_absolute_below_near_zero_threshold() -> None:
    ref = np.array([NEAR_ZERO_NORM / 10, 0.0, 0.0])
    val = ref + NEAR_ZERO_ABS_TOL / 10
    result = compare("x", ref, val, rel_tol=1e-9)
    assert result.mode == "absolute"
    assert result.passed


def test_compare_fails_when_absolute_error_exceeds_near_zero_tolerance() -> None:
    ref = np.zeros(3)
    val = np.array([1e-8, 0.0, 0.0])
    result = compare("x", ref, val, rel_tol=1e-9)
    assert result.mode == "absolute"
    assert not result.passed


def test_compare_selector_contract_is_exact_boolean() -> None:
    from comppareto.oracle.selectors import SelectorError, validate_selector

    good = np.array([[1.0, 0.0], [0.0, 1.0]])
    validate_selector(good)  # no exception == pass
    try:
        validate_selector(np.array([[0.3, 0.7]]))
        raised = False
    except SelectorError:
        raised = True
    assert raised
