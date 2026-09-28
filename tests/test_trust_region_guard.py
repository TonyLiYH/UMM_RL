"""Regression tests for T130: indefinite curvature and trust-region rejection.

The acceptance contract in ``trust_region_guard`` requires that both the
compensated (Schur-based) *predicted* change AND a freshly *measured* change
(with the private coordinate held frozen) both show a genuine decrease before
a step is accepted. These tests verify:

1. Each counterexample task constructs successfully (private curvature PD).
2. Schur complement eigenvalues match the analytic closed-form values.
3. ``trust_region_optimum`` correctly raises ``CurvatureError`` when fed
   ``task.schur()`` for indefinite cases (the existing guard already works
   *when called correctly*).
4. ``trust_region_optimum`` does NOT raise when fed ``task.h_xx`` alone,
   demonstrating the "silent convex projection" danger.
5. The naive (predicted-only) rule produces false accepts for all unsafe cases.
6. ``evaluate_step`` returns the expected verdict and correctly flags
   ``false_accept=True`` for every unsafe case and ``False`` for safe ones.
7. The corrected contract's measured false-accept rate is exactly 0 across
   the whole suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from comppareto import (
    CurvatureError,
    QuadraticTask,
    build_counterexample_suite,
    evaluate_step,
    frozen_measured_change,
    naive_predicted_only_accept,
    project_out_negative_curvature,
    schur_curvature,
    trust_region_optimum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def suite():
    return build_counterexample_suite()


def _case(suite, name):
    for c in suite:
        if c.name == name:
            return c
    raise KeyError(name)


# ---------------------------------------------------------------------------
# 1. All counterexample tasks must construct without error
# ---------------------------------------------------------------------------


def test_all_tasks_construct_successfully(suite):
    """Every counterexample must be constructable (private blocks are PD)."""
    for c in suite:
        assert isinstance(c.task, QuadraticTask), c.name


# ---------------------------------------------------------------------------
# 2. Schur eigenvalues match analytic values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_min",
    [
        ("pure_curvature_2d", -24.0),
        ("safe_control_2d", 0.75),
        ("two_negative_directions_dir_a", -24.0),
        ("two_negative_directions_dir_b", -24.0),
        ("reducible_mixed_step", -24.0),
        ("indefinite_but_genuinely_safe", -24.0),
    ],
)
def test_schur_eigenvalue_matches_analytic(suite, name, expected_min):
    c = _case(suite, name)
    curv = schur_curvature(c.task, tolerance=1e-9)
    assert abs(curv.min_eigenvalue - c.analytic_min_eigenvalue) < 1e-9, (
        f"{name}: got {curv.min_eigenvalue}, expected {c.analytic_min_eigenvalue}"
    )
    assert abs(curv.min_eigenvalue - expected_min) < 1e-9


def test_indefinite_cases_flagged(suite):
    for c in suite:
        curv = schur_curvature(c.task, tolerance=1e-9)
        should_be_indefinite = c.analytic_min_eigenvalue < -1e-9
        assert curv.indefinite == should_be_indefinite, (
            f"{c.name}: indefinite={curv.indefinite}, "
            f"min_eig={curv.min_eigenvalue}"
        )


def test_negative_directions_orthonormal(suite):
    """Negative-curvature eigenvectors from schur_curvature must be unit vectors."""
    for c in suite:
        curv = schur_curvature(c.task, tolerance=1e-9)
        for direction in curv.negative_directions:
            assert abs(np.linalg.norm(direction) - 1.0) < 1e-9, c.name


# ---------------------------------------------------------------------------
# 3. trust_region_optimum raises CurvatureError on indefinite schur()
#    (the existing guard already works when the caller uses schur() correctly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "pure_curvature_2d",
        "two_negative_directions_dir_a",
        "reducible_mixed_step",
        "indefinite_but_genuinely_safe",
    ],
)
def test_trust_region_optimum_raises_on_indefinite_schur(suite, name):
    """trust_region_optimum raises CurvatureError when fed task.schur()."""
    c = _case(suite, name)
    g = c.task.lifted_gradient()
    h = c.task.schur()
    n = h.shape[0]
    with pytest.raises(CurvatureError):
        trust_region_optimum(
            gradient=g, hessian=h, metric=np.eye(n), radius=c.radius
        )


# ---------------------------------------------------------------------------
# 4. trust_region_optimum does NOT raise when only fed h_xx
#    — demonstrating the "silent convex projection" danger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "pure_curvature_2d",
        "two_negative_directions_dir_a",
        "reducible_mixed_step",
        "indefinite_but_genuinely_safe",
    ],
)
def test_trust_region_optimum_silent_on_hxx_alone(suite, name):
    """trust_region_optimum does not detect indefiniteness when fed h_xx alone.

    h_xx is positive definite by construction for all counterexamples, so
    passing it directly (forgetting to use schur()) lets an indefinite-curvature
    step through silently.
    """
    c = _case(suite, name)
    g = c.task.local_gradient
    h_xx = c.task.h_xx
    n = h_xx.shape[0]
    # Must not raise -- PD h_xx passes the eigenvalue check even though the
    # true effective curvature is indefinite.
    result = trust_region_optimum(
        gradient=g, hessian=h_xx, metric=np.eye(n), radius=c.radius
    )
    assert result is not None


# ---------------------------------------------------------------------------
# 5. Naive (predicted-only) rule produces false accepts for unsafe cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["pure_curvature_2d",
                                   "two_negative_directions_dir_a",
                                   "two_negative_directions_dir_b",
                                   "reducible_mixed_step"])
def test_naive_rule_false_accept_for_unsafe_cases(suite, name):
    c = _case(suite, name)
    assert naive_predicted_only_accept(c.task, c.step, tolerance=1e-9), (
        f"{name}: naive rule should have accepted (false accept)"
    )
    measured = frozen_measured_change(c.task, c.step)
    assert measured >= -1e-9, (
        f"{name}: measured change {measured} should be non-negative for a false accept"
    )


# ---------------------------------------------------------------------------
# 6. evaluate_step returns the expected verdict and false_accept flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_verdict,expected_false_accept",
    [
        ("pure_curvature_2d", "reject", True),
        ("safe_control_2d", "accept", False),
        ("two_negative_directions_dir_a", "reject", True),
        ("two_negative_directions_dir_b", "reject", True),
        ("reducible_mixed_step", "reduce", True),
        ("indefinite_but_genuinely_safe", "accept", False),
    ],
)
def test_evaluate_step_verdict(suite, name, expected_verdict, expected_false_accept):
    c = _case(suite, name)
    rec = evaluate_step(c.task, c.step, tolerance=1e-9)
    assert rec.verdict == expected_verdict, (
        f"{name}: got verdict={rec.verdict}, expected={expected_verdict}"
    )
    assert rec.false_accept == expected_false_accept, (
        f"{name}: got false_accept={rec.false_accept}, "
        f"expected={expected_false_accept}"
    )


def test_reduce_case_reduced_step_genuinely_improves(suite):
    """For the 'reduce' case, the reduced step must be genuinely improving."""
    c = _case(suite, "reducible_mixed_step")
    rec = evaluate_step(c.task, c.step, tolerance=1e-9)
    assert rec.verdict == "reduce"
    assert rec.reduced_step is not None
    assert rec.reduced_accept is True
    assert rec.reduced_predicted_change is not None
    assert rec.reduced_measured_change is not None
    assert rec.reduced_predicted_change < -1e-9
    assert rec.reduced_measured_change < -1e-9
    # Reduced step is the v2 component only -- norm should be ~1.0
    assert abs(np.linalg.norm(rec.reduced_step) - 1.0) < 1e-6


def test_reduce_case_specific_values(suite):
    """Verify exact analytic values for the reducible_mixed_step case."""
    c = _case(suite, "reducible_mixed_step")
    rec = evaluate_step(c.task, c.step, tolerance=1e-9)
    assert abs(rec.predicted_change - (-127.38)) < 0.01
    assert abs(rec.measured_change - 0.62) < 0.01
    assert abs(rec.reduced_predicted_change - (-4.5)) < 1e-6
    assert abs(rec.reduced_measured_change - (-4.5)) < 1e-6


def test_pure_curvature_case_specific_values(suite):
    """Verify exact analytic values for the pure_curvature_2d case."""
    c = _case(suite, "pure_curvature_2d")
    rec = evaluate_step(c.task, c.step, tolerance=1e-9)
    assert abs(rec.predicted_change - (-12.0)) < 1e-9
    assert abs(rec.measured_change - 0.5) < 1e-9


def test_indefinite_but_safe_specific_values(suite):
    """Verify exact analytic values for indefinite_but_genuinely_safe."""
    c = _case(suite, "indefinite_but_genuinely_safe")
    rec = evaluate_step(c.task, c.step, tolerance=1e-9)
    assert abs(rec.predicted_change - (-325.0)) < 1e-6
    assert abs(rec.measured_change - (-12.5)) < 1e-6
    assert rec.corrected_accept is True


# ---------------------------------------------------------------------------
# 7. False-accept rate of the corrected contract is exactly 0
# ---------------------------------------------------------------------------


def test_corrected_contract_zero_false_accept_rate(suite):
    """The corrected contract (evaluate_step) must produce zero false accepts."""
    for c in suite:
        rec = evaluate_step(c.task, c.step, tolerance=1e-9)
        assert not rec.false_accept or rec.verdict != "accept", (
            f"{c.name}: corrected contract produced a false accept "
            f"(verdict={rec.verdict}, false_accept={rec.false_accept})"
        )
    # Also verify: for every 'accept' verdict, both predicted and measured < 0
    for c in suite:
        rec = evaluate_step(c.task, c.step, tolerance=1e-9)
        if rec.verdict == "accept":
            assert rec.predicted_change < -1e-9, c.name
            assert rec.measured_change < -1e-9, c.name


# ---------------------------------------------------------------------------
# 8. project_out_negative_curvature removes the unsafe component
# ---------------------------------------------------------------------------


def test_projection_removes_unsafe_component(suite):
    """After projecting out negative-curvature directions, step is orthogonal."""
    c = _case(suite, "reducible_mixed_step")
    curv = schur_curvature(c.task, tolerance=1e-9)
    reduced = project_out_negative_curvature(curv, c.step)
    for direction in curv.negative_directions:
        dot = abs(float(direction @ reduced))
        assert dot < 1e-10, f"residual dot product {dot}"


def test_projection_is_idempotent(suite):
    """Projecting twice gives the same result as projecting once."""
    c = _case(suite, "reducible_mixed_step")
    curv = schur_curvature(c.task, tolerance=1e-9)
    once = project_out_negative_curvature(curv, c.step)
    twice = project_out_negative_curvature(curv, once)
    assert np.allclose(once, twice, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. schur_curvature input validation
# ---------------------------------------------------------------------------


def test_schur_curvature_rejects_nonpositive_tolerance(suite):
    c = suite[0]
    with pytest.raises(ValueError, match="tolerance must be positive"):
        schur_curvature(c.task, tolerance=0.0)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        schur_curvature(c.task, tolerance=-1e-5)


def test_evaluate_step_rejects_nonpositive_tolerance(suite):
    c = suite[0]
    with pytest.raises(ValueError, match="tolerance must be positive"):
        evaluate_step(c.task, c.step, tolerance=-0.1)
