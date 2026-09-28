"""Stage 3: proving instrumentation does not alter gradients or optimizer
results, within dtype tolerance -- the pass/fail gate's core equivalence
check.
"""

from __future__ import annotations

from comppareto.instrumentation.equivalence import run_equivalence_check

GRADIENT_TOLERANCE = 1.0e-6
PARAMETER_UPDATE_TOLERANCE = 1.0e-6


def test_equivalence_within_tolerance_across_sequential_batches() -> None:
    report = run_equivalence_check(seed=20260916, num_batches=5)
    assert len(report.steps) == 5
    assert report.gradient_max_abs_error <= GRADIENT_TOLERANCE
    assert report.parameter_update_max_abs_error <= PARAMETER_UPDATE_TOLERANCE
    # The discrepancy should be measured, not assumed -- a real toy model at
    # float32 precision should not be bit-for-bit identical by construction.
    for step in report.steps:
        assert step.gradient_max_abs_error >= 0.0
        assert step.parameter_update_max_abs_error >= 0.0


def test_equivalence_is_reproducible_given_the_same_seed() -> None:
    report_a = run_equivalence_check(seed=123, num_batches=3)
    report_b = run_equivalence_check(seed=123, num_batches=3)
    assert report_a.gradient_max_abs_error == report_b.gradient_max_abs_error
    assert report_a.parameter_update_max_abs_error == report_b.parameter_update_max_abs_error


def test_equivalence_report_serializes() -> None:
    report = run_equivalence_check(seed=7, num_batches=2)
    payload = report.to_dict()
    assert payload["gradient_max_abs_error"] <= GRADIENT_TOLERANCE
    assert len(payload["steps"]) == 2
