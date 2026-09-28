"""Tests for the T120 comparison suite (production vs. independent reference).

See ``tasks/T120-independent-kkt-reference.md`` for the frozen protocol and
``src/comppareto/kkt_compare.py`` for the threshold rationale.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from comppareto import kkt_compare as compare


def test_fixed_private_response_schur_cases_all_pass() -> None:
    for case in compare.fixed_private_response_schur_cases():
        result = compare.compare_private_response_and_schur(case)
        assert result.passed, (result.case_id, result.failure_reasons, result.checks)


def test_fixed_trust_region_cases_all_pass() -> None:
    for case in compare.fixed_trust_region_cases():
        result = compare.compare_trust_region(case)
        assert result.passed, (result.case_id, result.failure_reasons, result.checks)


def test_fixed_negotiation_cases_all_pass() -> None:
    for case in compare.fixed_negotiation_cases():
        result = compare.compare_negotiation(case)
        assert result.passed, (result.case_id, result.failure_reasons, result.checks)


def test_random_cases_are_reproducible_given_a_seed() -> None:
    rng_a = np.random.default_rng(4242)
    rng_b = np.random.default_rng(4242)
    cases_a = compare.random_trust_region_cases(rng_a, 5)
    cases_b = compare.random_trust_region_cases(rng_b, 5)
    for case_a, case_b in zip(cases_a, cases_b, strict=True):
        np.testing.assert_array_equal(case_a["gradient"], case_b["gradient"])
        np.testing.assert_array_equal(case_a["hessian"], case_b["hessian"])


def test_random_negotiation_case_generator_produces_psd_schur_hessians() -> None:
    # Regression for the joint-PD-block generator fix: sampling h_xx and
    # h_phiphi as independently-PD blocks does NOT guarantee the resulting
    # Schur complement is PSD, which used to make trust_region_optimum raise
    # CurvatureError deep inside production for some random seeds.
    rng = np.random.default_rng(777)
    cases = compare.random_negotiation_cases(rng, 10)
    for case in cases:
        result = compare.compare_negotiation(case)
        assert "solver_error" not in result.checks, (case["case_id"], result.notes)


def test_run_comparison_suite_small_configuration_all_passed() -> None:
    suite = compare.run_comparison_suite(
        seed=1234,
        random_private_response_count=5,
        random_trust_region_count=5,
        random_negotiation_count=5,
    )
    assert suite.counts["total"] == suite.counts["private_response_schur"] + (
        suite.counts["trust_region"] + suite.counts["negotiation"]
    )
    assert suite.all_passed, [r.case_id for r in suite.failures]
    assert len(suite.failures) == 0


def test_full_suite_default_configuration_all_passed() -> None:
    suite = compare.run_comparison_suite(seed=20260928)
    assert suite.all_passed, [
        (r.case_id, r.failure_reasons, r.checks) for r in suite.failures
    ]
    # Sanity on case counts: 5 fixed + N random for private_response/schur,
    # 4 fixed + N random for trust_region, 2 fixed + N random for negotiation.
    assert suite.counts["private_response_schur"] == 5 + 40
    assert suite.counts["trust_region"] == 4 + 40
    assert suite.counts["negotiation"] == 2 + 20


def test_write_suite_artifacts_produces_expected_files(tmp_path) -> None:
    suite = compare.run_comparison_suite(
        seed=99,
        random_private_response_count=2,
        random_trust_region_count=2,
        random_negotiation_count=2,
    )
    out_dir = tmp_path / "suite-out"
    compare.write_suite_artifacts(suite, out_dir)

    residual_table = json.loads((out_dir / "residual-table.json").read_text())
    assert len(residual_table) == suite.counts["total"]
    assert {"case_id", "family", "checks", "thresholds", "passed"} <= set(
        residual_table[0].keys()
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["counts"] == suite.counts
    assert summary["all_passed"] == suite.all_passed
    assert "max_observed_by_check" in summary
    assert "thresholds" in summary

    failure_ledger = json.loads((out_dir / "failure-ledger.json").read_text())
    assert failure_ledger == []  # this configuration is expected to pass

    csv_text = (out_dir / "residual-table.csv").read_text()
    assert csv_text.splitlines()[0] == "case_id,family,generator,passed,failure_reasons"
    assert len(csv_text.splitlines()) == suite.counts["total"] + 1


def test_threshold_bar_has_margin_against_observed_maxima() -> None:
    """Guard against thresholds that are so loose they cannot fail.

    Every observed maximum across the full default suite should sit
    comfortably under its threshold (by at least 3x), so that a genuine
    regression of this magnitude would be caught rather than absorbed by
    slack in the bar itself.
    """

    suite = compare.run_comparison_suite(seed=20260928)
    max_by_check: dict[str, float] = {}
    for result in suite.results:
        for name, value in result.checks.items():
            max_by_check[name] = max(max_by_check.get(name, 0.0), value)

    for name, threshold in compare.THRESHOLDS.items():
        observed = max_by_check.get(name)
        if observed is None:
            continue
        assert observed <= threshold, (name, observed, threshold)
        # Skip the margin check for exactly-zero observations (e.g. an exact
        # rational path with literal 0 error) since any positive threshold
        # trivially has "infinite" margin there.
        if observed > 0:
            assert threshold / max(observed, 1e-300) >= 3.0, (
                name,
                observed,
                threshold,
            )
