from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from comppareto.oracle.highprecision import (
    KNOWN_FAILURES,
    exact_loss_change_hp,
    meta_loss_hp,
    quadratic_model_hp,
    recheck_known_failures,
    recheck_momentum_loss_change,
    recheck_momentum_loss_change_decimal,
    reconstruct_task_inputs,
    rerun_gradient_hp,
    sensitivity_hp,
)
from comppareto.oracle.hypergradient import exact_loss_change, quadratic_model, rerun_gradient
from comppareto.oracle.momentum import momentum_closed_form_state, momentum_sensitivity
from comppareto.oracle.sweep import enumerate_cases

RUN_CONFIG_PATH = Path(__file__).resolve().parents[2] / "runs" / "oracle-20260827-baseline" / "config.yaml"

FD_REL_TOL = 1e-6


def _load_run_config() -> dict:
    import yaml

    with open(RUN_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def run_config() -> dict:
    return _load_run_config()


@pytest.mark.parametrize("case_index,task_index", KNOWN_FAILURES)
def test_reconstruct_task_inputs_reproduces_known_failure_magnitude(
    run_config: dict, case_index: int, task_index: int
) -> None:
    # Independently re-derive that the reconstructed inputs land on the same
    # float64 relative error already recorded in the frozen failure ledger,
    # proving reconstruct_task_inputs replays run_case's RNG state correctly
    # rather than silently drifting onto a different task/case.
    specs = {spec.case_index: spec for spec in enumerate_cases(run_config)}
    spec = specs[case_index]
    task, x_i, phi_0, v_0, eta, beta, noise, step = reconstruct_task_inputs(spec, task_index)

    d = task.private_dim
    w_k = momentum_sensitivity(task, eta, beta, noise.shape[0])
    z_k_phi = w_k[:d]
    r_k_phi, _ = momentum_closed_form_state(task, np.zeros_like(x_i), phi_0, v_0, eta, beta, noise)
    phi_k, _ = momentum_closed_form_state(task, x_i, phi_0, v_0, eta, beta, noise)
    _, q = quadratic_model(task, z_k_phi, r_k_phi)
    grad = rerun_gradient(task, x_i, phi_k, z_k_phi)
    exact_delta = exact_loss_change(grad, q, step)

    def loss(xi: np.ndarray) -> float:
        phi_k_, _ = momentum_closed_form_state(task, xi, phi_0, v_0, eta, beta, noise)
        return task.meta_loss(xi, phi_k_)

    direct_delta = loss(x_i + step) - loss(x_i)
    rel_err = abs(exact_delta - direct_delta) / abs(direct_delta)

    assert rel_err > 1e-9
    assert rel_err < 1e-6


@pytest.mark.parametrize("case_index,task_index", KNOWN_FAILURES)
def test_recheck_momentum_loss_change_reports_all_fields(
    run_config: dict, case_index: int, task_index: int
) -> None:
    specs = {spec.case_index: spec for spec in enumerate_cases(run_config)}
    spec = specs[case_index]
    task, x_i, phi_0, v_0, eta, beta, noise, step = reconstruct_task_inputs(spec, task_index)

    report = recheck_momentum_loss_change(task, x_i, phi_0, v_0, eta, beta, noise, step)

    for key in ("float64", "longdouble", "decimal", "forward_error_vs_float64", "conditioning", "trajectory_amplification"):
        assert key in report
    for key in ("exact_delta", "direct_delta", "absolute_error", "relative_error"):
        assert key in report["float64"]
        assert key in report["longdouble"]
        assert key in report["decimal"]
    for key in ("precision_digits", "term_grad_dot_step", "term_half_step_q_step", "baseline_loss_magnitude"):
        assert key in report["decimal"]
    for key in ("cond_q", "cancellation_ratio", "term_grad_dot_step", "term_half_step_q_step"):
        assert key in report["conditioning"]
    assert isinstance(report["pure_cancellation"], bool)
    assert report["decimal"]["precision_digits"] >= 80

    # Every reported scalar must be finite and JSON-serializable as a plain float.
    json.dumps(report)
    for section in ("float64", "longdouble", "decimal"):
        for value in report[section].values():
            assert np.isfinite(value)
    assert np.isfinite(report["trajectory_amplification"])


@pytest.mark.parametrize("case_index,task_index", KNOWN_FAILURES)
def test_longdouble_recheck_does_not_relax_the_frozen_tolerance(
    run_config: dict, case_index: int, task_index: int
) -> None:
    # R5 explicitly forbids relaxing the float64 1e-9 tolerance: the reported
    # float64 relative error here must still exceed that tolerance, matching
    # failure_ledger.json, regardless of what the longdouble side finds.
    specs = {spec.case_index: spec for spec in enumerate_cases(run_config)}
    spec = specs[case_index]
    task, x_i, phi_0, v_0, eta, beta, noise, step = reconstruct_task_inputs(spec, task_index)

    report = recheck_momentum_loss_change(task, x_i, phi_0, v_0, eta, beta, noise, step)

    assert report["float64"]["relative_error"] > 1e-9


@pytest.mark.parametrize("case_index,task_index", KNOWN_FAILURES)
def test_longdouble_pipeline_reports_finite_diagnostic_values(
    run_config: dict, case_index: int, task_index: int
) -> None:
    # R7 (second local review): numpy.longdouble's width is platform-dependent
    # (e.g. 80-bit x86 extended on some hosts, identical to float64 on
    # others), so it can no longer be asserted to strictly beat float64's
    # relative error -- that assertion previously failed on a narrower-
    # longdouble host. It remains a useful secondary diagnostic (reported
    # alongside float64 and the platform-independent Decimal reference below)
    # but is no longer the authoritative independent check.
    specs = {spec.case_index: spec for spec in enumerate_cases(run_config)}
    spec = specs[case_index]
    task, x_i, phi_0, v_0, eta, beta, noise, step = reconstruct_task_inputs(spec, task_index)

    report = recheck_momentum_loss_change(task, x_i, phi_0, v_0, eta, beta, noise, step)

    assert np.isfinite(report["longdouble"]["relative_error"])
    assert report["conditioning"]["cond_q"] > 0
    assert report["conditioning"]["cancellation_ratio"] >= 1.0


@pytest.mark.parametrize("case_index,task_index", KNOWN_FAILURES)
def test_decimal_pipeline_establishes_pure_cancellation_for_known_failures(
    run_config: dict, case_index: int, task_index: int
) -> None:
    # R7's authoritative independent reference: a platform-independent
    # Decimal(precision>=80) reconstruction, built from exactly represented
    # float64 inputs via a fully independent code path (literal-recurrence
    # mirrors of momentum_unroll/momentum_sensitivity_trajectory, not the
    # matrix-power closed form and not a cast of an already-computed
    # float64/longdouble result). Its relative error must be dramatically
    # smaller than float64's -- by orders of magnitude, not just "smaller" --
    # to count as evidence of pure catastrophic cancellation rather than a
    # formula/implementation mismatch (which R7 requires be fixed in the
    # implementation, not the test, if it did not shrink this much).
    specs = {spec.case_index: spec for spec in enumerate_cases(run_config)}
    spec = specs[case_index]
    task, x_i, phi_0, v_0, eta, beta, noise, step = reconstruct_task_inputs(spec, task_index)

    report = recheck_momentum_loss_change(task, x_i, phi_0, v_0, eta, beta, noise, step)

    assert report["decimal"]["precision_digits"] >= 80
    assert report["decimal"]["relative_error"] < report["float64"]["relative_error"] / 100.0
    assert report["pure_cancellation"] is True
    assert report["trajectory_amplification"] > 0


def test_decimal_pipeline_matches_float64_pipeline_on_a_well_conditioned_case(
    rng: np.random.Generator,
) -> None:
    # Formula-correctness sanity check for the independent Decimal pipeline
    # (used above for the two known extended-precision-failure cases) against
    # the already-tested float64 pipeline, on a well-conditioned synthetic
    # case where no catastrophic cancellation occurs -- confirms the Decimal
    # mirror reproduces the same identity, not merely that it runs at higher
    # precision.
    from tests.oracle._helpers import make_task

    task = make_task(rng, 3, 4)
    eta, beta, steps = 0.05, 0.9, 5
    x_i = rng.standard_normal(3)
    phi_0 = rng.standard_normal(4)
    v_0 = rng.standard_normal(4)
    noise = 0.05 * rng.standard_normal((steps, 4))
    step = 0.02 * rng.standard_normal(3)

    d = task.private_dim
    phi_k, _ = momentum_closed_form_state(task, x_i, phi_0, v_0, eta, beta, noise)
    w_k = momentum_sensitivity(task, eta, beta, steps)
    z_k_phi = w_k[:d]
    r_k_phi, _ = momentum_closed_form_state(task, np.zeros_like(x_i), phi_0, v_0, eta, beta, noise)
    _, q = quadratic_model(task, z_k_phi, r_k_phi)
    grad = rerun_gradient(task, x_i, phi_k, z_k_phi)
    exact_delta_f64 = exact_loss_change(grad, q, step)

    def direct_loss(xi: np.ndarray) -> float:
        phi_k_, _ = momentum_closed_form_state(task, xi, phi_0, v_0, eta, beta, noise)
        return task.meta_loss(xi, phi_k_)

    direct_delta_f64 = direct_loss(x_i + step) - direct_loss(x_i)

    report = recheck_momentum_loss_change_decimal(task, x_i, phi_0, v_0, eta, beta, noise, step)

    assert abs(report["exact_delta"] - exact_delta_f64) / abs(exact_delta_f64) <= 1e-9
    assert abs(report["direct_delta"] - direct_delta_f64) / abs(direct_delta_f64) <= 1e-9
    assert report["relative_error"] < 1e-15


def test_recheck_known_failures_covers_both_ledger_entries(run_config: dict) -> None:
    report = recheck_known_failures(run_config)
    assert set(report.keys()) == {41, 287}
    assert report[41]["task_index"] == 4
    assert report[287]["task_index"] == 1
    for case_index, entry in report.items():
        assert entry["optimizer"] == "momentum"
        assert entry["stability_regime"] == "unstable"
        assert entry["float64"]["relative_error"] > 1e-9


def test_hp_helper_functions_match_float64_helpers_bit_for_bit_in_structure(rng: np.random.Generator) -> None:
    # Sanity check the hand-mirrored longdouble helpers against their float64
    # counterparts on a small synthetic case where both dtypes should agree
    # to float64 precision (confirms the mirror is formula-correct, not just
    # dtype-preserving).
    from comppareto.oracle.hypergradient import quadratic_model as _qm
    from comppareto.oracle.hypergradient import rerun_gradient as _rg
    from comppareto.oracle.momentum import momentum_sensitivity as _ms
    from tests.oracle._helpers import make_task

    task = make_task(rng, 3, 4)
    eta, beta, steps = 0.05, 0.9, 5
    d = task.private_dim

    w_k = _ms(task, eta, beta, steps)
    w_k_hp = sensitivity_hp(
        np.asarray(task.private_curvature, dtype=np.longdouble),
        np.asarray(task.h_phix, dtype=np.longdouble),
        eta,
        beta,
        steps,
    )
    rel_err = np.linalg.norm(np.asarray(w_k_hp, dtype=np.float64) - w_k) / np.linalg.norm(w_k)
    assert rel_err <= 1e-10

    x_i = rng.standard_normal(3)
    phi_0 = rng.standard_normal(4)
    v_0 = rng.standard_normal(4)
    noise = 0.05 * rng.standard_normal((steps, 4))

    from comppareto.oracle.momentum import momentum_closed_form_state as _cfs

    phi_k, v_k = _cfs(task, x_i, phi_0, v_0, eta, beta, noise)

    from comppareto.oracle.highprecision import closed_form_state_hp

    phi_k_hp, v_k_hp = closed_form_state_hp(
        np.asarray(task.private_curvature, dtype=np.longdouble),
        np.asarray(task.h_phix, dtype=np.longdouble),
        np.asarray(task.private_linear, dtype=np.longdouble),
        np.asarray(x_i, dtype=np.longdouble),
        np.asarray(phi_0, dtype=np.longdouble),
        np.asarray(v_0, dtype=np.longdouble),
        eta,
        beta,
        np.asarray(noise, dtype=np.longdouble),
    )
    assert np.linalg.norm(np.asarray(phi_k_hp, dtype=np.float64) - phi_k) / np.linalg.norm(phi_k) <= 1e-10
    assert np.linalg.norm(np.asarray(v_k_hp, dtype=np.float64) - v_k) / np.linalg.norm(v_k) <= 1e-10

    z_k_phi = w_k[:d]
    z_k_phi_hp = w_k_hp[:d]
    r_k_phi, _ = _cfs(task, np.zeros_like(x_i), phi_0, v_0, eta, beta, noise)
    r_k_phi_hp, _ = closed_form_state_hp(
        np.asarray(task.private_curvature, dtype=np.longdouble),
        np.asarray(task.h_phix, dtype=np.longdouble),
        np.asarray(task.private_linear, dtype=np.longdouble),
        np.zeros_like(x_i, dtype=np.longdouble),
        np.asarray(phi_0, dtype=np.longdouble),
        np.asarray(v_0, dtype=np.longdouble),
        eta,
        beta,
        np.asarray(noise, dtype=np.longdouble),
    )

    g_hat, q = _qm(task, z_k_phi, r_k_phi)
    g_hat_hp, q_hp = quadratic_model_hp(
        np.asarray(task.h_xx, dtype=np.longdouble),
        np.asarray(task.h_xphi, dtype=np.longdouble),
        np.asarray(task.h_phiphi, dtype=np.longdouble),
        np.asarray(task.a_meta, dtype=np.longdouble),
        np.asarray(task.b_meta, dtype=np.longdouble),
        z_k_phi_hp,
        r_k_phi_hp,
    )
    assert np.linalg.norm(np.asarray(q_hp, dtype=np.float64) - q) / np.linalg.norm(q) <= 1e-10

    grad = _rg(task, x_i, phi_k, z_k_phi)
    grad_hp = rerun_gradient_hp(
        np.asarray(task.a_meta, dtype=np.longdouble),
        np.asarray(task.h_xx, dtype=np.longdouble),
        np.asarray(task.h_xphi, dtype=np.longdouble),
        np.asarray(task.b_meta, dtype=np.longdouble),
        np.asarray(task.h_phiphi, dtype=np.longdouble),
        np.asarray(x_i, dtype=np.longdouble),
        phi_k_hp,
        z_k_phi_hp,
    )
    assert np.linalg.norm(np.asarray(grad_hp, dtype=np.float64) - grad) / np.linalg.norm(grad) <= 1e-10

    step = 0.02 * rng.standard_normal(3)
    exact_delta = exact_loss_change(grad, q, step)
    exact_delta_hp = exact_loss_change_hp(grad_hp, q_hp, np.asarray(step, dtype=np.longdouble))
    assert abs(float(exact_delta_hp) - exact_delta) / abs(exact_delta) <= 1e-10

    loss = task.meta_loss(x_i, phi_k)
    loss_hp = meta_loss_hp(
        np.asarray(task.a_meta, dtype=np.longdouble),
        np.asarray(task.h_xx, dtype=np.longdouble),
        np.asarray(task.h_xphi, dtype=np.longdouble),
        np.asarray(task.b_meta, dtype=np.longdouble),
        np.asarray(task.h_phiphi, dtype=np.longdouble),
        np.asarray(x_i, dtype=np.longdouble),
        phi_k_hp,
    )
    assert abs(float(loss_hp) - loss) / abs(loss) <= 1e-10
