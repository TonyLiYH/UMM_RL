"""Unit tests for ``comppareto.adapters.showo2.finite_diff`` (T215).

Pure PyTorch, small synthetic quantities -- no GPU, no real model. Checks
the scale-adaptive epsilon rule, the 4 fixed FD directions, the near-zero-
vs-relative-error gate, and an end-to-end central-difference check against a
known analytic gradient (a small quadratic form, so the finite-difference
estimate should match the true gradient far inside T215's own tolerance
gate).
"""

from __future__ import annotations

import pytest
import torch

from comppareto.adapters.showo2 import finite_diff


def test_scale_adaptive_eps_matches_formula() -> None:
    theta = torch.tensor([3.0, 4.0])  # norm = 5, numel = 2
    eps = finite_diff.scale_adaptive_eps(theta)
    expected = max(1e-3 * (5.0 / (2 ** 0.5)), 1e-6)
    assert eps == expected


def test_scale_adaptive_eps_floors_at_1e_minus_6_for_near_zero_tensor() -> None:
    theta = torch.zeros(10)
    assert finite_diff.scale_adaptive_eps(theta) == 1e-6


def test_scale_adaptive_eps_empty_tensor_is_floor() -> None:
    theta = torch.empty(0)
    assert finite_diff.scale_adaptive_eps(theta) == 1e-6


def test_rademacher_direction_is_unit_norm_and_deterministic_per_seed() -> None:
    d1 = finite_diff.rademacher_direction((6,), seed=42, dtype=torch.float32)
    d2 = finite_diff.rademacher_direction((6,), seed=42, dtype=torch.float32)
    d3 = finite_diff.rademacher_direction((6,), seed=43, dtype=torch.float32)

    assert torch.equal(d1, d2)
    assert not torch.equal(d1, d3)
    assert abs(float(d1.norm()) - 1.0) < 1e-6
    # entries must be proportional to +-1
    unit_scale = 1.0 / (6 ** 0.5)
    assert torch.allclose(d1.abs(), torch.full_like(d1, unit_scale), atol=1e-6)


def test_rademacher_direction_does_not_perturb_ambient_rng() -> None:
    torch.manual_seed(7)
    before = torch.randn(3).clone()
    torch.manual_seed(7)
    _ = finite_diff.rademacher_direction((100,), seed=99, dtype=torch.float32)
    after = torch.randn(3).clone()
    assert torch.equal(before, after)


def test_natural_direction_normalizes_raw_grad() -> None:
    raw_grad = torch.tensor([3.0, 4.0])
    direction = finite_diff.natural_direction(raw_grad)
    assert abs(float(direction.norm()) - 1.0) < 1e-6
    assert torch.allclose(direction, raw_grad / 5.0)


def test_natural_direction_raises_on_zero_gradient() -> None:
    with pytest.raises(ValueError):
        finite_diff.natural_direction(torch.zeros(4))


def test_build_directions_has_4_named_entries_all_unit_norm() -> None:
    theta_s = torch.randn(5)
    raw_grad = torch.randn(5)
    directions = finite_diff.build_directions(theta_s, raw_grad)

    assert set(directions.keys()) == {
        "rademacher_seed_42",
        "rademacher_seed_43",
        "rademacher_seed_44",
        "natural_raw_grad",
    }
    for name, d in directions.items():
        assert d.shape == theta_s.shape
        assert abs(float(d.norm()) - 1.0) < 1e-5, name


def test_gate_relative_branch_above_near_zero_threshold() -> None:
    mode, tol, passed = finite_diff.gate(reference_magnitude=1.0, error_value=5e-4)
    assert mode == "relative"
    assert tol == finite_diff.REL_TOL
    assert passed

    mode2, tol2, passed2 = finite_diff.gate(reference_magnitude=1.0, error_value=5e-2)
    assert mode2 == "relative"
    assert not passed2


def test_gate_absolute_branch_at_or_below_near_zero_threshold() -> None:
    mode, tol, passed = finite_diff.gate(reference_magnitude=1e-9, error_value=1e-7)
    assert mode == "absolute"
    assert tol == finite_diff.ABS_TOL
    assert passed

    mode2, tol2, passed2 = finite_diff.gate(reference_magnitude=1e-9, error_value=1e-5)
    assert mode2 == "absolute"
    assert not passed2


def test_compare_directional_relative_mode_pass_and_fail() -> None:
    # fd_value = (loss_plus - loss_minus) / (2*eps); pick numbers so
    # fd_value = 1.0 exactly, reference_magnitude = 1.0 > NEAR_ZERO_THRESHOLD.
    eps = 0.5
    loss_plus, loss_minus = 1.0, 0.0  # fd_value = (1-0)/(2*0.5) = 1.0
    passing = finite_diff.compare_directional("d", eps, analytic_value=1.0005, loss_plus=loss_plus, loss_minus=loss_minus)
    assert passing.mode == "relative"
    assert passing.passed

    failing = finite_diff.compare_directional("d", eps, analytic_value=1.5, loss_plus=loss_plus, loss_minus=loss_minus)
    assert failing.mode == "relative"
    assert not failing.passed


def test_compare_directional_absolute_mode_near_zero_reference() -> None:
    eps = 0.5
    loss_plus, loss_minus = 1e-9, -1e-9  # fd_value = 2e-9/1.0 = 2e-9 (near zero)
    passing = finite_diff.compare_directional("d", eps, analytic_value=2e-9 + 5e-7, loss_plus=loss_plus, loss_minus=loss_minus)
    assert passing.mode == "absolute"
    assert passing.passed

    failing = finite_diff.compare_directional("d", eps, analytic_value=2e-9 + 5e-5, loss_plus=loss_plus, loss_minus=loss_minus)
    assert failing.mode == "absolute"
    assert not failing.passed


def test_run_finite_difference_check_end_to_end_quadratic_loss() -> None:
    """A pure quadratic ``loss(theta_s) = 0.5 * theta_s^T @ H @ theta_s``
    (H symmetric positive-definite) has an EXACT analytic gradient
    ``H @ theta_s``, so the central-difference estimate at a modest fixed
    eps should match it well inside T215's relative tolerance for all 4
    directions.
    """

    gen = torch.Generator().manual_seed(0)
    n = 4
    m = torch.randn(n, n, generator=gen)
    h = m @ m.T + torch.eye(n)  # SPD
    theta_s = torch.randn(n, generator=gen)

    def loss_at(theta: torch.Tensor) -> float:
        return float(0.5 * theta @ h @ theta)

    analytic_grad = h @ theta_s
    raw_grad = analytic_grad.clone()  # any nonzero vector works for the "natural" direction

    results = finite_diff.run_finite_difference_check(theta_s, raw_grad, analytic_grad, loss_at)

    assert len(results) == 4
    for r in results:
        assert r.passed, (r.label, r.mode, r.error, r.eps, r.fd_value, r.analytic_value)


def test_run_finite_difference_check_detects_a_deliberately_wrong_gradient() -> None:
    """Sanity check on the test harness itself: an obviously wrong analytic
    gradient (scaled by 10x) must fail the gate, confirming the check is not
    vacuously passing.
    """

    gen = torch.Generator().manual_seed(1)
    n = 3
    m = torch.randn(n, n, generator=gen)
    h = m @ m.T + torch.eye(n)
    theta_s = torch.randn(n, generator=gen)

    def loss_at(theta: torch.Tensor) -> float:
        return float(0.5 * theta @ h @ theta)

    correct_grad = h @ theta_s
    wrong_grad = correct_grad * 10.0

    results = finite_diff.run_finite_difference_check(theta_s, correct_grad, wrong_grad, loss_at)
    assert any(not r.passed for r in results)
