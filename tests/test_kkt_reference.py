"""Tests for the independent KKT/direct-solve reference (T120).

These tests exercise ``comppareto.kkt_reference`` on its own terms: basic
correctness of each solve path, cross-checks between the Cholesky and
exact-rational paths (both independent of production), and validation/error
behavior. Cross-checks against the *production* solver in
``comppareto.quadratic`` live in ``tests/test_kkt_compare.py`` instead, since
mixing that comparison into this file would blur the "reference code must
not import the production solution functions under test" boundary that this
module itself observes.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from comppareto import kkt_reference as ref


def test_module_does_not_import_production() -> None:
    """The frozen protocol requires the reference to never import production.

    The module docstring is allowed to *mention* ``comppareto.quadratic`` in
    prose (it does, to explain the independence strategy) -- what must never
    happen is an actual ``import`` statement pulling it in. (This test
    deliberately does not check ``sys.modules``: other test modules in the
    same pytest process legitimately import ``comppareto.quadratic``
    directly, e.g. to drive the comparison suite, so process-wide module
    state is not a reliable signal here -- only the reference module's own
    source is.)
    """

    import ast
    import inspect

    import comppareto.kkt_reference as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    assert not any("quadratic" in name for name in imported_names)


# ---------------------------------------------------------------------------
# Private response / Schur complement (Cholesky path)
# ---------------------------------------------------------------------------


def test_independent_private_response_matches_hand_solved_system() -> None:
    h_xphi = np.array([[0.5], [-0.3]])
    h_phiphi = np.array([[1.2]])
    mu = 0.8
    local_step = np.array([0.3, -0.1])
    # curvature = 1.2 + 0.8 = 2.0; rhs = -(h_xphi^T @ local_step) = -(0.5*0.3 + -0.3*-0.1)
    expected_rhs = -(0.5 * 0.3 + (-0.3) * (-0.1))
    expected = np.array([expected_rhs / 2.0])
    result = ref.independent_private_response(h_xphi, h_phiphi, mu, local_step)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_independent_schur_complement_matches_hand_solved_system() -> None:
    h_xx = np.array([[3.0, 0.4], [0.4, 2.0]])
    h_xphi = np.array([[0.5], [-0.3]])
    h_phiphi = np.array([[1.2]])
    mu = 0.8
    # curvature = 2.0; correction = h_xphi @ h_xphi.T / 2.0
    correction = (h_xphi @ h_xphi.T) / 2.0
    expected = h_xx - correction
    result = ref.independent_schur_complement(h_xx, h_xphi, h_phiphi, mu)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_exact_rational_solve_basic_system() -> None:
    matrix = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(3)]]
    rhs = [Fraction(5), Fraction(10)]
    solution = ref.exact_rational_solve(matrix, rhs)
    assert solution == [Fraction(1), Fraction(3)]


def test_exact_rational_solve_rejects_singular_matrix() -> None:
    matrix = [[Fraction(1), Fraction(2)], [Fraction(2), Fraction(4)]]
    rhs = [Fraction(1), Fraction(2)]
    with pytest.raises(ValueError, match="singular"):
        ref.exact_rational_solve(matrix, rhs)


def test_exact_rational_solve_rejects_non_square_matrix() -> None:
    matrix = [[Fraction(1), Fraction(2), Fraction(3)], [Fraction(1), Fraction(1), Fraction(1)]]
    rhs = [Fraction(1), Fraction(1)]
    with pytest.raises(ValueError, match="square"):
        ref.exact_rational_solve(matrix, rhs)


def test_exact_rational_private_response_matches_cholesky_path() -> None:
    h_xphi = np.array([[0.5], [-0.3]])
    h_phiphi = np.array([[1.2]])
    mu = 0.8
    local_step = np.array([0.3, -0.1])
    cholesky_result = ref.independent_private_response(h_xphi, h_phiphi, mu, local_step)
    exact_result = ref.exact_rational_private_response(h_xphi, h_phiphi, mu, local_step)
    exact_float = np.array([float(v) for v in exact_result])
    np.testing.assert_allclose(exact_float, cholesky_result, atol=1e-12)


def test_exact_rational_schur_complement_matches_cholesky_path() -> None:
    h_xx = np.array([[4.0, 0.5], [0.5, 3.0]])
    h_xphi = np.array([[0.6, -0.2], [0.1, 0.3]])
    h_phiphi = np.array([[1.4, 0.1], [0.1, 0.9]])
    mu = 0.5
    cholesky_result = ref.independent_schur_complement(h_xx, h_xphi, h_phiphi, mu)
    exact_result = ref.exact_rational_schur_complement(h_xx, h_xphi, h_phiphi, mu)
    exact_float = np.array([[float(v) for v in row] for row in exact_result])
    np.testing.assert_allclose(exact_float, cholesky_result, atol=1e-12)


# ---------------------------------------------------------------------------
# Trust-region subproblem
# ---------------------------------------------------------------------------


def test_trust_region_eigen_boundary_active_matches_hand_solution() -> None:
    gradient = np.array([-2.0, 0.0])
    hessian = np.eye(2)
    metric = np.eye(2)
    radius = 0.5
    result = ref.independent_trust_region_eigen(
        gradient=gradient, hessian=hessian, metric=metric, radius=radius
    )
    np.testing.assert_allclose(result.step, [0.5, 0.0], atol=1e-10)
    assert result.boundary_active is True
    assert result.attainable_gain == pytest.approx(0.875, abs=1e-10)
    assert result.multiplier == pytest.approx(3.0, abs=1e-8)
    assert result.residual.stationarity < 1e-9
    assert result.residual.complementarity < 1e-9


def test_trust_region_eigen_interior_case_is_unconstrained_minimizer() -> None:
    gradient = np.array([1.0, -2.0])
    hessian = np.array([[2.0, 0.1], [0.1, 3.0]])
    metric = np.eye(2)
    radius = 5.0
    result = ref.independent_trust_region_eigen(
        gradient=gradient, hessian=hessian, metric=metric, radius=radius
    )
    expected_step = -np.linalg.solve(hessian, gradient)
    np.testing.assert_allclose(result.step, expected_step, atol=1e-10)
    assert result.boundary_active is False
    assert result.multiplier == 0.0


def test_trust_region_eigen_rejects_indefinite_hessian() -> None:
    gradient = np.array([1.0, 0.0])
    hessian = np.array([[1.0, 0.0], [0.0, -1.0]])
    metric = np.eye(2)
    with pytest.raises(ValueError, match="positive semidefinite"):
        ref.independent_trust_region_eigen(
            gradient=gradient, hessian=hessian, metric=metric, radius=1.0
        )


def test_trust_region_eigen_rejects_non_pd_metric() -> None:
    gradient = np.array([1.0, 0.0])
    hessian = np.eye(2)
    metric = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="positive definite"):
        ref.independent_trust_region_eigen(
            gradient=gradient, hessian=hessian, metric=metric, radius=1.0
        )


def test_trust_region_eigen_rejects_mismatched_shapes() -> None:
    gradient = np.array([1.0, 0.0, 0.0])
    hessian = np.eye(2)
    metric = np.eye(2)
    with pytest.raises(ValueError, match="matching shapes"):
        ref.independent_trust_region_eigen(
            gradient=gradient, hessian=hessian, metric=metric, radius=1.0
        )


def test_trust_region_eigen_rejects_non_positive_radius() -> None:
    gradient = np.array([1.0, 0.0])
    hessian = np.eye(2)
    metric = np.eye(2)
    with pytest.raises(ValueError, match="radius"):
        ref.independent_trust_region_eigen(
            gradient=gradient, hessian=hessian, metric=metric, radius=0.0
        )


def test_trust_region_blackbox_matches_eigen_on_boundary_case() -> None:
    gradient = np.array([-2.0, 0.0])
    hessian = np.eye(2)
    metric = np.eye(2)
    radius = 0.5
    eigen = ref.independent_trust_region_eigen(
        gradient=gradient, hessian=hessian, metric=metric, radius=radius
    )
    blackbox = ref.independent_trust_region_blackbox(
        gradient=gradient, hessian=hessian, metric=metric, radius=radius
    )
    np.testing.assert_allclose(blackbox.step, eigen.step, atol=1e-3)
    assert blackbox.multiplier == pytest.approx(eigen.multiplier, abs=1e-2)
    assert blackbox.residual.stationarity < 1e-6


def test_trust_region_blackbox_matches_eigen_on_interior_case() -> None:
    gradient = np.array([1.0, -2.0])
    hessian = np.array([[2.0, 0.1], [0.1, 3.0]])
    metric = np.eye(2)
    radius = 5.0
    eigen = ref.independent_trust_region_eigen(
        gradient=gradient, hessian=hessian, metric=metric, radius=radius
    )
    blackbox = ref.independent_trust_region_blackbox(
        gradient=gradient, hessian=hessian, metric=metric, radius=radius
    )
    np.testing.assert_allclose(blackbox.step, eigen.step, atol=1e-4)
    assert blackbox.boundary_active is False


# ---------------------------------------------------------------------------
# Negotiation
# ---------------------------------------------------------------------------


def test_negotiation_solve_two_task_balanced_case() -> None:
    gradients = [np.array([-2.0, 0.0]), np.array([0.0, -1.0])]
    hessians = [np.eye(2), np.eye(2)]
    metric = np.eye(2)
    radius = 0.5
    epsilons = [1e-8, 2e-8]
    attainable = np.array(
        [
            ref.independent_trust_region_eigen(
                gradient=g, hessian=h, metric=metric, radius=radius
            ).attainable_gain
            for g, h in zip(gradients, hessians)
        ]
    )
    result = ref.independent_negotiation_solve(
        gradients=gradients,
        hessians=hessians,
        metric=metric,
        radius=radius,
        epsilons=epsilons,
        attainable_gains=attainable,
    )
    # At the max-min optimum both tasks' retained gains should be (nearly)
    # equal, and the reference's own KKT residual should vanish.
    assert result.retained_gains[0] == pytest.approx(
        result.retained_gains[1], abs=1e-3
    )
    assert result.residual.stationarity < 1e-6
    assert result.residual.complementarity < 1e-3


def test_negotiation_solve_rejects_too_few_tasks() -> None:
    with pytest.raises(ValueError, match="matching gradients"):
        ref.independent_negotiation_solve(
            gradients=[np.array([1.0])],
            hessians=[np.eye(1)],
            metric=np.eye(1),
            radius=1.0,
            epsilons=[1e-6],
            attainable_gains=np.array([1.0]),
        )


def test_negotiation_solve_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="matching gradients"):
        ref.independent_negotiation_solve(
            gradients=[np.array([1.0]), np.array([1.0])],
            hessians=[np.eye(1)],
            metric=np.eye(1),
            radius=1.0,
            epsilons=[1e-6, 1e-6],
            attainable_gains=np.array([1.0, 1.0]),
        )


# ---------------------------------------------------------------------------
# Late-binding closure regression (see review history / CHANGELOG): every
# task's constraint must depend on *its own* gradient and Hessian, not the
# last task's, even though only ``idx``/``d`` were captured as explicit
# default arguments in an earlier draft.
# ---------------------------------------------------------------------------


def test_negotiation_solve_per_task_gradients_are_not_aliased() -> None:
    # Two tasks with very different gradients: if task constraints were
    # accidentally sharing one task's gradient/hessian (the late-binding
    # closure bug fixed during T120 development), the solution would ignore
    # one task's true objective and the two retained gains would disagree
    # sharply, and/or the KKT stationarity residual would be large.
    gradients = [np.array([-3.0, 0.0]), np.array([0.0, -0.2])]
    hessians = [np.eye(2), np.eye(2)]
    metric = np.eye(2)
    radius = 0.5
    epsilons = [1e-6, 1e-6]
    attainable = np.array(
        [
            ref.independent_trust_region_eigen(
                gradient=g, hessian=h, metric=metric, radius=radius
            ).attainable_gain
            for g, h in zip(gradients, hessians)
        ]
    )
    result = ref.independent_negotiation_solve(
        gradients=gradients,
        hessians=hessians,
        metric=metric,
        radius=radius,
        epsilons=epsilons,
        attainable_gains=attainable,
    )
    assert result.residual.stationarity < 1e-4
    assert result.retained_gains[0] == pytest.approx(
        result.retained_gains[1], abs=1e-2
    )
