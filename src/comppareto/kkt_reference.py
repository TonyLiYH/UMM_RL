"""Independent KKT / direct-solve reference for the compensation-aware quadratic.

Frozen protocol (see ``tasks/T120-independent-kkt-reference.md``): this module
must not import or call the production solution functions under test in
``comppareto.quadratic`` -- ``QuadraticTask.private_response``,
``QuadraticTask.schur``, ``QuadraticTask.direct_change``,
``QuadraticTask.compensated_change``, ``trust_region_optimum``, or
``negotiate_retained_gain``. It never imports ``comppareto.quadratic``.
Every quantity below is re-derived directly from the raw problem arrays
(``h_xx``, ``h_xphi``, ``h_phiphi``, ``mu``, gradients, selectors) using
numerical routines chosen to be a *different code path* from production, not
merely a re-typed copy of the same algorithm:

- Private-response stationarity and the Schur complement are solved with
  ``scipy.linalg.cho_factor``/``cho_solve`` (Cholesky, LAPACK
  ``potrf``/``potrs``), instead of production's ``numpy.linalg.solve`` (LU,
  LAPACK ``gesv``).
- Fixed-case, exactly-rational problems additionally get a from-scratch exact
  Gauss-Jordan solve over ``fractions.Fraction`` with no floating point and no
  numpy/scipy involvement at all -- the strongest available independence
  argument for the cases where it applies.
- The ellipsoidal trust-region subproblem has *two* independent references:
  (1) the classical generalized-eigenvalue / secular-equation method
  (Gander-Golub-von Matt), using ``scipy.linalg.eigh(H, M)`` to
  simultaneously diagonalize in the ``M`` metric and ``scipy.optimize.brentq``
  on the secular equation, and (2) a black-box interior-point/SQP solve
  (``scipy.optimize.minimize(method="trust-constr")``) directly on the
  original, untransformed variables with an explicit nonlinear ellipsoid
  constraint. These two share no code with each other or with production's
  Cholesky-transform-plus-manual-bisection loop.
- The max-min retained-gain negotiation is solved with
  ``scipy.optimize.minimize(method="trust-constr")`` -- an interior-point SQP
  distinct from production's SLSQP -- and every returned solution is
  independently certified by evaluating the KKT stationarity,
  complementary-slackness, and primal-feasibility residuals of the *original*
  problem directly, using the solver's returned Lagrange-multiplier
  estimates. The solver's own ``success`` flag is never trusted alone.

Honesty note on the trust-region eigen reference: the generalized-eigenvalue
secular equation and production's Cholesky-transform secular equation are the
same underlying mathematical object (the trust-region secular equation),
viewed in two different bases. They are implementation-independent (different
LAPACK entry points, different code, no shared closures) and therefore still
catch algebraic/implementation bugs in either path, but they are not two
mathematically unrelated theories of the same problem. The trust-constr
black-box cross-check has no such caveat: it is a fully separate algorithm
family (primal-dual interior point / SQP on the original coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import NonlinearConstraint, brentq, minimize

FloatArray = NDArray[np.float64]

__all__ = [
    "KKTResidual",
    "TrustRegionReference",
    "NegotiationReference",
    "independent_private_response",
    "independent_schur_complement",
    "exact_rational_solve",
    "exact_rational_private_response",
    "exact_rational_schur_complement",
    "independent_trust_region_eigen",
    "independent_trust_region_blackbox",
    "independent_negotiation_solve",
]


def _array(value: NDArray[np.floating], *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


# ---------------------------------------------------------------------------
# 1. Private-response stationarity and Schur complement (Cholesky, scipy.linalg)
# ---------------------------------------------------------------------------


def independent_private_response(
    h_xphi: NDArray[np.floating],
    h_phiphi: NDArray[np.floating],
    mu: float,
    local_step: NDArray[np.floating],
) -> FloatArray:
    """Solve ``(h_phiphi + mu*I) u = -h_xphi^T local_step`` via Cholesky.

    This is the exact stationary point of the unconstrained convex quadratic
    in the private displacement ``u`` for a fixed shared displacement
    ``local_step``: the gradient of ``J_i`` with respect to ``phi_i`` is
    ``h_xphi^T local_step + (h_phiphi + mu*I) u``, and setting it to zero
    gives this linear system (see ``docs/theory/formulation.md`` section 3).
    """

    h_xphi = _array(h_xphi, name="h_xphi")
    h_phiphi = _array(h_phiphi, name="h_phiphi")
    local_step = _array(local_step, name="local_step")
    private_dim = h_phiphi.shape[0]
    curvature = h_phiphi + mu * np.eye(private_dim)
    rhs = -(h_xphi.T @ local_step)
    factor = cho_factor(curvature, lower=True)
    return np.asarray(cho_solve(factor, rhs), dtype=np.float64)


def independent_schur_complement(
    h_xx: NDArray[np.floating],
    h_xphi: NDArray[np.floating],
    h_phiphi: NDArray[np.floating],
    mu: float,
) -> FloatArray:
    """Return ``h_xx - h_xphi (h_phiphi + mu*I)^{-1} h_xphi^T`` via Cholesky."""

    h_xx = _array(h_xx, name="h_xx")
    h_xphi = _array(h_xphi, name="h_xphi")
    h_phiphi = _array(h_phiphi, name="h_phiphi")
    private_dim = h_phiphi.shape[0]
    curvature = h_phiphi + mu * np.eye(private_dim)
    factor = cho_factor(curvature, lower=True)
    correction = h_xphi @ cho_solve(factor, h_xphi.T)
    return h_xx - correction


# ---------------------------------------------------------------------------
# 2. Exact rational reference (no numpy/scipy at all) for fixed rational cases
# ---------------------------------------------------------------------------


def exact_rational_solve(
    matrix: list[list[Fraction]], rhs: list[Fraction]
) -> list[Fraction]:
    """Solve ``matrix @ x = rhs`` exactly by Gauss-Jordan elimination.

    Every entry must already be (or be losslessly convertible to)
    ``fractions.Fraction``. No floating-point arithmetic and no numpy/scipy
    call occurs anywhere in this function.
    """

    n = len(matrix)
    if any(len(row) != n for row in matrix) or len(rhs) != n:
        raise ValueError("matrix must be square and match the right-hand side")
    augmented = [
        [Fraction(matrix[row][col]) for col in range(n)] + [Fraction(rhs[row])]
        for row in range(n)
    ]
    for col in range(n):
        pivot_row = next(
            (row for row in range(col, n) if augmented[row][col] != 0), None
        )
        if pivot_row is None:
            raise ValueError("matrix is singular in exact rational solve")
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]
        for row in range(n):
            if row != col and augmented[row][col] != 0:
                factor = augmented[row][col]
                augmented[row] = [
                    a - factor * b for a, b in zip(augmented[row], augmented[col])
                ]
    return [row[n] for row in augmented]


def _to_fraction_matrix(value: NDArray[np.floating]) -> list[list[Fraction]]:
    return [[Fraction(str(entry)) for entry in row] for row in value.tolist()]


def _to_fraction_vector(value: NDArray[np.floating]) -> list[Fraction]:
    return [Fraction(str(entry)) for entry in value.tolist()]


def exact_rational_private_response(
    h_xphi: NDArray[np.floating],
    h_phiphi: NDArray[np.floating],
    mu: float,
    local_step: NDArray[np.floating],
) -> list[Fraction]:
    """Exact rational counterpart of :func:`independent_private_response`.

    Intended for fixed cases whose inputs are exact decimal literals (so
    ``Fraction(str(x))`` round-trips exactly); random floating-point cases
    should use the Cholesky reference instead.
    """

    h_xphi = _array(h_xphi, name="h_xphi")
    h_phiphi = _array(h_phiphi, name="h_phiphi")
    local_step = _array(local_step, name="local_step")
    private_dim = h_phiphi.shape[0]
    mu_fraction = Fraction(str(mu))
    curvature = _to_fraction_matrix(h_phiphi)
    for index in range(private_dim):
        curvature[index][index] += mu_fraction
    h_xphi_fraction = _to_fraction_matrix(h_xphi)
    local_step_fraction = _to_fraction_vector(local_step)
    rhs = [
        -sum(
            h_xphi_fraction[row][col] * local_step_fraction[row]
            for row in range(h_xphi.shape[0])
        )
        for col in range(private_dim)
    ]
    return exact_rational_solve(curvature, rhs)


def exact_rational_schur_complement(
    h_xx: NDArray[np.floating],
    h_xphi: NDArray[np.floating],
    h_phiphi: NDArray[np.floating],
    mu: float,
) -> list[list[Fraction]]:
    """Exact rational counterpart of :func:`independent_schur_complement`."""

    h_xx = _array(h_xx, name="h_xx")
    h_xphi = _array(h_xphi, name="h_xphi")
    h_phiphi = _array(h_phiphi, name="h_phiphi")
    local_dim = h_xx.shape[0]
    private_dim = h_phiphi.shape[0]
    mu_fraction = Fraction(str(mu))
    curvature = _to_fraction_matrix(h_phiphi)
    for index in range(private_dim):
        curvature[index][index] += mu_fraction
    h_xphi_fraction = _to_fraction_matrix(h_xphi)
    h_xx_fraction = _to_fraction_matrix(h_xx)

    # Solve curvature @ X = h_xphi^T column by column (X is private x local).
    columns: list[list[Fraction]] = []
    for local_index in range(local_dim):
        rhs = [h_xphi_fraction[local_index][p] for p in range(private_dim)]
        columns.append(exact_rational_solve(curvature, rhs))
    # columns[j][p] = X[p, j]; correction[i, j] = sum_p h_xphi[i, p] * X[p, j]
    result: list[list[Fraction]] = []
    for i in range(local_dim):
        row: list[Fraction] = []
        for j in range(local_dim):
            correction = sum(
                h_xphi_fraction[i][p] * columns[j][p] for p in range(private_dim)
            )
            row.append(h_xx_fraction[i][j] - correction)
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# 3. Trust-region subproblem: two independent references
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KKTResidual:
    """Independently evaluated KKT residual of a returned candidate point."""

    stationarity: float
    complementarity: float
    primal_feasibility: float
    dual_feasibility: float


@dataclass(frozen=True)
class TrustRegionReference:
    step: FloatArray
    objective_change: float
    attainable_gain: float
    boundary_active: bool
    multiplier: float
    residual: KKTResidual
    method: str


def _validate_symmetric(matrix: FloatArray, name: str) -> FloatArray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-9, rtol=1e-9):
        raise ValueError(f"{name} must be symmetric")
    return matrix


def independent_trust_region_eigen(
    *,
    gradient: NDArray[np.floating],
    hessian: NDArray[np.floating],
    metric: NDArray[np.floating],
    radius: float,
    tolerance: float = 1e-12,
) -> TrustRegionReference:
    """Solve ``min g^T x + 1/2 x^T H x`` s.t. ``x^T M x <= r^2`` by the
    generalized-eigenvalue secular-equation method (Gander-Golub-von Matt).

    ``scipy.linalg.eigh(H, M)`` returns eigenvalues ``w`` (ascending) and an
    ``M``-orthonormal eigenvector matrix ``V`` (``V^T M V = I``,
    ``V^T H V = diag(w)``). In the transformed coordinates ``x = V y`` the
    constraint becomes the *unweighted* ball ``||y|| <= r`` and the objective
    separates coordinate-wise, so the boundary solution is the root of the
    scalar secular equation ``sum_k c_k^2 / (w_k + lambda)^2 = r^2`` with
    ``c = V^T g``, found by bisection (``scipy.optimize.brentq``).
    """

    g = _array(gradient, name="gradient")
    h = _validate_symmetric(_array(hessian, name="hessian"), "hessian")
    m = _validate_symmetric(_array(metric, name="metric"), "metric")
    if g.ndim != 1 or h.shape != (g.size, g.size) or m.shape != (g.size, g.size):
        raise ValueError("gradient, hessian, and metric must have matching shapes")
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and positive")
    if np.min(np.linalg.eigvalsh(m)) <= 0:
        raise ValueError("metric must be positive definite")
    if np.min(np.linalg.eigvalsh(h)) < -tolerance:
        raise ValueError("hessian must be positive semidefinite")

    eigenvalues, eigenvectors = eigh(h, m)
    c = eigenvectors.T @ g

    def secular(lam: float) -> float:
        denom = eigenvalues + lam
        return float(np.sum((c**2) / (denom**2)) - radius**2)

    w_min = float(eigenvalues[0])
    interior_multiplier = max(0.0, -w_min)
    if interior_multiplier == 0.0:
        # Candidate unconstrained/interior minimizer.
        safe = np.where(np.abs(eigenvalues) < tolerance, 1.0, eigenvalues)
        y_interior = -c / safe
        y_interior = np.where(np.abs(eigenvalues) < tolerance, 0.0, y_interior)
        if np.linalg.norm(y_interior) <= radius + tolerance:
            step = eigenvectors @ y_interior
            change = float(g @ step + 0.5 * step @ h @ step)
            residual = _trust_region_residual(
                g, h, m, step, multiplier=0.0, radius=radius
            )
            return TrustRegionReference(
                step=step,
                objective_change=change,
                attainable_gain=max(0.0, -change),
                boundary_active=False,
                multiplier=0.0,
                residual=residual,
                method="eigen-secular",
            )

    lower = interior_multiplier + tolerance
    upper = max(lower * 2.0, 1.0)
    while secular(upper) > 0:
        upper *= 2.0
        if upper > 1e18:
            raise RuntimeError("secular equation bracket search diverged")
    lam = brentq(secular, lower, upper, xtol=1e-14, rtol=1e-14, maxiter=500)
    denom = eigenvalues + lam
    y = -c / denom
    step = eigenvectors @ y
    change = float(g @ step + 0.5 * step @ h @ step)
    residual = _trust_region_residual(g, h, m, step, multiplier=lam, radius=radius)
    return TrustRegionReference(
        step=step,
        objective_change=change,
        attainable_gain=max(0.0, -change),
        boundary_active=True,
        multiplier=float(lam),
        residual=residual,
        method="eigen-secular",
    )


def _trust_region_residual(
    gradient: FloatArray,
    hessian: FloatArray,
    metric: FloatArray,
    step: FloatArray,
    *,
    multiplier: float,
    radius: float,
) -> KKTResidual:
    stationarity = float(
        np.linalg.norm(hessian @ step + multiplier * metric @ step + gradient)
    )
    constraint_value = float(step @ metric @ step - radius**2)
    complementarity = abs(multiplier * constraint_value)
    primal_feasibility = max(0.0, constraint_value)
    dual_feasibility = max(0.0, -multiplier)
    return KKTResidual(
        stationarity=stationarity,
        complementarity=complementarity,
        primal_feasibility=primal_feasibility,
        dual_feasibility=dual_feasibility,
    )


def independent_trust_region_blackbox(
    *,
    gradient: NDArray[np.floating],
    hessian: NDArray[np.floating],
    metric: NDArray[np.floating],
    radius: float,
    tolerance: float = 1e-10,
) -> TrustRegionReference:
    """Solve the same trust-region subproblem with a black-box interior-point
    SQP solver (``scipy.optimize.minimize(method="trust-constr")``) acting
    directly on the original, untransformed variables -- a fully separate
    algorithm family from both production and :func:`independent_trust_region_eigen`.
    """

    g = _array(gradient, name="gradient")
    h = _validate_symmetric(_array(hessian, name="hessian"), "hessian")
    m = _validate_symmetric(_array(metric, name="metric"), "metric")
    dim = g.size

    def objective(x: FloatArray) -> float:
        return float(g @ x + 0.5 * x @ h @ x)

    def objective_grad(x: FloatArray) -> FloatArray:
        return g + h @ x

    def objective_hess(_x: FloatArray) -> FloatArray:
        return h

    def constraint_value(x: FloatArray) -> FloatArray:
        return np.array([x @ m @ x])

    def constraint_jac(x: FloatArray) -> FloatArray:
        return (2.0 * m @ x).reshape(1, dim)

    def constraint_hess(_x: FloatArray, v: FloatArray) -> FloatArray:
        return 2.0 * v[0] * m

    nonlinear_constraint = NonlinearConstraint(
        constraint_value,
        -np.inf,
        radius**2,
        jac=constraint_jac,
        hess=constraint_hess,
    )
    result = minimize(
        objective,
        np.zeros(dim),
        method="trust-constr",
        jac=objective_grad,
        hess=objective_hess,
        constraints=[nonlinear_constraint],
        options={"gtol": 1e-14, "xtol": 1e-16, "barrier_tol": 1e-14, "maxiter": 5000},
    )
    if not result.success:
        raise RuntimeError(f"black-box trust-region solve failed: {result.message}")
    step = np.asarray(result.x, dtype=np.float64)
    # NonlinearConstraint's multiplier is defined against the constraint
    # value x^T M x directly (gradient 2*M@x); the (H + lambda*M) stationarity
    # form used elsewhere in this module wants the coefficient of M@x alone,
    # so the raw multiplier must be doubled.
    raw_multiplier = float(result.v[0][0]) if len(result.v[0]) else 0.0
    multiplier = 2.0 * raw_multiplier
    change = float(g @ step + 0.5 * step @ h @ step)
    boundary_active = bool(step @ m @ step >= radius**2 - 1e-6)
    residual = _trust_region_residual(
        g, h, m, step, multiplier=max(0.0, multiplier), radius=radius
    )
    return TrustRegionReference(
        step=step,
        objective_change=change,
        attainable_gain=max(0.0, -change),
        boundary_active=boundary_active,
        multiplier=multiplier,
        residual=residual,
        method="trust-constr-blackbox",
    )


# ---------------------------------------------------------------------------
# 4. Max-min retained-gain negotiation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NegotiationReference:
    step: FloatArray
    tau: float
    attainable_gains: FloatArray
    retained_gains: FloatArray
    residual: KKTResidual
    method: str


def independent_negotiation_solve(
    *,
    gradients: list[NDArray[np.floating]],
    hessians: list[NDArray[np.floating]],
    metric: NDArray[np.floating],
    radius: float,
    epsilons: list[float],
    attainable_gains: NDArray[np.floating],
    penalty: float = 0.0,
) -> NegotiationReference:
    """Solve the max-min retained-gain diagnostic independently.

    ``attainable_gains`` must already be computed by an independent
    trust-region reference (:func:`independent_trust_region_eigen` or
    :func:`independent_trust_region_blackbox`), not by production's
    ``trust_region_optimum`` -- this function only supplies an independent
    *negotiation* solve given those attainable gains. Uses
    ``scipy.optimize.minimize(method="trust-constr")``, a different SQP
    family from production's SLSQP, and independently recomputes the KKT
    residual of the returned point rather than trusting the solver's
    ``success`` flag.
    """

    if len(gradients) < 2 or len(hessians) != len(gradients) or len(epsilons) != len(
        gradients
    ):
        raise ValueError("provide matching gradients, hessians, and epsilons")
    m = _validate_symmetric(_array(metric, name="metric"), "metric")
    global_dim = gradients[0].size
    num_tasks = len(gradients)
    denominators = np.asarray(attainable_gains, dtype=np.float64) + np.asarray(
        epsilons, dtype=np.float64
    )

    def local_change(index: int, step: FloatArray) -> float:
        return float(gradients[index] @ step + 0.5 * step @ hessians[index] @ step)

    def objective(z: FloatArray) -> float:
        step = z[:-1]
        tau = z[-1]
        return float(-tau + 0.5 * penalty * step @ m @ step)

    def objective_grad(z: FloatArray) -> FloatArray:
        step = z[:-1]
        grad = np.zeros_like(z)
        grad[:-1] = penalty * (m @ step)
        grad[-1] = -1.0
        return grad

    def objective_hess(_z: FloatArray) -> FloatArray:
        hess = np.zeros((global_dim + 1, global_dim + 1))
        hess[:-1, :-1] = penalty * m
        return hess

    def ellipsoid_value(z: FloatArray) -> FloatArray:
        step = z[:-1]
        return np.array([step @ m @ step])

    def ellipsoid_jac(z: FloatArray) -> FloatArray:
        step = z[:-1]
        row = np.zeros(global_dim + 1)
        row[:-1] = 2.0 * (m @ step)
        return row.reshape(1, -1)

    def ellipsoid_hess(_z: FloatArray, v: FloatArray) -> FloatArray:
        hess = np.zeros((global_dim + 1, global_dim + 1))
        hess[:-1, :-1] = 2.0 * v[0] * m
        return hess

    constraints = [
        NonlinearConstraint(
            ellipsoid_value,
            -np.inf,
            radius**2,
            jac=ellipsoid_jac,
            hess=ellipsoid_hess,
        )
    ]

    for index in range(num_tasks):
        denom = float(denominators[index])
        hess_i = hessians[index]
        grad_i = gradients[index]

        def task_value(
            z: FloatArray, idx: int = index, d: float = denom
        ) -> FloatArray:
            step = z[:-1]
            tau = z[-1]
            return np.array([local_change(idx, step) / d + tau])

        def task_jac(
            z: FloatArray,
            d: float = denom,
            g_i: FloatArray = grad_i,
            h_i: FloatArray = hess_i,
        ) -> FloatArray:
            step = z[:-1]
            row = np.zeros(global_dim + 1)
            row[:-1] = (g_i + h_i @ step) / d
            row[-1] = 1.0
            return row.reshape(1, -1)

        def task_hess(
            _z: FloatArray, v: FloatArray, d: float = denom, h_i: FloatArray = hess_i
        ) -> FloatArray:
            hess = np.zeros((global_dim + 1, global_dim + 1))
            hess[:-1, :-1] = v[0] * h_i / d
            return hess

        constraints.append(
            NonlinearConstraint(task_value, -np.inf, 0.0, jac=task_jac, hess=task_hess)
        )

    initial = np.zeros(global_dim + 1)
    result = minimize(
        objective,
        initial,
        method="trust-constr",
        jac=objective_grad,
        hess=objective_hess,
        constraints=constraints,
        options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 4000},
    )
    if not result.success:
        raise RuntimeError(f"independent negotiation solve failed: {result.message}")

    z = np.asarray(result.x, dtype=np.float64)
    step = z[:-1]
    tau = float(z[-1])
    retained = np.array(
        [-local_change(index, step) / denominators[index] for index in range(num_tasks)]
    )
    residual = _negotiation_residual(
        gradients=gradients,
        hessians=hessians,
        metric=m,
        denominators=denominators,
        step=step,
        tau=tau,
        radius=radius,
        penalty=penalty,
        multipliers=result.v,
    )
    return NegotiationReference(
        step=step,
        tau=tau,
        attainable_gains=np.asarray(attainable_gains, dtype=np.float64),
        retained_gains=retained,
        residual=residual,
        method="trust-constr-blackbox",
    )


def _negotiation_residual(
    *,
    gradients: list[FloatArray],
    hessians: list[FloatArray],
    metric: FloatArray,
    denominators: FloatArray,
    step: FloatArray,
    tau: float,
    radius: float,
    penalty: float,
    multipliers: list[FloatArray],
) -> KKTResidual:
    """Recompute the KKT residual of ``(step, tau)`` from first principles.

    ``multipliers`` is ``result.v`` from ``trust-constr``: one array per
    constraint, in the same order they were passed in (ellipsoid first, then
    one per task). This does not trust the solver's own ``success`` flag --
    it independently re-evaluates stationarity and complementary slackness.
    """

    ellipsoid_multiplier = max(0.0, float(multipliers[0][0]))
    task_multipliers = [max(0.0, float(mult[0])) for mult in multipliers[1:]]

    grad_step = penalty * (metric @ step)
    grad_step += ellipsoid_multiplier * 2.0 * (metric @ step)
    grad_tau = -1.0
    for index, mult in enumerate(task_multipliers):
        grad_step += mult * (gradients[index] + hessians[index] @ step) / denominators[index]
        grad_tau += mult
    stationarity = float(np.linalg.norm(np.append(grad_step, grad_tau)))

    ellipsoid_constraint = float(step @ metric @ step - radius**2)
    complementarity = abs(ellipsoid_multiplier * ellipsoid_constraint)
    primal_feasibility = max(0.0, ellipsoid_constraint)
    dual_feasibility = max(0.0, -ellipsoid_multiplier)
    for index, mult in enumerate(task_multipliers):
        local = float(
            gradients[index] @ step + 0.5 * step @ hessians[index] @ step
        )
        constraint_value = local / denominators[index] + tau
        complementarity += abs(mult * constraint_value)
        primal_feasibility = max(primal_feasibility, constraint_value)
        dual_feasibility = max(dual_feasibility, -mult)

    return KKTResidual(
        stationarity=stationarity,
        complementarity=complementarity,
        primal_feasibility=primal_feasibility,
        dual_feasibility=dual_feasibility,
    )
