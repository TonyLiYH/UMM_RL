"""Comparison suite: production ``comppareto.quadratic`` vs. the independent
KKT/direct-solve reference in ``comppareto.kkt_reference``.

Implements the T120 frozen-protocol comparison suite (see
``tasks/T120-independent-kkt-reference.md``): fixed hand-picked convex cases
plus seeded random convex cases, each checked for parameter difference,
objective difference, feasibility violation, and KKT-residual magnitude
against preregistered thresholds (:data:`THRESHOLDS`). This module is the
*only* place in the T120 deliverable that is allowed to import both
``comppareto.quadratic`` (production, under test) and
``comppareto.kkt_reference`` (the independent reference) -- the reference
module itself never imports production, per the frozen protocol.

Run as a script to regenerate the residual tables, summary, and failure
ledger used by the run manifest under ``runs/t1b-kkt-*/``::

    PYTHONPATH=src python3 -m comppareto.kkt_compare \\
        --seed 20260928 --out-dir runs/t1b-kkt-20260928-suite
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from comppareto import kkt_reference as ref
from comppareto import quadratic as prod

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Preregistered thresholds, tiered by algorithm-family independence.
#
# The Cholesky private-response/Schur path and the exact-rational path solve
# literally the same linear system as production (different LAPACK routine /
# no floating point at all respectively), so they are expected to agree to
# near machine precision. The generalized-eigenvalue trust-region secular
# solve is the same secular equation as production viewed in a different
# basis (see the "Honesty note" in kkt_reference.py), so it is held to a
# tight but not machine-precision bar. The trust-constr black-box solves (for
# both the trust-region subproblem and the negotiation max-min problem) are a
# fully separate algorithm family (interior-point SQP) with looser default
# solver tolerances, so they are held to a visibly looser bar; the actually
# achieved numbers are always recorded alongside the threshold so a reviewer
# can rescale if this bar is judged too loose or too tight.
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, float] = {
    "private_response_param": 1e-8,
    "schur_param": 1e-8,
    "exact_rational_param": 1e-8,
    "trust_region_eigen_param": 1e-6,
    "trust_region_eigen_objective": 1e-8,
    "trust_region_eigen_kkt_stationarity": 1e-6,
    "trust_region_eigen_kkt_complementarity": 1e-6,
    "trust_region_blackbox_param": 5e-3,
    "trust_region_blackbox_objective": 1e-3,
    "trust_region_blackbox_kkt_stationarity": 1e-4,
    "trust_region_blackbox_kkt_complementarity": 5e-4,
    "negotiation_param": 5e-3,
    "negotiation_tau": 5e-3,
    "negotiation_retained_gain": 5e-3,
    "negotiation_kkt_stationarity": 1e-4,
    "negotiation_kkt_complementarity": 1e-4,
}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    family: str
    generator: str
    checks: dict[str, float]
    thresholds: dict[str, float]
    passed: bool
    failure_reasons: list[str]
    notes: str = ""


# ---------------------------------------------------------------------------
# Fixed hand-picked cases
# ---------------------------------------------------------------------------


def _make_task(**kwargs: object) -> prod.QuadraticTask:
    return prod.QuadraticTask(**kwargs)  # type: ignore[arg-type]


def fixed_private_response_schur_cases() -> list[dict[str, object]]:
    identity2 = np.eye(2)
    return [
        {
            "case_id": "fixed-pr-basic-2d",
            "exact_rational": True,
            "local_gradient": np.array([0.4, -0.2]),
            "h_xx": np.array([[3.0, 0.4], [0.4, 2.0]]),
            "h_xphi": np.array([[0.5], [-0.3]]),
            "h_phiphi": np.array([[1.2]]),
            "mu": 0.8,
            "selector": identity2,
            "global_step": np.array([0.3, -0.1]),
            "notes": "test_quadratic.py::make_task fixture, reused verbatim",
        },
        {
            "case_id": "fixed-pr-near-singular-private-curvature",
            "exact_rational": True,
            "local_gradient": np.array([1.0, 0.5]),
            "h_xx": np.array([[2.0, 0.0], [0.0, 2.0]]),
            "h_xphi": np.array([[0.2], [0.1]]),
            "h_phiphi": np.array([[0.001]]),
            "mu": 0.0001,
            "selector": identity2,
            "global_step": np.array([0.05, -0.05]),
            "notes": "private curvature 0.0011, near-singular but PD",
        },
        {
            "case_id": "fixed-pr-indefinite-h_phiphi-regularized",
            "exact_rational": True,
            "local_gradient": np.array([-0.5, 0.2]),
            "h_xx": np.array([[1.5, -0.2], [-0.2, 1.0]]),
            "h_xphi": np.array([[0.4], [-0.4]]),
            "h_phiphi": np.array([[-0.5]]),
            "mu": 1.0,
            "selector": identity2,
            "global_step": np.array([0.2, 0.2]),
            "notes": "h_phiphi itself indefinite (-0.5); mu=1.0 regularizes to PD (0.5)",
        },
        {
            "case_id": "fixed-pr-asymmetric-selector",
            "exact_rational": False,
            "local_gradient": np.array([0.7]),
            "h_xx": np.array([[2.5]]),
            "h_xphi": np.array([[0.3, -0.1]]),
            "h_phiphi": np.array([[1.5, 0.05], [0.05, 1.1]]),
            "mu": 0.3,
            "selector": np.array([[0.0, 1.0]]),
            "global_step": np.array([0.1, -0.4]),
            "notes": "local_dim=1 selected out of global_dim=2 via a non-identity selector",
        },
        {
            "case_id": "fixed-pr-multi-private-dim",
            "exact_rational": True,
            "local_gradient": np.array([0.9, -1.1]),
            "h_xx": np.array([[4.0, 0.5], [0.5, 3.0]]),
            "h_xphi": np.array([[0.6, -0.2], [0.1, 0.3]]),
            "h_phiphi": np.array([[1.4, 0.1], [0.1, 0.9]]),
            "mu": 0.5,
            "selector": identity2,
            "global_step": np.array([-0.2, 0.3]),
            "notes": "private_dim=2 exercises multi-column exact-rational Schur",
        },
    ]


def fixed_trust_region_cases() -> list[dict[str, object]]:
    return [
        {
            "case_id": "fixed-tr-boundary-active",
            "gradient": np.array([-2.0, 0.0]),
            "hessian": np.eye(2),
            "metric": np.eye(2),
            "radius": 0.5,
            "notes": "boundary-active; mirrors test_quadratic.py boundary test",
        },
        {
            "case_id": "fixed-tr-interior",
            "gradient": np.array([1.0, -2.0]),
            "hessian": np.array([[2.0, 0.1], [0.1, 3.0]]),
            "metric": np.eye(2),
            "radius": 5.0,
            "notes": "interior (unconstrained) minimizer strictly inside the ball",
        },
        {
            "case_id": "fixed-tr-nonidentity-metric-boundary",
            "gradient": np.array([-1.0, -1.5]),
            "hessian": np.array([[1.0, 0.0], [0.0, 0.5]]),
            "metric": np.array([[1.5, 0.0], [0.0, 0.8]]),
            "radius": 0.6,
            "notes": "non-identity metric, boundary-active",
        },
        {
            "case_id": "fixed-tr-near-singular-hessian",
            "gradient": np.array([0.3, 0.1, -0.4]),
            "hessian": np.array(
                [[1e-6, 0.0, 0.0], [0.0, 2.0, 0.1], [0.0, 0.1, 1.5]]
            ),
            "metric": np.eye(3),
            "radius": 1.0,
            "notes": "hessian nearly singular along one axis (1e-6 eigenvalue)",
        },
    ]


def fixed_negotiation_cases() -> list[dict[str, object]]:
    identity2 = np.eye(2)
    two_task = {
        "case_id": "fixed-neg-two-task-balanced",
        "tasks": [
            {
                "local_gradient": np.array([-2.0, 0.0]),
                "h_xx": np.eye(2),
                "h_xphi": np.zeros((2, 1)),
                "h_phiphi": np.eye(1),
                "mu": 0.5,
                "selector": identity2,
            },
            {
                "local_gradient": np.array([0.0, -1.0]),
                "h_xx": np.eye(2),
                "h_xphi": np.zeros((2, 1)),
                "h_phiphi": np.eye(1),
                "mu": 0.5,
                "selector": identity2,
            },
        ],
        "metric": np.eye(2),
        "radius": 0.5,
        "epsilons": [1e-8, 2e-8],
        "notes": "mirrors test_quadratic.py negotiation fixture",
    }
    three_task = {
        "case_id": "fixed-neg-three-task-anisotropic",
        "tasks": [
            {
                "local_gradient": np.array([-1.5, 0.3]),
                "h_xx": np.array([[2.0, 0.1], [0.1, 1.5]]),
                "h_xphi": np.zeros((2, 1)),
                "h_phiphi": np.eye(1),
                "mu": 0.5,
                "selector": identity2,
            },
            {
                "local_gradient": np.array([0.2, -1.0]),
                "h_xx": np.array([[1.0, -0.2], [-0.2, 2.0]]),
                "h_xphi": np.zeros((2, 1)),
                "h_phiphi": np.eye(1),
                "mu": 0.5,
                "selector": identity2,
            },
            {
                "local_gradient": np.array([-0.5, -0.5]),
                "h_xx": np.eye(2) * 1.2,
                "h_xphi": np.zeros((2, 1)),
                "h_phiphi": np.eye(1),
                "mu": 0.5,
                "selector": identity2,
            },
        ],
        "metric": np.array([[1.5, 0.0], [0.0, 0.8]]),
        "radius": 0.7,
        "epsilons": [1e-6, 1e-6, 1e-6],
        "notes": "three tasks, anisotropic hessians and metric",
    }
    return [two_task, three_task]


# ---------------------------------------------------------------------------
# Random convex case generators (seeded)
# ---------------------------------------------------------------------------


def _random_task_arrays(
    rng: np.random.Generator, local_dim: int, private_dim: int
) -> dict[str, FloatArray | float]:
    # Sample ONE joint PD matrix over [x; phi] and split it into blocks,
    # rather than sampling h_xx and h_phiphi independently: PD-ness of the
    # two diagonal blocks alone does NOT guarantee the Schur complement
    # h_xx - h_xphi (h_phiphi+mu*I)^-1 h_xphi^T is PSD (that only follows
    # from the *joint* block matrix being PD). Principal submatrices of a PD
    # matrix are PD, and the Schur complement of a PD matrix is PD, so this
    # construction guarantees both by design.
    total_dim = local_dim + private_dim
    c = rng.normal(size=(total_dim, total_dim))
    joint = c @ c.T + np.eye(total_dim) * float(0.5 + rng.random())
    h_xx = joint[:local_dim, :local_dim]
    h_xphi = joint[:local_dim, local_dim:]
    private_curvature_target = joint[local_dim:, local_dim:]
    mu = float(rng.uniform(0.0, 0.5))
    h_phiphi = private_curvature_target - mu * np.eye(private_dim)
    local_gradient = rng.normal(size=local_dim)
    return {
        "local_gradient": local_gradient,
        "h_xx": h_xx,
        "h_xphi": h_xphi,
        "h_phiphi": h_phiphi,
        "mu": mu,
        "selector": np.eye(local_dim),
    }


def random_private_response_schur_cases(
    rng: np.random.Generator, count: int
) -> list[dict[str, object]]:
    cases = []
    for index in range(count):
        local_dim = int(rng.integers(1, 4))
        private_dim = int(rng.integers(1, 4))
        arrays = _random_task_arrays(rng, local_dim, private_dim)
        global_step = rng.normal(size=local_dim)
        cases.append(
            {
                "case_id": f"random-pr-{index:03d}",
                "exact_rational": False,
                "global_step": global_step,
                "notes": f"local_dim={local_dim} private_dim={private_dim}",
                **arrays,
            }
        )
    return cases


def random_trust_region_cases(
    rng: np.random.Generator, count: int
) -> list[dict[str, object]]:
    cases = []
    for index in range(count):
        dim = int(rng.integers(2, 5))
        a = rng.normal(size=(dim, dim))
        hessian = a @ a.T + np.eye(dim) * 0.1
        gradient = rng.normal(size=dim) * 2.0
        m = rng.normal(size=(dim, dim))
        metric = m @ m.T + np.eye(dim) * 0.5
        radius = float(rng.uniform(0.1, 3.0))
        cases.append(
            {
                "case_id": f"random-tr-{index:03d}",
                "gradient": gradient,
                "hessian": hessian,
                "metric": metric,
                "radius": radius,
                "notes": f"dim={dim}",
            }
        )
    return cases


def random_negotiation_cases(
    rng: np.random.Generator, count: int
) -> list[dict[str, object]]:
    cases = []
    for index in range(count):
        num_tasks = int(rng.integers(2, 4))
        local_dim = int(rng.integers(2, 4))
        tasks = []
        for _ in range(num_tasks):
            private_dim = int(rng.integers(1, 3))
            arrays = _random_task_arrays(rng, local_dim, private_dim)
            tasks.append(arrays)
        m = rng.normal(size=(local_dim, local_dim))
        metric = m @ m.T + np.eye(local_dim) * 0.5
        radius = float(rng.uniform(0.2, 2.0))
        epsilons = [float(rng.uniform(1e-6, 1e-3)) for _ in range(num_tasks)]
        cases.append(
            {
                "case_id": f"random-neg-{index:03d}",
                "tasks": tasks,
                "metric": metric,
                "radius": radius,
                "epsilons": epsilons,
                "notes": f"num_tasks={num_tasks} local_dim={local_dim}",
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Comparison routines
# ---------------------------------------------------------------------------


def compare_private_response_and_schur(case: dict[str, object]) -> CaseResult:
    task = _make_task(
        local_gradient=case["local_gradient"],
        h_xx=case["h_xx"],
        h_xphi=case["h_xphi"],
        h_phiphi=case["h_phiphi"],
        mu=case["mu"],
        selector=case["selector"],
    )
    global_step = np.asarray(case["global_step"], dtype=np.float64)
    local_step = task._local_step(global_step)

    prod_private = task.private_response(global_step)
    prod_schur = task.schur()
    prod_change = task.direct_change(global_step, prod_private)
    prod_compensated = task.compensated_change(global_step)

    ref_private = ref.independent_private_response(
        task.h_xphi, task.h_phiphi, task.mu, local_step
    )
    ref_schur = ref.independent_schur_complement(
        task.h_xx, task.h_xphi, task.h_phiphi, task.mu
    )

    checks: dict[str, float] = {
        "private_response_param": float(np.max(np.abs(prod_private - ref_private))),
        "schur_param": float(np.max(np.abs(prod_schur - ref_schur))),
        "objective_direct_vs_compensated": float(abs(prod_change - prod_compensated)),
    }
    thresholds = {
        "private_response_param": THRESHOLDS["private_response_param"],
        "schur_param": THRESHOLDS["schur_param"],
        "objective_direct_vs_compensated": 1e-8,
    }

    if case.get("exact_rational"):
        exact_private = ref.exact_rational_private_response(
            task.h_xphi, task.h_phiphi, task.mu, local_step
        )
        exact_private_float = np.array([float(v) for v in exact_private])
        exact_schur = ref.exact_rational_schur_complement(
            task.h_xx, task.h_xphi, task.h_phiphi, task.mu
        )
        exact_schur_float = np.array(
            [[float(v) for v in row] for row in exact_schur]
        )
        checks["exact_rational_param"] = float(
            max(
                np.max(np.abs(prod_private - exact_private_float)),
                np.max(np.abs(prod_schur - exact_schur_float)),
            )
        )
        thresholds["exact_rational_param"] = THRESHOLDS["exact_rational_param"]

    failure_reasons = [
        name
        for name, value in checks.items()
        if name in thresholds and value > thresholds[name]
    ]
    return CaseResult(
        case_id=str(case["case_id"]),
        family="private_response_schur",
        generator="fixed" if str(case["case_id"]).startswith("fixed") else "random",
        checks=checks,
        thresholds=thresholds,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
        notes=str(case.get("notes", "")),
    )


def compare_trust_region(case: dict[str, object]) -> CaseResult:
    gradient = np.asarray(case["gradient"], dtype=np.float64)
    hessian = np.asarray(case["hessian"], dtype=np.float64)
    metric = np.asarray(case["metric"], dtype=np.float64)
    radius = float(case["radius"])

    prod_result = prod.trust_region_optimum(
        gradient=gradient, hessian=hessian, metric=metric, radius=radius
    )
    eigen = ref.independent_trust_region_eigen(
        gradient=gradient, hessian=hessian, metric=metric, radius=radius
    )
    try:
        blackbox = ref.independent_trust_region_blackbox(
            gradient=gradient, hessian=hessian, metric=metric, radius=radius
        )
        blackbox_error: str | None = None
    except RuntimeError as exc:  # solver non-convergence is a recorded result
        blackbox = None
        blackbox_error = str(exc)

    checks: dict[str, float] = {
        "trust_region_eigen_param": float(
            np.max(np.abs(prod_result.step - eigen.step))
        ),
        "trust_region_eigen_objective": float(
            abs(prod_result.objective_change - eigen.objective_change)
        ),
        "trust_region_eigen_kkt_stationarity": eigen.residual.stationarity,
        "trust_region_eigen_kkt_complementarity": eigen.residual.complementarity,
        "boundary_active_agreement": float(
            prod_result.boundary_active != eigen.boundary_active
        ),
    }
    thresholds = {
        "trust_region_eigen_param": THRESHOLDS["trust_region_eigen_param"],
        "trust_region_eigen_objective": THRESHOLDS["trust_region_eigen_objective"],
        "trust_region_eigen_kkt_stationarity": THRESHOLDS[
            "trust_region_eigen_kkt_stationarity"
        ],
        "trust_region_eigen_kkt_complementarity": THRESHOLDS[
            "trust_region_eigen_kkt_complementarity"
        ],
        "boundary_active_agreement": 0.5,
    }

    if blackbox is not None:
        # Cross-check: does the independently-derived KKT stationarity
        # residual, evaluated at *production's* step using the black-box
        # solver's multiplier estimate, also vanish? This guards against two
        # independent algorithms silently agreeing on a jointly-wrong answer.
        cross_residual = ref._trust_region_residual(
            gradient,
            hessian,
            metric,
            prod_result.step,
            multiplier=max(0.0, blackbox.multiplier),
            radius=radius,
        )
        checks["trust_region_blackbox_param"] = float(
            np.max(np.abs(prod_result.step - blackbox.step))
        )
        checks["trust_region_blackbox_objective"] = float(
            abs(prod_result.objective_change - blackbox.objective_change)
        )
        checks["trust_region_blackbox_kkt_stationarity"] = blackbox.residual.stationarity
        checks["trust_region_blackbox_kkt_complementarity"] = (
            blackbox.residual.complementarity
        )
        checks["trust_region_blackbox_cross_stationarity_at_prod_step"] = (
            cross_residual.stationarity
        )
        thresholds["trust_region_blackbox_param"] = THRESHOLDS[
            "trust_region_blackbox_param"
        ]
        thresholds["trust_region_blackbox_objective"] = THRESHOLDS[
            "trust_region_blackbox_objective"
        ]
        thresholds["trust_region_blackbox_kkt_stationarity"] = THRESHOLDS[
            "trust_region_blackbox_kkt_stationarity"
        ]
        thresholds["trust_region_blackbox_kkt_complementarity"] = THRESHOLDS[
            "trust_region_blackbox_kkt_complementarity"
        ]
        thresholds["trust_region_blackbox_cross_stationarity_at_prod_step"] = THRESHOLDS[
            "trust_region_blackbox_kkt_stationarity"
        ]
    else:
        checks["trust_region_blackbox_error"] = 1.0
        thresholds["trust_region_blackbox_error"] = 0.5

    failure_reasons = [
        name
        for name, value in checks.items()
        if name in thresholds and value > thresholds[name]
    ]
    notes = str(case.get("notes", ""))
    if blackbox_error:
        notes = f"{notes} | blackbox solver failed: {blackbox_error}"
    return CaseResult(
        case_id=str(case["case_id"]),
        family="trust_region",
        generator="fixed" if str(case["case_id"]).startswith("fixed") else "random",
        checks=checks,
        thresholds=thresholds,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
        notes=notes,
    )


def compare_negotiation(case: dict[str, object]) -> CaseResult:
    tasks = [_make_task(**task_arrays) for task_arrays in case["tasks"]]
    metric = np.asarray(case["metric"], dtype=np.float64)
    radius = float(case["radius"])
    epsilons = list(case["epsilons"])

    try:
        prod_result = prod.negotiate_retained_gain(
            tasks, metric=metric, radius=radius, epsilons=epsilons
        )
        prod_error: str | None = None
    except RuntimeError as exc:
        prod_result = None
        prod_error = f"production: {exc}"

    gradients = [task.lifted_gradient() for task in tasks]
    hessians = [task.selector.T @ task.schur() @ task.selector for task in tasks]
    # attainable_gains computed via the independent eigen trust-region
    # reference (never production's trust_region_optimum), per the docstring
    # contract of independent_negotiation_solve.
    attainable = np.array(
        [
            ref.independent_trust_region_eigen(
                gradient=gradient, hessian=hessian, metric=metric, radius=radius
            ).attainable_gain
            for gradient, hessian in zip(gradients, hessians, strict=True)
        ]
    )

    try:
        ref_result = ref.independent_negotiation_solve(
            gradients=gradients,
            hessians=hessians,
            metric=metric,
            radius=radius,
            epsilons=epsilons,
            attainable_gains=attainable,
        )
        ref_error: str | None = None
    except RuntimeError as exc:
        ref_result = None
        ref_error = f"reference: {exc}"

    checks: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    notes = str(case.get("notes", ""))

    if prod_result is not None and ref_result is not None:
        checks["negotiation_param"] = float(
            np.max(np.abs(prod_result.step - ref_result.step))
        )
        checks["negotiation_tau"] = float(abs(prod_result.tau - ref_result.tau))
        checks["negotiation_retained_gain"] = float(
            np.max(np.abs(prod_result.retained_gains - ref_result.retained_gains))
        )
        checks["negotiation_kkt_stationarity"] = ref_result.residual.stationarity
        checks["negotiation_kkt_complementarity"] = ref_result.residual.complementarity
        thresholds = {
            "negotiation_param": THRESHOLDS["negotiation_param"],
            "negotiation_tau": THRESHOLDS["negotiation_tau"],
            "negotiation_retained_gain": THRESHOLDS["negotiation_retained_gain"],
            "negotiation_kkt_stationarity": THRESHOLDS["negotiation_kkt_stationarity"],
            "negotiation_kkt_complementarity": THRESHOLDS[
                "negotiation_kkt_complementarity"
            ],
        }
    else:
        checks["solver_error"] = 1.0
        thresholds["solver_error"] = 0.5
        notes = f"{notes} | {prod_error or ''} {ref_error or ''}".strip()

    failure_reasons = [
        name
        for name, value in checks.items()
        if name in thresholds and value > thresholds[name]
    ]
    return CaseResult(
        case_id=str(case["case_id"]),
        family="negotiation",
        generator="fixed" if str(case["case_id"]).startswith("fixed") else "random",
        checks=checks,
        thresholds=thresholds,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Suite driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteResult:
    seed: int
    counts: dict[str, int]
    results: list[CaseResult]

    @property
    def all_passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> list[CaseResult]:
        return [result for result in self.results if not result.passed]


def run_comparison_suite(
    *,
    seed: int = 20260928,
    random_private_response_count: int = 40,
    random_trust_region_count: int = 40,
    random_negotiation_count: int = 20,
) -> SuiteResult:
    rng = np.random.default_rng(seed)

    pr_cases = fixed_private_response_schur_cases() + random_private_response_schur_cases(
        rng, random_private_response_count
    )
    tr_cases = fixed_trust_region_cases() + random_trust_region_cases(
        rng, random_trust_region_count
    )
    neg_cases = fixed_negotiation_cases() + random_negotiation_cases(
        rng, random_negotiation_count
    )

    results: list[CaseResult] = []
    for case in pr_cases:
        results.append(compare_private_response_and_schur(case))
    for case in tr_cases:
        results.append(compare_trust_region(case))
    for case in neg_cases:
        results.append(compare_negotiation(case))

    counts = {
        "private_response_schur": len(pr_cases),
        "trust_region": len(tr_cases),
        "negotiation": len(neg_cases),
        "total": len(results),
    }
    return SuiteResult(seed=seed, counts=counts, results=results)


def _result_to_json(result: CaseResult) -> dict[str, object]:
    return asdict(result)


def write_suite_artifacts(suite: SuiteResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_table = [_result_to_json(result) for result in suite.results]
    (out_dir / "residual-table.json").write_text(
        json.dumps(residual_table, indent=2, default=float) + "\n"
    )

    lines = [
        "case_id,family,generator,passed,failure_reasons",
    ]
    for result in suite.results:
        lines.append(
            ",".join(
                [
                    result.case_id,
                    result.family,
                    result.generator,
                    str(result.passed),
                    ";".join(result.failure_reasons),
                ]
            )
        )
    (out_dir / "residual-table.csv").write_text("\n".join(lines) + "\n")

    max_by_check: dict[str, float] = {}
    for result in suite.results:
        for name, value in result.checks.items():
            max_by_check[name] = max(max_by_check.get(name, 0.0), value)

    summary = {
        "seed": suite.seed,
        "counts": suite.counts,
        "all_passed": suite.all_passed,
        "num_failed": len(suite.failures),
        "max_observed_by_check": max_by_check,
        "thresholds": THRESHOLDS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    failure_ledger = [_result_to_json(result) for result in suite.failures]
    (out_dir / "failure-ledger.json").write_text(
        json.dumps(failure_ledger, indent=2, default=float) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/t1b-kkt-suite")
    )
    parser.add_argument("--random-private-response-count", type=int, default=40)
    parser.add_argument("--random-trust-region-count", type=int, default=40)
    parser.add_argument("--random-negotiation-count", type=int, default=20)
    args = parser.parse_args()

    suite = run_comparison_suite(
        seed=args.seed,
        random_private_response_count=args.random_private_response_count,
        random_trust_region_count=args.random_trust_region_count,
        random_negotiation_count=args.random_negotiation_count,
    )
    write_suite_artifacts(suite, args.out_dir)
    print(
        f"kkt_compare: total={suite.counts['total']} "
        f"failed={len(suite.failures)} all_passed={suite.all_passed}"
    )
    return 0 if suite.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
