"""Generate seeded overlap-family cases and verify them against independent references.

Three independent verification legs, one per objective in
``tasks/T110-overlap-family.md``:

1. **Block lifting** -- ``QuadraticTask.lifted_gradient()`` and the raw
   shared-curvature lift ``selector.T @ h_xx @ selector`` are each compared
   against an independently coded scatter (index assignment, not matrix
   multiplication) of the same local blocks into global coordinates.
2. **Objective changes** -- the Schur-complement identity
   ``compensated_change(step) == direct_change(step, private_response(step))``
   is checked (this identity already lives in ``comppareto.quadratic``, so
   this is a numerical re-verification across many random cases, not a new
   claim), *and* ``private_response``/``compensated_change`` are separately
   cross-checked against an independent linear conjugate-gradient solve of
   the private stationarity system (``scipy.sparse.linalg.cg``) -- a Krylov
   iterative algorithm, not the closed-form direct ``numpy.linalg.solve``
   used internally. (An initial version of this check used scipy's
   *nonlinear* CG minimizer -- Polak-Ribiere with an inexact line search --
   which stalled short of the closed-form solution on ~9% of a 450-case
   smoke sweep whenever the private curvature's condition number was large;
   those were verification-code convergence artifacts, not
   ``quadratic.py`` bugs: the closed-form Schur identity itself was exact to
   ~1e-16 in every one of those cases. Linear CG is the correct independent
   reference for an exact SPD quadratic and has a finite-termination
   guarantee in exact arithmetic.)
3. **Safe-set relations** -- for every ordered task pair ``(i, j)`` with a
   coordinate private to ``i`` relative to ``j`` (i.e. ``S_i \\ S_j`` is
   nonempty), a perturbation supported only on ``S_i \\ S_j`` must leave
   task ``j``'s ``compensated_change`` exactly unchanged: that perturbation
   lies in task ``j``'s safe set by construction. This is vacuous (no
   perturbation directions exist) for every pair under the ``full`` regime
   and maximal (covers the whole task-``i`` block) under ``disjoint``,
   giving the two regimes' declared boundary-control behavior an executable
   check rather than decorative language.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import cg as linear_cg

from comppareto.oracle.generation import generate_coupling, generate_curvature
from comppareto.overlap.regimes import OverlapRegime, REGIMES, sample_num_tasks, sample_supports
from comppareto.quadratic import QuadraticTask

FloatArray = NDArray[np.float64]

DEFAULT_TOLERANCES: dict[str, float] = {
    "block_lift": 1e-12,
    "objective_change": 1e-9,
    "objective_change_independent": 1e-6,
    "private_response": 1e-6,
    "objective_scale_floor": 1e-9,
    "safe_set": 1e-9,
}

_REGIME_STREAM_OFFSET: dict[OverlapRegime, int] = {
    "disjoint": 0,
    "partial": 1,
    "full": 2,
}


@dataclass(frozen=True)
class TaskSpec:
    support: tuple[int, ...]
    private_dim: int
    mu: float
    condition_number: float
    coupling_rank: int
    gradient_scale: float


@dataclass(frozen=True)
class OverlapCase:
    case_index: int
    config_seed: int
    regime: OverlapRegime
    global_dim: int
    num_tasks: int
    supports: tuple[tuple[int, ...], ...]
    tasks: tuple[QuadraticTask, ...]
    task_specs: tuple[TaskSpec, ...]
    global_step: FloatArray


def _selector_from_support(support: tuple[int, ...], global_dim: int) -> FloatArray:
    local_dim = len(support)
    selector = np.zeros((local_dim, global_dim), dtype=np.float64)
    for row, col in enumerate(support):
        selector[row, col] = 1.0
    return selector


def _case_seed_sequence(
    config_seed: int, case_index: int, regime: OverlapRegime
) -> np.random.SeedSequence:
    """Independent stream per (config_seed, regime, case_index).

    Keyed on the regime's fixed integer offset (not ``hash()``, which is
    process-randomized for strings) so a case is exactly reproducible in
    isolation and changing one regime's generation cannot perturb another
    regime's realized cases for the same ``case_index``.
    """

    return np.random.SeedSequence([config_seed, _REGIME_STREAM_OFFSET[regime], case_index])


def generate_case(
    config_seed: int,
    case_index: int,
    regime: OverlapRegime,
    *,
    max_tasks: int = 5,
    private_dim_range: tuple[int, int] = (1, 6),
    mu_range: tuple[float, float] = (0.05, 2.0),
    condition_number_range: tuple[float, float] = (1.0, 1000.0),
    gradient_scale_range: tuple[float, float] = (0.1, 10.0),
    global_dim_cycle: int = 30,
    global_dim_floor: int = 2,
) -> OverlapCase:
    """Deterministically generate one seeded case from ``(config_seed, case_index, regime)``.

    ``global_dim`` cycles through ``[global_dim_floor, global_dim_floor + global_dim_cycle]``
    (default ``[2, 32]``, per the frozen protocol's "dimensions 2-32") as
    ``case_index`` increases, so a run of >=100 seeds per regime realizes
    every dimension in range at least three times.
    """

    seq = _case_seed_sequence(config_seed, case_index, regime)
    structure_seq, curvature_seq, coupling_seq, gradient_seq, probe_seq = seq.spawn(5)
    structure_rng = np.random.default_rng(structure_seq)
    curvature_rng = np.random.default_rng(curvature_seq)
    coupling_rng = np.random.default_rng(coupling_seq)
    gradient_rng = np.random.default_rng(gradient_seq)
    probe_rng = np.random.default_rng(probe_seq)

    global_dim = global_dim_floor + (case_index % (global_dim_cycle + 1))
    num_tasks = sample_num_tasks(structure_rng, global_dim, max_tasks=max_tasks)
    supports = sample_supports(structure_rng, regime, num_tasks, global_dim)

    tasks: list[QuadraticTask] = []
    specs: list[TaskSpec] = []
    for support in supports:
        local_dim = len(support)
        private_dim = int(curvature_rng.integers(private_dim_range[0], private_dim_range[1] + 1))
        mu = float(curvature_rng.uniform(*mu_range))
        condition_number = float(
            np.exp(
                curvature_rng.uniform(
                    np.log(condition_number_range[0]), np.log(condition_number_range[1])
                )
            )
        )
        coupling_rank = int(coupling_rng.integers(1, max(1, min(local_dim, private_dim)) + 1))
        gradient_scale = float(
            np.exp(
                gradient_rng.uniform(
                    np.log(gradient_scale_range[0]), np.log(gradient_scale_range[1])
                )
            )
        )

        h_xx = generate_curvature(curvature_rng, local_dim, condition_number)
        h_phiphi = generate_curvature(curvature_rng, private_dim, condition_number)
        h_xphi = generate_coupling(coupling_rng, local_dim, private_dim, coupling_rank)
        local_gradient = gradient_scale * gradient_rng.standard_normal(local_dim)
        selector = _selector_from_support(support, global_dim)

        task = QuadraticTask(
            local_gradient=local_gradient,
            h_xx=h_xx,
            h_xphi=h_xphi,
            h_phiphi=h_phiphi,
            mu=mu,
            selector=selector,
        )
        tasks.append(task)
        specs.append(
            TaskSpec(
                support=support,
                private_dim=private_dim,
                mu=mu,
                condition_number=condition_number,
                coupling_rank=coupling_rank,
                gradient_scale=gradient_scale,
            )
        )

    global_step = probe_rng.standard_normal(global_dim)
    return OverlapCase(
        case_index=case_index,
        config_seed=config_seed,
        regime=regime,
        global_dim=global_dim,
        num_tasks=num_tasks,
        supports=supports,
        tasks=tuple(tasks),
        task_specs=tuple(specs),
        global_step=global_step,
    )


@dataclass(frozen=True)
class CheckOutcome:
    passed: bool
    error: float
    tolerance: float


def _outcome(error: float, tolerance: float) -> CheckOutcome:
    return CheckOutcome(passed=bool(error <= tolerance), error=float(error), tolerance=float(tolerance))


def independent_lift_gradient(
    support: tuple[int, ...], global_dim: int, local_gradient: FloatArray
) -> FloatArray:
    """Scatter ``local_gradient`` into global coordinates without matrix multiplication."""

    global_vector = np.zeros(global_dim, dtype=np.float64)
    for row, col in enumerate(support):
        global_vector[col] += local_gradient[row]
    return global_vector


def independent_lift_hessian(
    support: tuple[int, ...], global_dim: int, h_xx: FloatArray
) -> FloatArray:
    """Scatter ``h_xx`` into a global-dimension matrix without matrix multiplication."""

    idx = np.asarray(support, dtype=np.intp)
    global_h = np.zeros((global_dim, global_dim), dtype=np.float64)
    global_h[np.ix_(idx, idx)] = h_xx
    return global_h


def independent_gather(support: tuple[int, ...], vector: FloatArray) -> FloatArray:
    """Gather the local sub-vector from ``vector`` by index, not by ``selector @ vector``."""

    idx = np.asarray(support, dtype=np.intp)
    return vector[idx]


@dataclass(frozen=True)
class LinearSolveResult:
    x: FloatArray
    success: bool
    info: int


def independent_private_optimum(
    task: QuadraticTask,
    local_step: FloatArray,
    *,
    x0: FloatArray | None = None,
    rtol: float = 1e-12,
    maxiter: int = 2000,
) -> LinearSolveResult:
    """Solve the private stationarity system ``private_curvature @ phi == -linear``
    with linear (Krylov) conjugate gradient -- a different numerical algorithm
    than the closed-form ``numpy.linalg.solve`` that
    ``QuadraticTask.private_response`` uses internally.

    Linear CG (unlike scipy's nonlinear ``minimize(..., method="CG")``, which
    uses an inexact-line-search Polak-Ribiere iteration and can stall well
    short of the optimum on ill-conditioned problems) has a finite-termination
    guarantee for an exact SPD system in exact arithmetic, so it is the
    appropriate independent reference here.
    """

    private_curvature = task.h_phiphi + task.mu * np.eye(task.h_phiphi.shape[0])
    linear = task.h_xphi.T @ local_step
    rhs = -linear

    if x0 is None:
        x0 = np.zeros_like(linear)
    x, info = linear_cg(private_curvature, rhs, x0=x0, rtol=rtol, atol=0.0, maxiter=maxiter)
    return LinearSolveResult(x=x, success=bool(info == 0), info=int(info))


@dataclass(frozen=True)
class TaskCheckResult:
    task_index: int
    block_lift_gradient: CheckOutcome
    block_lift_hessian: CheckOutcome
    objective_change_direct: CheckOutcome
    private_response_independent: CheckOutcome
    objective_change_independent: CheckOutcome
    solver_converged: bool


def verify_task(
    case: OverlapCase, task_index: int, *, tolerances: dict[str, float]
) -> TaskCheckResult:
    task = case.tasks[task_index]
    support = case.supports[task_index]
    global_step = case.global_step

    lift_grad_indep = independent_lift_gradient(support, case.global_dim, task.local_gradient)
    lift_grad_formula = task.lifted_gradient()
    grad_err = float(np.max(np.abs(lift_grad_indep - lift_grad_formula)))
    block_lift_gradient = _outcome(grad_err, tolerances["block_lift"])

    lift_hess_indep = independent_lift_hessian(support, case.global_dim, task.h_xx)
    lift_hess_formula = task.selector.T @ task.h_xx @ task.selector
    hess_err = float(np.max(np.abs(lift_hess_indep - lift_hess_formula)))
    block_lift_hessian = _outcome(hess_err, tolerances["block_lift"])

    local_step_indep = independent_gather(support, global_step)
    private = task.private_response(global_step)
    change_a = task.compensated_change(global_step)
    change_b = task.direct_change(global_step, private)
    direct_err = abs(change_a - change_b) / (abs(change_a) + tolerances["objective_scale_floor"])
    objective_change_direct = _outcome(direct_err, tolerances["objective_change"])

    solved = independent_private_optimum(task, local_step_indep)
    private_err = float(np.max(np.abs(solved.x - private))) / (
        float(np.max(np.abs(private))) + tolerances["objective_scale_floor"]
    )
    private_response_independent = _outcome(private_err, tolerances["private_response"])

    change_c = task.direct_change(global_step, solved.x)
    change_err = abs(change_a - change_c) / (abs(change_a) + tolerances["objective_scale_floor"])
    objective_change_independent = _outcome(change_err, tolerances["objective_change_independent"])

    return TaskCheckResult(
        task_index=task_index,
        block_lift_gradient=block_lift_gradient,
        block_lift_hessian=block_lift_hessian,
        objective_change_direct=objective_change_direct,
        private_response_independent=private_response_independent,
        objective_change_independent=objective_change_independent,
        solver_converged=bool(solved.success),
    )


@dataclass(frozen=True)
class SafeSetPairResult:
    task_i: int
    task_j: int
    shared_size: int
    private_to_i_size: int
    invariance: CheckOutcome | None


def verify_safe_set_pair(
    case: OverlapCase,
    task_i: int,
    task_j: int,
    rng: np.random.Generator,
    *,
    tolerance: float,
) -> SafeSetPairResult:
    support_i = set(case.supports[task_i])
    support_j = set(case.supports[task_j])
    shared = support_i & support_j
    private_to_i = sorted(support_i - support_j)

    if not private_to_i:
        return SafeSetPairResult(
            task_i=task_i,
            task_j=task_j,
            shared_size=len(shared),
            private_to_i_size=0,
            invariance=None,
        )

    delta = np.zeros(case.global_dim, dtype=np.float64)
    delta[private_to_i] = rng.standard_normal(len(private_to_i))
    task_j_obj = case.tasks[task_j]
    before = task_j_obj.compensated_change(case.global_step)
    after = task_j_obj.compensated_change(case.global_step + delta)
    error = abs(after - before)
    return SafeSetPairResult(
        task_i=task_i,
        task_j=task_j,
        shared_size=len(shared),
        private_to_i_size=len(private_to_i),
        invariance=_outcome(error, tolerance),
    )


@dataclass(frozen=True)
class CaseResult:
    case: OverlapCase
    task_checks: tuple[TaskCheckResult, ...]
    safe_set_pairs: tuple[SafeSetPairResult, ...]
    all_passed: bool


def verify_case(
    case: OverlapCase,
    *,
    tolerances: dict[str, float] = DEFAULT_TOLERANCES,
) -> CaseResult:
    task_checks = tuple(
        verify_task(case, index, tolerances=tolerances) for index in range(case.num_tasks)
    )

    pair_seq = np.random.SeedSequence(
        [case.config_seed, _REGIME_STREAM_OFFSET[case.regime], case.case_index, 0xA5AFE5E7]
    )
    pair_rng = np.random.default_rng(pair_seq)
    pairs: list[SafeSetPairResult] = []
    for task_i in range(case.num_tasks):
        for task_j in range(case.num_tasks):
            if task_i == task_j:
                continue
            pairs.append(
                verify_safe_set_pair(case, task_i, task_j, pair_rng, tolerance=tolerances["safe_set"])
            )

    task_checks_passed = all(
        tc.block_lift_gradient.passed
        and tc.block_lift_hessian.passed
        and tc.objective_change_direct.passed
        and tc.private_response_independent.passed
        and tc.objective_change_independent.passed
        for tc in task_checks
    )
    safe_set_passed = all(pair.invariance is None or pair.invariance.passed for pair in pairs)

    return CaseResult(
        case=case,
        task_checks=task_checks,
        safe_set_pairs=tuple(pairs),
        all_passed=task_checks_passed and safe_set_passed,
    )
