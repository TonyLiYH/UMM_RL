"""One reproducible oracle case: generate a graph-family instance, choose the
stability regime, run the optimizer's finite-response simulation, and cross-
check every task against the three §7 reference methods.

Matches ``docs/theory/oracle-spec.md`` end to end; ties together
``selectors``, ``generation``, ``stability``, ``noise``, ``sgd``/``momentum``,
``hypergradient``, and ``crosscheck``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from comppareto.oracle import crosscheck as cc
from comppareto.oracle import generation as gen
from comppareto.oracle import hypergradient as hg
from comppareto.oracle import pareto
from comppareto.oracle import sgd
from comppareto.oracle import stability as st
from comppareto.oracle.momentum import (
    momentum_closed_form_state,
    momentum_input_jacobian,
    momentum_sensitivity,
    momentum_sensitivity_trajectory,
    momentum_transition_jacobian,
    momentum_unroll,
)
from comppareto.oracle.noise import NoiseModel
from comppareto.oracle.seeds import CaseSeeds, case_seeds
from comppareto.oracle.selectors import BlockLayout, GraphFamily, build_incidence

FloatArray = NDArray[np.float64]

Optimizer = Literal["sgd", "momentum"]


def selector_hash(selector: FloatArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(selector, dtype=np.float64).tobytes()).hexdigest()


@dataclass(frozen=True)
class CaseSpec:
    case_index: int
    config_seed: int
    family: GraphFamily
    num_tasks: int
    num_blocks: int
    block_width: int
    private_dims: tuple[int, ...]
    condition_number: float
    coupling_rank: int
    mu: float
    optimizer: Optimizer
    beta: float | None
    horizon: int
    stability_regime: Literal["stable", "unstable"]
    noise_kind: str
    noise_sigma: float
    noise_rho: float
    gradient_cosine_target: float | None
    gradient_scale_target: float | None
    keep_full_detail: bool = False


@dataclass(frozen=True)
class TaskCaseResult:
    task_index: int
    private_dim: int
    shared_dim: int
    selector_hash: str
    eta: float
    stability_target_or_realized: float
    stability_realized_actual: float
    condition_number_realized: float
    coupling_rank_realized: int
    checks: cc.CaseChecks
    detail: dict | None = None


@dataclass(frozen=True)
class CaseResult:
    spec: CaseSpec
    incidence_hash: str
    designated_pair: tuple[int, int] | None
    gradient_cosine_realized: float | None
    gradient_scale_realized: float | None
    tasks: tuple[TaskCaseResult, ...]
    pareto_reference: dict

    @property
    def all_passed(self) -> bool:
        return all(t.checks.all_passed for t in self.tasks)


def _resolve_block_widths(num_blocks: int, block_width: int) -> tuple[int, ...]:
    return tuple(block_width for _ in range(num_blocks))


def run_case(spec: CaseSpec) -> CaseResult:
    seeds: CaseSeeds = case_seeds(spec.config_seed, spec.case_index)

    layout = BlockLayout(_resolve_block_widths(spec.num_blocks, spec.block_width))
    incidence = build_incidence(spec.family, spec.num_tasks, spec.num_blocks, seeds.graph_structure)
    incidence_hash = hashlib.sha256(incidence.tobytes()).hexdigest()

    noise_model = NoiseModel(
        kind=spec.noise_kind,  # type: ignore[arg-type]
        sigma=spec.noise_sigma,
        rho=spec.noise_rho,
    )

    tasks, diagnostics = gen.generate_tasks(
        layout,
        incidence,
        seeds.curvature,
        private_dims=spec.private_dims,
        condition_number=spec.condition_number,
        coupling_rank=spec.coupling_rank,
        mu=spec.mu,
        gradient_cosine_target=spec.gradient_cosine_target,
        gradient_scale_target=spec.gradient_scale_target,
    )
    per_task_results = []
    task_selectors: list[FloatArray] = []
    task_local_gradients: list[FloatArray] = []
    for i, task in enumerate(tasks):
        p_i = task.shared_dim
        x_i = seeds.gradient.standard_normal(p_i)
        phi_0 = seeds.gradient.standard_normal(task.private_dim)
        probe = seeds.probe_direction.standard_normal(p_i)
        probe /= np.linalg.norm(probe)
        step = 0.01 * seeds.probe_direction.standard_normal(p_i)
        d = task.private_dim

        if spec.optimizer == "sgd":
            eigs_c = np.linalg.eigvalsh(task.private_curvature)
            eta, target_rho = st.sgd_eta_for_regime(eigs_c, spec.stability_regime, seeds.noise)
            noise = noise_model.sample(seeds.noise, spec.horizon, task.private_dim)
            checks = cc.check_sgd_case(task, x_i, phi_0, eta, noise, probe, step)
            realized_actual = st.spectral_radius(sgd.sgd_state_jacobian(task, eta))

            # analytic_grad/q_i_k duplicate crosscheck.check_sgd_case's internal
            # computation rather than changing its return type, per the standing
            # decision to keep that already-tested function's signature stable.
            phi_k_closed = sgd.sgd_closed_form_state(task, x_i, phi_0, eta, noise)
            z_k = sgd.sgd_sensitivity(task, eta, spec.horizon)
            analytic_grad = hg.rerun_gradient(task, x_i, phi_k_closed, z_k)
            r_k = sgd.sgd_closed_form_state(task, np.zeros_like(x_i), phi_0, eta, noise)
            _, q_i_k = hg.quadratic_model(task, z_k, r_k)

            detail = None
            if spec.keep_full_detail:
                phi_trajectory = sgd.sgd_unroll(task, x_i, phi_0, eta, noise)
                z_trajectory = sgd.sgd_sensitivity_trajectory(task, eta, spec.horizon)
                j_k = sgd.sgd_state_jacobian(task, eta)
                b_k = sgd.sgd_input_jacobian(task, eta)
                detail = {
                    "x_i": x_i.tolist(),
                    "phi_0": phi_0.tolist(),
                    "noise": noise.tolist(),
                    "state_trajectory": {"phi": phi_trajectory.tolist()},
                    "per_step_jacobians": {
                        "J_k": np.repeat(j_k[np.newaxis], spec.horizon, axis=0).tolist(),
                        "B_k": np.repeat(b_k[np.newaxis], spec.horizon, axis=0).tolist(),
                    },
                    "sensitivity_trajectory": {"Z": z_trajectory.tolist()},
                    "exact_local_gradient": analytic_grad.tolist(),
                    "Q_i_K": q_i_k.tolist(),
                    "selector": task.selector.tolist(),
                    "selector_hash": selector_hash(task.selector),
                    "case_index": spec.case_index,
                    "task_index": i,
                }
        elif spec.optimizer == "momentum":
            beta = spec.beta if spec.beta is not None else 0.9
            eta, target_rho = st.momentum_eta_for_regime(task, beta, spec.stability_regime, seeds.noise)
            noise = noise_model.sample(seeds.noise, spec.horizon, task.private_dim)
            v_0 = seeds.gradient.standard_normal(task.private_dim)
            checks = cc.check_momentum_case(task, x_i, phi_0, v_0, eta, beta, noise, probe, step)
            realized_actual = st.spectral_radius(momentum_transition_jacobian(task, eta, beta))

            phi_k_closed, v_k_closed = momentum_closed_form_state(task, x_i, phi_0, v_0, eta, beta, noise)
            w_k = momentum_sensitivity(task, eta, beta, spec.horizon)
            z_k_phi = w_k[:d]
            analytic_grad = hg.rerun_gradient(task, x_i, phi_k_closed, z_k_phi)
            r_k_phi, _ = momentum_closed_form_state(task, np.zeros_like(x_i), phi_0, v_0, eta, beta, noise)
            _, q_i_k = hg.quadratic_model(task, z_k_phi, r_k_phi)

            detail = None
            if spec.keep_full_detail:
                phi_trajectory, v_trajectory = momentum_unroll(task, x_i, phi_0, v_0, eta, beta, noise)
                w_trajectory = momentum_sensitivity_trajectory(task, eta, beta, spec.horizon)
                a_k = momentum_transition_jacobian(task, eta, beta)
                b_k = momentum_input_jacobian(task, eta)
                detail = {
                    "x_i": x_i.tolist(),
                    "phi_0": phi_0.tolist(),
                    "v_0": v_0.tolist(),
                    "noise": noise.tolist(),
                    "state_trajectory": {"phi": phi_trajectory.tolist(), "v": v_trajectory.tolist()},
                    "per_step_jacobians": {
                        "J_k": np.repeat(a_k[np.newaxis], spec.horizon, axis=0).tolist(),
                        "B_k": np.repeat(b_k[np.newaxis], spec.horizon, axis=0).tolist(),
                    },
                    "sensitivity_trajectory": {"W": w_trajectory.tolist()},
                    "exact_local_gradient": analytic_grad.tolist(),
                    "Q_i_K": q_i_k.tolist(),
                    "selector": task.selector.tolist(),
                    "selector_hash": selector_hash(task.selector),
                    "case_index": spec.case_index,
                    "task_index": i,
                }
        else:  # pragma: no cover - exhaustive Literal
            raise ValueError(f"unknown optimizer {spec.optimizer!r}")

        task_selectors.append(task.selector)
        task_local_gradients.append(analytic_grad)

        per_task_results.append(
            TaskCaseResult(
                task_index=i,
                private_dim=task.private_dim,
                shared_dim=p_i,
                selector_hash=selector_hash(task.selector),
                eta=eta,
                stability_target_or_realized=target_rho,
                stability_realized_actual=realized_actual,
                condition_number_realized=diagnostics.condition_number_realized[i],
                coupling_rank_realized=diagnostics.coupling_rank_realized[i],
                checks=checks,
                detail=detail,
            )
        )

    # R3: independent exact/high-accuracy common-descent (Pareto) reference
    # over the tasks' real lifted exact rerun-response gradients (spec §6),
    # not random probe directions -- see comppareto.oracle.pareto.
    pareto_reference = pareto.case_pareto_reference(task_selectors, task_local_gradients)

    return CaseResult(
        spec=spec,
        incidence_hash=incidence_hash,
        designated_pair=diagnostics.designated_pair,
        gradient_cosine_realized=diagnostics.gradient_cosine_realized,
        gradient_scale_realized=diagnostics.gradient_scale_realized,
        tasks=tuple(per_task_results),
        pareto_reference=pareto_reference,
    )
