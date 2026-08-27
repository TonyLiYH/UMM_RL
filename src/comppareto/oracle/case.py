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
from comppareto.oracle import stability as st
from comppareto.oracle.momentum import momentum_transition_jacobian
from comppareto.oracle.noise import NoiseModel
from comppareto.oracle.seeds import CaseSeeds, case_seeds
from comppareto.oracle.selectors import BlockLayout, GraphFamily, build_incidence
from comppareto.oracle.sgd import sgd_state_jacobian

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
    for i, task in enumerate(tasks):
        p_i = task.shared_dim
        x_i = seeds.gradient.standard_normal(p_i)
        phi_0 = seeds.gradient.standard_normal(task.private_dim)
        probe = seeds.probe_direction.standard_normal(p_i)
        probe /= np.linalg.norm(probe)
        step = 0.01 * seeds.probe_direction.standard_normal(p_i)

        if spec.optimizer == "sgd":
            eigs_c = np.linalg.eigvalsh(task.private_curvature)
            eta, target_rho = st.sgd_eta_for_regime(eigs_c, spec.stability_regime, seeds.noise)
            noise = noise_model.sample(seeds.noise, spec.horizon, task.private_dim)
            checks = cc.check_sgd_case(task, x_i, phi_0, eta, noise, probe, step)
            realized_actual = st.spectral_radius(sgd_state_jacobian(task, eta))
        elif spec.optimizer == "momentum":
            beta = spec.beta if spec.beta is not None else 0.9
            eta, target_rho = st.momentum_eta_for_regime(task, beta, spec.stability_regime, seeds.noise)
            noise = noise_model.sample(seeds.noise, spec.horizon, task.private_dim)
            v_0 = seeds.gradient.standard_normal(task.private_dim)
            checks = cc.check_momentum_case(task, x_i, phi_0, v_0, eta, beta, noise, probe, step)
            realized_actual = st.spectral_radius(momentum_transition_jacobian(task, eta, beta))
        else:  # pragma: no cover - exhaustive Literal
            raise ValueError(f"unknown optimizer {spec.optimizer!r}")

        detail = None
        if spec.keep_full_detail:
            detail = {"x_i": x_i.tolist(), "phi_0": phi_0.tolist(), "noise": noise.tolist()}

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

    return CaseResult(
        spec=spec,
        incidence_hash=incidence_hash,
        designated_pair=diagnostics.designated_pair,
        gradient_cosine_realized=diagnostics.gradient_cosine_realized,
        gradient_scale_realized=diagnostics.gradient_scale_realized,
        tasks=tuple(per_task_results),
    )
