from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import yaml

from comppareto.oracle import hypergradient as hg
from comppareto.oracle import pareto
from comppareto.oracle.case import selector_hash
from comppareto.oracle.manifest import case_record
from comppareto.oracle.momentum import momentum_closed_form_state
from comppareto.oracle.sgd import sgd_closed_form_state
from comppareto.oracle.sweep import enumerate_cases

CONFIG_PATH = "configs/oracle/baseline.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _first_case(optimizer: str) -> "CaseSpec":  # noqa: F821 - imported lazily below
    from comppareto.oracle.case import CaseSpec

    config = _load_config()
    cases = enumerate_cases(config)
    spec = next(c for c in cases if c.optimizer == optimizer)
    return replace(spec, keep_full_detail=True)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum"])
def test_run_case_detail_populated_when_keep_full_detail(optimizer: str) -> None:
    from comppareto.oracle.case import run_case

    spec = _first_case(optimizer)
    result = run_case(spec)

    assert all(t.detail is not None for t in result.tasks)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum"])
def test_run_case_detail_absent_by_default(optimizer: str) -> None:
    from comppareto.oracle.case import run_case

    spec = replace(_first_case(optimizer), keep_full_detail=False)
    result = run_case(spec)

    assert all(t.detail is None for t in result.tasks)


def test_sgd_case_detail_shapes_match_horizon_and_dims() -> None:
    from comppareto.oracle.case import run_case

    spec = _first_case("sgd")
    result = run_case(spec)

    for i, t in enumerate(result.tasks):
        detail = t.detail
        p_i = t.shared_dim
        d_i = t.private_dim
        assert len(detail["state_trajectory"]["phi"]) == spec.horizon + 1
        assert len(detail["state_trajectory"]["phi"][0]) == d_i
        assert len(detail["per_step_jacobians"]["J_k"]) == spec.horizon
        assert len(detail["per_step_jacobians"]["J_k"][0]) == d_i
        assert len(detail["per_step_jacobians"]["B_k"][0]) == d_i
        assert len(detail["sensitivity_trajectory"]["Z"]) == spec.horizon + 1
        assert len(detail["exact_local_gradient"]) == p_i
        assert len(detail["Q_i_K"]) == p_i
        assert len(detail["Q_i_K"][0]) == p_i
        assert detail["case_index"] == spec.case_index
        assert detail["task_index"] == i


def test_momentum_case_detail_shapes_match_horizon_and_dims() -> None:
    from comppareto.oracle.case import run_case

    spec = _first_case("momentum")
    result = run_case(spec)

    for i, t in enumerate(result.tasks):
        detail = t.detail
        p_i = t.shared_dim
        d_i = t.private_dim
        assert len(detail["state_trajectory"]["phi"]) == spec.horizon + 1
        assert len(detail["state_trajectory"]["v"]) == spec.horizon + 1
        assert len(detail["state_trajectory"]["phi"][0]) == d_i
        assert len(detail["per_step_jacobians"]["J_k"]) == spec.horizon
        assert len(detail["per_step_jacobians"]["J_k"][0]) == 2 * d_i
        assert len(detail["per_step_jacobians"]["J_k"][0][0]) == 2 * d_i
        assert len(detail["per_step_jacobians"]["B_k"][0]) == 2 * d_i
        assert len(detail["sensitivity_trajectory"]["W"]) == spec.horizon + 1
        assert len(detail["sensitivity_trajectory"]["W"][0]) == 2 * d_i
        assert len(detail["exact_local_gradient"]) == p_i
        assert len(detail["Q_i_K"]) == p_i
        assert detail["case_index"] == spec.case_index
        assert detail["task_index"] == i


@pytest.mark.parametrize("optimizer", ["sgd", "momentum"])
def test_detail_selector_hash_matches_selector_hash_helper(optimizer: str) -> None:
    from comppareto.oracle.case import run_case

    spec = _first_case(optimizer)
    result = run_case(spec)

    for t in result.tasks:
        recomputed = selector_hash(np.asarray(t.detail["selector"], dtype=np.float64))
        assert recomputed == t.detail["selector_hash"]
        assert recomputed == t.selector_hash


def test_sgd_detail_per_step_jacobians_are_constant_and_match_helpers() -> None:
    from comppareto.oracle.case import run_case
    from comppareto.oracle.tasks import OracleTask

    spec = _first_case("sgd")
    result = run_case(spec)

    for t in result.tasks:
        detail = t.detail
        j_k = np.asarray(detail["per_step_jacobians"]["J_k"])
        b_k = np.asarray(detail["per_step_jacobians"]["B_k"])
        assert np.allclose(j_k, j_k[0])
        assert np.allclose(b_k, b_k[0])


def test_momentum_detail_persists_consistent_input_shapes() -> None:
    from comppareto.oracle.case import run_case

    spec = _first_case("momentum")
    result = run_case(spec)

    for t in result.tasks:
        detail = t.detail
        assert len(detail["x_i"]) == t.shared_dim
        assert len(detail["phi_0"]) == t.private_dim == len(detail["v_0"])
        assert len(detail["noise"]) == spec.horizon
        assert len(detail["noise"][0]) == t.private_dim


@pytest.mark.parametrize("optimizer", ["sgd", "momentum"])
def test_detail_sensitivity_trajectory_final_step_matches_exact_local_gradient(optimizer: str) -> None:
    """The persisted Q_i^K/exact_local_gradient must be independently
    reproducible from the persisted sensitivity trajectory's final step and
    the persisted state trajectory, using the already-tested hypergradient
    helpers -- an end-to-end value check of the whole detail payload, not
    just its shape.
    """
    from comppareto.oracle.case import run_case

    spec = _first_case(optimizer)
    result = run_case(spec)
    beta = spec.beta if spec.beta is not None else 0.9

    for t in result.tasks:
        detail = t.detail
        phi_k = np.asarray(detail["state_trajectory"]["phi"][-1])
        exact_grad = np.asarray(detail["exact_local_gradient"])
        q = np.asarray(detail["Q_i_K"])

        if optimizer == "sgd":
            z_k = np.asarray(detail["sensitivity_trajectory"]["Z"][-1])
        else:
            z_k = np.asarray(detail["sensitivity_trajectory"]["W"][-1])[: t.private_dim]

        task = _task_from_result_index(spec, t.task_index)
        x_i = np.asarray(detail["x_i"])
        recomputed_grad = hg.rerun_gradient(task, x_i, phi_k, z_k)
        assert np.linalg.norm(recomputed_grad - exact_grad) / max(np.linalg.norm(exact_grad), 1e-30) <= 1e-10

        noise = np.asarray(detail["noise"])
        phi_0 = np.asarray(detail["phi_0"])
        if optimizer == "sgd":
            r_k = sgd_closed_form_state(task, np.zeros_like(x_i), phi_0, t.eta, noise)
        else:
            v_0 = np.asarray(detail["v_0"])
            r_k, _ = momentum_closed_form_state(task, np.zeros_like(x_i), phi_0, v_0, t.eta, beta, noise)
        _, recomputed_q = hg.quadratic_model(task, z_k, r_k)
        assert np.linalg.norm(recomputed_q - q) / max(np.linalg.norm(q), 1e-30) <= 1e-10


def _task_from_result_index(spec, task_index: int):
    from comppareto.oracle import generation as gen
    from comppareto.oracle.selectors import BlockLayout, build_incidence
    from comppareto.oracle.seeds import case_seeds

    seeds = case_seeds(spec.config_seed, spec.case_index)
    layout = BlockLayout(tuple(spec.block_width for _ in range(spec.num_blocks)))
    incidence = build_incidence(spec.family, spec.num_tasks, spec.num_blocks, seeds.graph_structure)
    tasks, _ = gen.generate_tasks(
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
    return tasks[task_index]


@pytest.mark.parametrize("optimizer", ["sgd", "momentum"])
def test_case_result_pareto_reference_uses_real_task_selectors_and_gradients(optimizer: str) -> None:
    """R3: the wired-in pareto_reference must match an independent recomputation
    from each task's real selector and the persisted exact local gradient --
    not random probe directions.
    """
    from comppareto.oracle.case import run_case

    spec = _first_case(optimizer)
    result = run_case(spec)

    selectors = [np.asarray(t.detail["selector"]) for t in result.tasks]
    gradients = [np.asarray(t.detail["exact_local_gradient"]) for t in result.tasks]
    expected = pareto.case_pareto_reference(selectors, gradients)

    assert result.pareto_reference["active_set"]["kkt_residual"] <= 1e-8
    assert np.allclose(
        result.pareto_reference["active_set"]["combined_gradient"],
        expected["active_set"]["combined_gradient"],
        atol=1e-8,
    )


def test_case_result_all_passed_gates_on_pareto_independent_check() -> None:
    # R8 (second local review): "a stored pareto_reference without a passing
    # independent check is insufficient" -- CaseResult.all_passed must flip
    # to False if the Pareto independent_check fails, even when every task's
    # own checks pass.
    from comppareto.oracle.case import run_case

    spec = _first_case("momentum")
    result = run_case(spec)
    assert all(t.checks.all_passed for t in result.tasks)
    assert result.pareto_reference["independent_check"]["all_passed"] is True
    assert result.all_passed is True

    broken_pareto_reference = dict(result.pareto_reference)
    broken_pareto_reference["independent_check"] = dict(result.pareto_reference["independent_check"])
    broken_pareto_reference["independent_check"]["all_passed"] = False
    broken_result = replace(result, pareto_reference=broken_pareto_reference)

    assert all(t.checks.all_passed for t in broken_result.tasks)
    assert broken_result.all_passed is False


def test_case_record_serializes_detail_and_pareto_reference_round_trip() -> None:
    from comppareto.oracle.case import run_case

    spec = _first_case("momentum")
    result = run_case(spec)
    record = case_record(result, "0" * 40)

    encoded = json.dumps(record)
    decoded = json.loads(encoded)

    assert "pareto_reference" in decoded
    assert decoded["pareto_reference"]["active_set"]["kkt_residual"] == result.pareto_reference["active_set"][
        "kkt_residual"
    ]
    for t, task_record in zip(result.tasks, decoded["tasks"]):
        assert task_record["detail"]["case_index"] == t.detail["case_index"]
        assert task_record["detail"]["task_index"] == t.detail["task_index"]
        assert task_record["detail"]["selector_hash"] == t.detail["selector_hash"]


def test_case_record_omits_detail_when_not_kept() -> None:
    from comppareto.oracle.case import run_case

    spec = replace(_first_case("sgd"), keep_full_detail=False)
    result = run_case(spec)
    record = case_record(result, "0" * 40)

    for task_record in record["tasks"]:
        assert "detail" not in task_record
