"""Deterministic run manifest: per-case summary records and configuration hashing.

Matches ``docs/theory/oracle-spec.md`` section 11 (handoff artifact schema).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from comppareto.oracle.case import CaseResult
from comppareto.oracle.crosscheck import CheckResult


def _check_dict(check: CheckResult) -> dict[str, Any]:
    return {"error": check.error, "tolerance": check.tolerance, "mode": check.mode, "passed": check.passed}


def config_hash(spec_dict: dict[str, Any]) -> str:
    encoded = json.dumps(spec_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def case_record(result: CaseResult, source_revision: str) -> dict[str, Any]:
    spec = asdict(result.spec)
    record: dict[str, Any] = {
        "case_index": result.spec.case_index,
        "config_seed": result.spec.config_seed,
        "family": result.spec.family,
        "num_tasks": result.spec.num_tasks,
        "num_blocks": result.spec.num_blocks,
        "block_width": result.spec.block_width,
        "optimizer": result.spec.optimizer,
        "beta": result.spec.beta,
        "horizon": result.spec.horizon,
        "stability_regime": result.spec.stability_regime,
        "noise_kind": result.spec.noise_kind,
        "noise_sigma": result.spec.noise_sigma,
        "noise_rho": result.spec.noise_rho,
        "incidence_hash": result.incidence_hash,
        "designated_pair": result.designated_pair,
        "gradient_cosine_target": result.spec.gradient_cosine_target,
        "gradient_cosine_realized": result.gradient_cosine_realized,
        "gradient_scale_target": result.spec.gradient_scale_target,
        "gradient_scale_realized": result.gradient_scale_realized,
        "source_revision": source_revision,
        "configuration_hash": config_hash(spec),
        "all_passed": result.all_passed,
        "tasks": [],
        # R3: independent exact/high-accuracy common-descent (Pareto) reference
        # over the tasks' real lifted exact rerun-response gradients -- see
        # comppareto.oracle.pareto.case_pareto_reference.
        "pareto_reference": result.pareto_reference,
    }
    for t in result.tasks:
        task_record: dict[str, Any] = {
            "task_index": t.task_index,
            "private_dim": t.private_dim,
            "shared_dim": t.shared_dim,
            "selector_hash": t.selector_hash,
            "eta": t.eta,
            "stability_target_or_realized": t.stability_target_or_realized,
            "stability_realized_actual": t.stability_realized_actual,
            "condition_number_realized": t.condition_number_realized,
            "coupling_rank_realized": t.coupling_rank_realized,
            "checks": {
                "state": _check_dict(t.checks.state),
                "hypergradient": _check_dict(t.checks.hypergradient),
                "loss_change": _check_dict(t.checks.loss_change),
                "finite_difference": [_check_dict(c) for c in t.checks.finite_difference],
            },
            "all_passed": t.checks.all_passed,
        }
        # R2: full state trajectory, per-step Jacobians, sensitivity
        # trajectory, exact local gradient, and Q_i^K -- only populated when
        # spec.keep_full_detail selects this case for the detailed subset
        # (docs/theory/oracle-spec.md section 11).
        if t.detail is not None:
            task_record["detail"] = t.detail
        record["tasks"].append(task_record)
    output_encoded = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    record["output_hash"] = hashlib.sha256(output_encoded).hexdigest()
    return record


def build_run_manifest(
    *,
    run_id: str,
    task_id: str,
    run_kind: str,
    source_revision: str,
    execution_revision: str,
    dirty: bool,
    config_sha256: str,
    environment: dict[str, Any],
    status: str,
    result_files: list[str],
    artifacts: list[dict[str, Any]],
    retry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a ``schemas/run-manifest.schema.json``-valid top-level object.

    Separate from :func:`case_record`, which builds one per-case entry for
    ``case-records.json`` -- this is the schema-required envelope written as
    ``manifest.json`` (R1: the manifest itself must be a JSON object, not the
    flat per-case array).
    """

    return {
        "schema_version": 1,
        "run_id": run_id,
        "task_id": task_id,
        "run_kind": run_kind,
        "source_revision": source_revision,
        "execution_revision": execution_revision,
        "dirty": dirty,
        "config_sha256": config_sha256,
        "environment": environment,
        "status": status,
        "result_files": result_files,
        "artifacts": artifacts,
        "retry": retry,
    }
