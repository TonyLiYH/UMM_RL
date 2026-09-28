"""Deterministic run manifest: per-case summary records and configuration hashing.

Mirrors ``comppareto.oracle.manifest``'s conventions (schema-valid envelope
object separate from the flat per-case array) for T110's overlap-family runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from comppareto.overlap.verify import CaseResult, CheckOutcome, SafeSetPairResult, TaskCheckResult


def _check_dict(check: CheckOutcome) -> dict[str, Any]:
    return {"passed": check.passed, "error": check.error, "tolerance": check.tolerance}


def _task_check_dict(check: TaskCheckResult) -> dict[str, Any]:
    return {
        "task_index": check.task_index,
        "block_lift_gradient": _check_dict(check.block_lift_gradient),
        "block_lift_hessian": _check_dict(check.block_lift_hessian),
        "objective_change_direct": _check_dict(check.objective_change_direct),
        "private_response_independent": _check_dict(check.private_response_independent),
        "objective_change_independent": _check_dict(check.objective_change_independent),
        "solver_converged": check.solver_converged,
        "all_passed": (
            check.block_lift_gradient.passed
            and check.block_lift_hessian.passed
            and check.objective_change_direct.passed
            and check.private_response_independent.passed
            and check.objective_change_independent.passed
        ),
    }


def _safe_set_pair_dict(pair: SafeSetPairResult) -> dict[str, Any]:
    return {
        "task_i": pair.task_i,
        "task_j": pair.task_j,
        "shared_size": pair.shared_size,
        "private_to_i_size": pair.private_to_i_size,
        "invariance": None if pair.invariance is None else _check_dict(pair.invariance),
        "vacuous": pair.invariance is None,
    }


def config_hash(spec_dict: dict[str, Any]) -> str:
    encoded = json.dumps(spec_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def case_record(result: CaseResult, source_revision: str) -> dict[str, Any]:
    case = result.case
    record: dict[str, Any] = {
        "case_index": case.case_index,
        "config_seed": case.config_seed,
        "regime": case.regime,
        "global_dim": case.global_dim,
        "num_tasks": case.num_tasks,
        "supports": [list(support) for support in case.supports],
        "task_specs": [asdict(spec) for spec in case.task_specs],
        "source_revision": source_revision,
        "all_passed": result.all_passed,
        "task_checks": [_task_check_dict(tc) for tc in result.task_checks],
        "safe_set_pairs": [_safe_set_pair_dict(pair) for pair in result.safe_set_pairs],
    }
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
    """Assemble a ``schemas/run-manifest.schema.json``-valid top-level object."""

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
