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
    }
    for t in result.tasks:
        record["tasks"].append(
            {
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
        )
    output_encoded = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    record["output_hash"] = hashlib.sha256(output_encoded).hexdigest()
    return record
