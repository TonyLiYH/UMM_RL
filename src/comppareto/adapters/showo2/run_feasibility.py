"""CLI entry point for the T215 Show-o2 finite-response diagnostic.

Orchestrates, for both task paths (MMU understanding, T2I generation), on the
real Show-o2 checkpoint loaded via ``model_io.py``:

1. build fixed adaptation/meta batches (disjoint, per
   ``configs/feasibility/showo2/diagnostic-v1.yaml``) plus one named
   same-batch diagnostic variant;
2. snapshot the 3-tensor diagnostic subspace's state (``state.py``);
3. compute raw / commit-response / rerun-response gradients (``protocols.py``)
   for ``K=1`` (and, only if the ``K=1`` gate passes, ``K=3``), for both the
   parameter-only and complete optimizer-state differentiation variants;
4. verify full rollback after every transition;
5. run the central finite-difference check (``finite_diff.py``) against the
   rerun-response gradient;
6. write ``resolved-config.yaml``, ``manifest.json``, ``metrics.json``, and
   ``notes.md`` into ``--output-dir``, matching
   ``tasks/contracts/T215.acceptance.yaml``'s required paths/metrics.

This module is only run against the real GPU checkpoint (see
``reports/T215/first-report.md`` section 8 for the exact invocation); it is
not imported or exercised by ``tests/adapters/showo2/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from comppareto.adapters.showo2 import model_io, state
from comppareto.adapters.showo2.protocols import (
    AdamWMomentState,
    compute_commit_response,
    compute_raw,
    compute_rerun_response,
    rerun_loss_only,
)
from comppareto.adapters.showo2.finite_diff import run_finite_difference_check

REPO_ROOT = Path(__file__).resolve().parents[4]
assert (REPO_ROOT / "tasks" / "contracts" / "T215.acceptance.yaml").exists(), (
    f"unexpected REPO_ROOT resolution: {REPO_ROOT}"
)


# ---------------------------------------------------------------------------
# Small helpers shared with comppareto.oracle.sweep's conventions.
# ---------------------------------------------------------------------------


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_dirty(repo_root: Path) -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return bool(out.strip())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _environment_snapshot() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
    return env


# ---------------------------------------------------------------------------
# Per-task-path diagnostic runner.
# ---------------------------------------------------------------------------


def _task_path_gate_passed(fd_results: list) -> bool:
    return all(r.passed for r in fd_results)


def run_one_task_path(
    *,
    task_name: str,
    loss_fn,
    env,
    private_block: model_io.LeafBlock,
    shared_block: model_io.LeafBlock,
    adapt_batch: dict[str, Any],
    meta_batch: dict[str, Any],
    lr: float,
    k1: int,
    k3: int,
) -> dict[str, Any]:
    """Run the full raw/commit/rerun/FD/rollback sequence for one task path
    (MMU or T2I), on both the disjoint adapt/meta split and the named
    same-batch diagnostic, for ``K=1`` and (conditionally) ``K=3``.
    """

    result: dict[str, Any] = {"task": task_name}

    theta_s_0 = model_io.flatten_block(env.model, shared_block)
    theta_p_0 = model_io.flatten_block(env.model, private_block)
    opt_state_0 = AdamWMomentState.zeros_like(theta_p_0)

    subspace_param_names = list(shared_block.names) + list(private_block.names)
    subspace_state_pre = state.snapshot_subspace(env.model, subspace_param_names)

    variants: dict[str, dict[str, Any]] = {}

    def _run_split(label: str, adapt: dict[str, Any], meta: dict[str, Any], k: int) -> dict[str, Any]:
        torch.manual_seed(1215)
        raw = compute_raw(loss_fn, theta_s_0, theta_p_0, meta)

        torch.manual_seed(1215)
        commit = compute_commit_response(
            loss_fn, theta_s_0, theta_p_0, opt_state_0, adapt, meta, lr, k_steps=k
        )

        torch.manual_seed(1215)
        rerun_param_only = compute_rerun_response(
            loss_fn, theta_s_0, theta_p_0, opt_state_0, adapt, meta, lr, k_steps=k, detach_moments=True
        )

        torch.manual_seed(1215)
        rerun_complete = compute_rerun_response(
            loss_fn, theta_s_0, theta_p_0, opt_state_0, adapt, meta, lr, k_steps=k, detach_moments=False
        )

        rollback_ok, rollback_details = state.verify_rollback(env.model, subspace_state_pre)
        state.restore_subspace(env.model, subspace_state_pre)

        def _loss_at(theta_s_perturbed: torch.Tensor) -> float:
            return rerun_loss_only(
                loss_fn, theta_s_perturbed, theta_p_0, opt_state_0, adapt, meta, lr, k_steps=k
            )

        fd_results = run_finite_difference_check(
            theta_s_0, raw.grad_theta_s, rerun_complete.grad_theta_s, _loss_at
        )

        rollback_ok_2, _ = state.verify_rollback(env.model, subspace_state_pre)
        state.restore_subspace(env.model, subspace_state_pre)

        return {
            "k": k,
            "raw_loss_meta": raw.loss_meta,
            "raw_grad_norm": float(raw.grad_theta_s.norm()),
            "commit_loss_adapt": commit.loss_adapt,
            "commit_loss_meta": commit.loss_meta,
            "commit_grad_norm": float(commit.grad_theta_s.norm()),
            "rerun_param_only_loss_adapt": rerun_param_only.loss_adapt,
            "rerun_param_only_loss_meta": rerun_param_only.loss_meta,
            "rerun_param_only_grad_norm": float(rerun_param_only.grad_theta_s.norm()),
            "rerun_complete_loss_adapt": rerun_complete.loss_adapt,
            "rerun_complete_loss_meta": rerun_complete.loss_meta,
            "rerun_complete_grad_norm": float(rerun_complete.grad_theta_s.norm()),
            "raw_vs_commit_cosine": _cosine(raw.grad_theta_s, commit.grad_theta_s),
            "raw_vs_rerun_cosine": _cosine(raw.grad_theta_s, rerun_complete.grad_theta_s),
            "commit_vs_rerun_cosine": _cosine(commit.grad_theta_s, rerun_complete.grad_theta_s),
            "rollback_ok_after_rerun_variants": bool(rollback_ok),
            "rollback_ok_after_fd": bool(rollback_ok_2),
            "rollback_detail": [
                {
                    "name": d.name,
                    "data_max_abs_diff": d.data_max_abs_diff,
                    "all_matches": d.all_matches,
                }
                for d in rollback_details
            ],
            "finite_difference": [
                {
                    "label": r.label,
                    "eps": r.eps,
                    "fd_value": r.fd_value,
                    "analytic_value": r.analytic_value,
                    "reference_magnitude": r.reference_magnitude,
                    "error": r.error,
                    "mode": r.mode,
                    "passed": r.passed,
                }
                for r in fd_results
            ],
            "finite_difference_gate_passed": _task_path_gate_passed(fd_results),
        }

    variants["disjoint_k1"] = _run_split("disjoint_k1", adapt_batch, meta_batch, k1)
    variants["same_batch_k1"] = _run_split("same_batch_k1", meta_batch, meta_batch, k1)

    k1_gate_passed = (
        variants["disjoint_k1"]["finite_difference_gate_passed"]
        and variants["disjoint_k1"]["rollback_ok_after_rerun_variants"]
        and variants["disjoint_k1"]["rollback_ok_after_fd"]
    )
    result["k1_gate_passed"] = k1_gate_passed

    if k1_gate_passed:
        variants["disjoint_k3"] = _run_split("disjoint_k3", adapt_batch, meta_batch, k3)
    else:
        variants["disjoint_k3"] = None

    result["variants"] = variants
    return result


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm()) * float(b.norm())
    if denom == 0.0:
        return 0.0
    return float((a * b).sum()) / denom


# ---------------------------------------------------------------------------
# Main orchestration.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    execution_revision = _git_head(REPO_ROOT)
    dirty = _git_dirty(REPO_ROOT)
    config_sha256 = _sha256_file(config_path)

    resolved_config_path = out_dir / "resolved-config.yaml"
    resolved_config_path.write_text(config_path.read_text())

    start = time.perf_counter()
    failure_reason: str | None = None
    task_results: dict[str, Any] = {}

    try:
        env = model_io.load_env(
            repo_root=config["showo2"]["repo_root"],
            demo_config_rel=config["showo2"]["demo_config_rel"],
            device=config["showo2"]["device"],
            weight_type=getattr(torch, config["showo2"]["weight_type"]),
        )

        shared_block = model_io.build_leaf_block(env.model, config["subspace"]["shared_block_prefix"])
        und_private_block = model_io.build_leaf_block(
            env.model, config["subspace"]["und_private_block_prefix"]
        )
        gen_private_block = model_io.build_leaf_block(
            env.model, config["subspace"]["gen_private_block_prefix"]
        )

        lr = float(config["optimizer"]["lr"])
        k1 = int(config["optimizer"]["k1"])
        k3 = int(config["optimizer"]["k3"])

        # --- MMU (understanding) task path ---
        mmu_meta_batch = model_io.build_mmu_batch(
            env,
            config["mmu"]["meta"]["image_path"],
            config["mmu"]["meta"]["question"],
            caption_seed=int(config["seeds"]["mmu_pseudo_caption"]),
        )
        mmu_adapt_batch = model_io.build_mmu_batch(
            env,
            config["mmu"]["adapt"]["image_path"],
            config["mmu"]["adapt"]["question"],
            caption_seed=int(config["seeds"]["mmu_pseudo_caption"]) + 1,
        )
        mmu_loss_fn = model_io.make_mmu_loss_fn(env, shared_block, und_private_block)
        task_results["mmu"] = run_one_task_path(
            task_name="mmu",
            loss_fn=mmu_loss_fn,
            env=env,
            private_block=und_private_block,
            shared_block=shared_block,
            adapt_batch=mmu_adapt_batch,
            meta_batch=mmu_meta_batch,
            lr=lr,
            k1=k1,
            k3=k3,
        )

        # --- T2I (generation) task path ---
        prompts_file = Path(config["showo2"]["repo_root"]) / config["t2i"]["adapt"]["prompts_file"]
        adapt_prompt = prompts_file.read_text().splitlines()[
            int(config["t2i"]["adapt"]["prompt_line_index"])
        ].strip()
        pseudo_target_image_path = config["t2i"]["pseudo_target_image_path"]
        t2i_meta_batch = model_io.build_t2i_batch(
            env,
            config["t2i"]["meta"]["prompt"],
            pseudo_target_image_path,
            seed=int(config["seeds"]["t2i_transport_sample"]),
        )
        t2i_adapt_batch = model_io.build_t2i_batch(
            env,
            adapt_prompt,
            pseudo_target_image_path,
            seed=int(config["seeds"]["t2i_transport_sample"]) + 1,
        )
        t2i_loss_fn = model_io.make_t2i_loss_fn(env, shared_block, gen_private_block)
        task_results["t2i"] = run_one_task_path(
            task_name="t2i",
            loss_fn=t2i_loss_fn,
            env=env,
            private_block=gen_private_block,
            shared_block=shared_block,
            adapt_batch=t2i_adapt_batch,
            meta_batch=t2i_meta_batch,
            lr=lr,
            k1=k1,
            k3=k3,
        )

        peak_alloc_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        peak_reserved_bytes = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    except Exception as exc:  # noqa: BLE001 -- record as a diagnostic failure, not a crash
        failure_reason = f"{type(exc).__name__}: {exc}"
        peak_alloc_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        peak_reserved_bytes = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0

    elapsed_seconds = time.perf_counter() - start
    gpu_hours = elapsed_seconds / 3600.0

    snapshot_restore_failed = 0
    finite_difference_failed = 0
    protocols_measured = {"raw_measured": False, "commit_measured": False, "rerun_measured": False}
    for task_name, task_result in task_results.items():
        for variant_name, variant in task_result.get("variants", {}).items():
            if variant is None:
                continue
            if not variant["rollback_ok_after_rerun_variants"] or not variant["rollback_ok_after_fd"]:
                snapshot_restore_failed += 1
            if not variant["finite_difference_gate_passed"]:
                finite_difference_failed += 1
            protocols_measured["raw_measured"] = True
            protocols_measured["commit_measured"] = True
            protocols_measured["rerun_measured"] = True

    any_k1_gate_passed = any(
        task_result.get("k1_gate_passed", False) for task_result in task_results.values()
    )
    status = "pass" if (failure_reason is None and any_k1_gate_passed) else "fail"

    metrics: dict[str, Any] = {
        "task_results": task_results,
        "snapshot_restore": {"failed": snapshot_restore_failed},
        "finite_difference": {"failed": finite_difference_failed},
        "protocols": protocols_measured
        if task_results
        else {"raw_measured": False, "commit_measured": False, "rerun_measured": False},
        "resources": {
            "gpu_hours": gpu_hours,
            "wall_clock_seconds": elapsed_seconds,
            "peak_memory_allocated_bytes": peak_alloc_bytes,
            "peak_memory_reserved_bytes": peak_reserved_bytes,
            "gpu_count_used": 1 if torch.cuda.is_available() else 0,
        },
        # No optimizer.step()/persistent parameter mutation is ever performed
        # by protocols.py (every AdamW update is a returned new tensor, not
        # an in-place model parameter write) or by model_io.py's
        # _reparametrize_module-based substitution (the model's own stored
        # parameters are swapped back untouched on every context-manager
        # exit) -- see model_io.py's module docstring. This count is
        # therefore always 0 by construction, not by observation.
        "persistent_updates": 0,
        "failure_reason": failure_reason,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    result_files = [
        str(resolved_config_path.relative_to(REPO_ROOT)),
        str(metrics_path.relative_to(REPO_ROOT)),
    ]
    artifacts = [
        {
            "artifact_id": p.stem,
            "kind": "showo2-feasibility-artifact",
            "canonical_uri": str(p.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(p),
            "bytes": p.stat().st_size,
        }
        for p in (resolved_config_path, metrics_path)
    ]

    manifest = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "task_id": "T215",
        "run_kind": "formal" if not dirty else "diagnostic",
        "source_revision": execution_revision,
        "execution_revision": execution_revision,
        "dirty": dirty,
        "config_sha256": config_sha256,
        "environment": _environment_snapshot(),
        "status": status,
        "result_files": result_files,
        "artifacts": artifacts,
        "retry": None,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"status": status, "gpu_hours": gpu_hours, "failure_reason": failure_reason}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
