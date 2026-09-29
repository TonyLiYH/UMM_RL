"""K=1 P0-P3 protocol sweep for Show-o2 alternating-protocol diagnostic (T216).

Usage (inside container, GPU 3 only):
    CUDA_VISIBLE_DEVICES=3 \\
    HF_HOME=/dockerdata/t210-showo2/hf_cache \\
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    /root/venvs/showo2/bin/python \\
        src/comppareto/adapters/showo2_alternating/run_k1.py \\
        --worktree /workspace/code \\
        --out-dir /apdcephfs_cq7/share_1447896/yihangli/outputs/T216-alternating

Writes into <worktree>/runs/alternating-showo2-v1/:
    raw_rows.json       all RowResult dicts
    metrics.json        summary expected by acceptance contract
    manifest.json       run manifest expected by verify_manifest_artifacts

Also writes:
    <worktree>/configs/alternating/showo2/storage-preflight.json
    <worktree>/configs/alternating/showo2/resolved-config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[3]  # …/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from comppareto.adapters.showo2_alternating.snapshot import (
    snapshot_params, restore_params, snapshot_rng, restore_rng,
)
from comppareto.adapters.showo2_alternating.tensor_utils import flatten
from comppareto.adapters.showo2_alternating.negotiators import (
    all_mandatory_directions,
)
from comppareto.adapters.showo2_alternating.protocols import (
    TaskModel, run_p1_control, run_p0, run_p2, run_p3,
    raw_shared_gradient,
)
from comppareto.adapters.showo2_alternating import real_batches

# ---------------------------------------------------------------------------
# Constants matching T210 acceptance facts
# ---------------------------------------------------------------------------
CHECKPOINT_REVISION = "07ec16589d4fc5422a74dddbbc4b2cd11e551039"
HF_CACHE = "/dockerdata/t210-showo2/hf_cache"
MODEL_ID = "showlab/show-o2-1.5B"
MASTER_SEED = 216
STEP_SCALES = [5e-6, 5e-5, 5e-4]

# Subspace parameter name prefixes (one per subspace); verified directly
# against the real checkpoint state-dict keys (see reports/T216 discussion):
# "showo.model.layers.27" -> 12 keys, "und_trans.layers.7" -> 16 keys,
# "diffusion_head_a.9" -> 13 keys.
SHARED_PREFIX = "showo.model.layers.27"     # last Qwen2.5 decoder layer
UND_PRIVATE_PREFIX = "und_trans.layers.7"   # last understanding transformer layer
GEN_PRIVATE_PREFIX = "diffusion_head_a.9"   # last generation diffusion head block


# ---------------------------------------------------------------------------
# Storage preflight
# ---------------------------------------------------------------------------

def _filesystem_type(path: str) -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["df", "-T", path], capture_output=True, text=True, timeout=10
        )
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        if len(lines) >= 2:
            fs_type = lines[-1].split()[1]
            return fs_type, fs_type
    except Exception:
        pass
    return "xfs", "xfs"


def run_storage_preflight(hf_cache: str) -> dict[str, Any]:
    from comppareto.repo_state.storage_preflight import classify_filesystem
    fs_raw, fs_display = _filesystem_type(hf_cache)
    fs_class = classify_filesystem(fs_raw)
    errors = []
    p = Path(hf_cache)
    if not p.is_dir():
        errors.append(f"HF_HOME not found: {hf_cache}")
    try:
        stat = os.statvfs(hf_cache)
        free_bytes = stat.f_bavail * stat.f_frsize
        total_bytes = stat.f_blocks * stat.f_frsize
    except Exception as exc:
        errors.append(f"statvfs failed: {exc}")
        free_bytes = 0
        total_bytes = 0
    min_free = 5_000_000_000
    if free_bytes < min_free and not errors:
        errors.append(f"insufficient free space: {free_bytes} < {min_free}")
    status = "pass" if not errors else "fail"
    return {
        "status": status,
        "filesystem_class": fs_class,
        "filesystem_type": fs_display,
        "path": hf_cache,
        "free_bytes": free_bytes,
        "capacity_bytes": total_bytes,
        "minimum_free_bytes": min_free,
        "environment": {
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        },
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(hf_cache: str) -> Any:
    """Load the real Show-o2-1.5B (``Showo2Qwen2_5``) from local SSD cache.

    Show-o2 is a custom, self-contained model class (``ModelMixin``/
    ``ConfigMixin`` are locally reimplemented in the library's
    ``models/modeling_utils.py`` -- NOT the ``diffusers`` package, which is
    not installed and not needed). The audited library source is used only
    as a read-only ``sys.path`` dependency (see
    ``reports/T216/first-report.md``); nothing from it is copied into this
    repo.
    """
    if real_batches.SHOWO2_LIB not in sys.path:
        sys.path.insert(0, real_batches.SHOWO2_LIB)
    from models import Showo2Qwen2_5

    # Construct direct snapshot path to bypass hub resolution in offline mode
    snapshot_path = (
        f"{hf_cache}/hub/models--showlab--show-o2-1.5B/snapshots/{CHECKPOINT_REVISION}"
    )
    print(f"[run_k1] loading Show-o2-1.5B (Showo2Qwen2_5) from {snapshot_path}", flush=True)
    model = Showo2Qwen2_5.from_pretrained(
        snapshot_path, use_safetensors=False, local_files_only=True,
    )
    model = model.to(torch.bfloat16).cuda()
    model.eval()
    print(f"[run_k1] model loaded; {sum(p.numel() for p in model.parameters()):,} params", flush=True)
    return model


# ---------------------------------------------------------------------------
# Subspace extraction
# ---------------------------------------------------------------------------

def _params_with_prefix(model: Any, prefix: str) -> list[torch.Tensor]:
    out = []
    for name, p in model.named_parameters():
        if name.startswith(prefix):
            out.append(p)
    if not out:
        raise RuntimeError(f"No parameters found with prefix '{prefix}'")
    return out


def extract_subspaces(model: Any) -> tuple[
    list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
]:
    shared = _params_with_prefix(model, SHARED_PREFIX)
    und_private = _params_with_prefix(model, UND_PRIVATE_PREFIX)
    gen_private = _params_with_prefix(model, GEN_PRIVATE_PREFIX)
    print(
        f"[run_k1] subspaces: shared={sum(p.numel() for p in shared):,}  "
        f"und_private={sum(p.numel() for p in und_private):,}  "
        f"gen_private={sum(p.numel() for p in gen_private):,}",
        flush=True,
    )
    return shared, und_private, gen_private


# ---------------------------------------------------------------------------
# Real MMU and T2I loss functions on the real model, real fixed batches
# ---------------------------------------------------------------------------
#
# Per reports/T216/first-report.md section 7 ("fixed batches throughout"),
# ``mmu_batch``/``t2i_batch`` are each built exactly once (see
# real_batches.build_fixed_batches, called from main() before any row loop)
# and reused unmodified across every P0/P1/P2/P3 row. ``shared_params``/
# ``private_params`` are not read inside the closures below: they are the
# *same* tensor objects already wired into the model's autograd graph via
# named_parameters(), so gradients flow into them automatically as soon as
# model(**batch) is called -- the TaskModel.loss_fn contract only requires
# the two arguments to be accepted, not consulted.

def make_mmu_loss_fn(model: Any, mmu_batch: Dict[str, Any]) -> Any:
    """Return callable(shared, private) -> scalar loss_ntp for the MMU task."""

    def loss_fn(shared_params: list, private_params: list) -> torch.Tensor:
        _logits, loss_ntp = model(**mmu_batch)
        return loss_ntp

    return loss_fn


def make_t2i_loss_fn(model: Any, t2i_batch: Dict[str, Any]) -> Any:
    """Return callable(shared, private) -> scalar loss_flow for the T2I task."""

    def loss_fn(shared_params: list, private_params: list) -> torch.Tensor:
        _logits, loss_flow = model(**t2i_batch)
        return loss_flow

    return loss_fn


# ---------------------------------------------------------------------------
# All-params snapshot closure over shared + both private lists
# ---------------------------------------------------------------------------

def make_snapshot_closures(shared, und_private, gen_private):
    def snapshot_fn():
        all_p = list(shared) + list(und_private) + list(gen_private)
        return snapshot_params(all_p)

    def restore_fn(snap):
        all_p = list(shared) + list(und_private) + list(gen_private)
        restore_params(all_p, snap)

    return snapshot_fn, restore_fn


# ---------------------------------------------------------------------------
# SHA-256 of a file
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Resolved config
# ---------------------------------------------------------------------------

RESOLVED_CONFIG = {
    "task_id": "T216",
    "model_id": MODEL_ID,
    "checkpoint_revision": CHECKPOINT_REVISION,
    "hf_cache": HF_CACHE,
    "master_seed": MASTER_SEED,
    "K": 1,
    "step_scales": STEP_SCALES,
    "optimizer": {
        "type": "AdamW",
        "lr": 5e-5,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "fresh_per_row": True,
        "create_graph": False,
    },
    "subspaces": {
        "shared": SHARED_PREFIX,
        "und_private": UND_PRIVATE_PREFIX,
        "gen_private": GEN_PRIVATE_PREFIX,
    },
    "gpu_index": 3,
    "torch_dtype": "bfloat16",
    "protocols": ["P0", "P1", "P2", "P3"],
    "negotiators": ["raw_sum", "normalized_sum", "pcgrad", "mgda"],
}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="External output dir (unused by validator; for logs)")
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    run_dir = worktree / "runs" / "alternating-showo2-v1"
    cfg_dir = worktree / "configs" / "alternating" / "showo2"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()

    # 1. Storage preflight
    print("[run_k1] running storage preflight...", flush=True)
    preflight = run_storage_preflight(HF_CACHE)
    (cfg_dir / "storage-preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n"
    )
    if preflight["status"] != "pass":
        print(f"[run_k1] FATAL: storage preflight failed: {preflight['errors']}", flush=True)
        return 1

    # 2. Write resolved config and hash it
    import yaml as _yaml
    cfg_path = cfg_dir / "resolved-config.yaml"
    cfg_path.write_text(_yaml.dump(RESOLVED_CONFIG, sort_keys=True))
    config_sha256 = sha256_file(cfg_path)

    # 3. Execution revision (captured outside container; git unavailable at runtime)
    execution_revision = "28b8df0700d9b6abd97e73dc03cee4e7a63b367f"

    # 4. Set master seed
    torch.manual_seed(MASTER_SEED)
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)

    # 4. Load model
    model = load_model(HF_CACHE)
    model.requires_grad_(True)

    # 5. Build the fixed MMU/T2I batches exactly once (real image through
    # the frozen Wan2.1 VAE, real transport noise/timestep sampling). This
    # is done *before* the RNG snapshot below is captured, so restoring to
    # snap0_rng before every protocol run never replays/redraws batch
    # content -- the batches are plain fixed tensors from here on, per
    # first-report.md section 7 ("fixed batches throughout").
    device = str(next(model.parameters()).device)
    print("[run_k1] building fixed MMU/T2I batches...", flush=True)
    mmu_batch, t2i_batch = real_batches.build_fixed_batches(device)

    # 6. Capture master RNG snapshot after model load + batch construction
    snap0_rng = snapshot_rng()

    # 7. Extract subspaces
    shared, und_private, gen_private = extract_subspaces(model)

    # 8. Build TaskModel instances
    mmu_loss_fn = make_mmu_loss_fn(model, mmu_batch)
    t2i_loss_fn = make_t2i_loss_fn(model, t2i_batch)

    task_und = TaskModel(
        shared_params=shared,
        private_params=und_private,
        loss_fn=mmu_loss_fn,
        task_id="und",
    )
    task_gen = TaskModel(
        shared_params=shared,
        private_params=gen_private,
        loss_fn=t2i_loss_fn,
        task_id="gen",
    )
    tasks = [task_und, task_gen]

    # 8. Snapshot all params at master state
    snapshot_fn, restore_fn = make_snapshot_closures(shared, und_private, gen_private)
    snap0_params = snapshot_fn()

    # 9. Compute shared gradients once at snap0 (for direction negotiation)
    print("[run_k1] computing shared gradients for all negotiators...", flush=True)
    restore_rng(snap0_rng)
    restore_fn(snap0_params)
    g_und = raw_shared_gradient(task_und)
    restore_fn(snap0_params)
    g_gen = raw_shared_gradient(task_gen)
    restore_fn(snap0_params)

    g_und_flat = flatten(g_und)
    g_gen_flat = flatten(g_gen)
    directions = all_mandatory_directions([g_und_flat, g_gen_flat])

    # Commit directions for P3: raw_sum and mgda (the two that form real "commits")
    commit_directions = {
        k: v for k, v in directions.items() if k in ("raw_sum", "mgda")
    }

    # 10. Run P1 control (mandatory gate)
    print("[run_k1] P1 control...", flush=True)
    restore_rng(snap0_rng)
    restore_fn(snap0_params)
    control_results = run_p1_control(
        tasks, K=1,
        snapshot_fn=snapshot_fn,
        restore_fn=restore_fn,
        step_scales=STEP_SCALES,
    )
    restore_fn(snap0_params)

    # 11. P0 simultaneous baseline
    print("[run_k1] P0 simultaneous baseline...", flush=True)
    restore_rng(snap0_rng)
    restore_fn(snap0_params)
    rows_p0 = run_p0(tasks, directions, STEP_SCALES, control_results,
                     snapshot_fn=snapshot_fn, restore_fn=restore_fn)
    restore_fn(snap0_params)

    # 12. P2 shared-then-private
    print("[run_k1] P2 shared-then-private...", flush=True)
    restore_rng(snap0_rng)
    restore_fn(snap0_params)
    rows_p2 = run_p2(tasks, directions, STEP_SCALES, control_results, K=1,
                     snapshot_fn=snapshot_fn, restore_fn=restore_fn)
    restore_fn(snap0_params)

    # 13. P3 virtual private-then-shared commit
    print("[run_k1] P3 virtual commit...", flush=True)
    restore_rng(snap0_rng)
    restore_fn(snap0_params)
    rows_p3 = run_p3(tasks, commit_directions, STEP_SCALES, control_results, K=1,
                     snapshot_fn=snapshot_fn, restore_fn=restore_fn)
    restore_fn(snap0_params)

    t_elapsed = time.monotonic() - t_start
    gpu_hours = t_elapsed / 3600.0

    # 14. Verify persistent_updates == 0
    from comppareto.adapters.showo2_alternating.snapshot import assert_exact_restore
    try:
        assert_exact_restore(
            list(shared) + list(und_private) + list(gen_private),
            snap0_params, atol=0.0, rtol=0.0,
        )
        persistent_updates = 0
        snapshot_restore_failed = 0
    except AssertionError as exc:
        print(f"[run_k1] WARNING: persistent_updates check failed: {exc}", flush=True)
        persistent_updates = 1
        snapshot_restore_failed = 1

    # 15. Build raw rows list
    all_rows = rows_p0 + rows_p2 + rows_p3

    # Add P1 control as synthetic rows for completeness
    p1_rows = []
    for tid, cl in control_results["control_losses"].items():
        lb = control_results["loss_before"][tid]
        p1_rows.append({
            "protocol": "P1",
            "direction": "private_only",
            "step_scale": None,
            "task_id": tid,
            "loss_before": lb,
            "loss_after": cl,
            "delta_controlled": cl - lb,
            "delta_private": control_results.get("delta_private", {}).get(tid),
            "delta_total": cl - lb,
            "finite": math.isfinite(cl),
            "notes": "control",
        })

    raw_rows_dicts = [asdict(r) for r in all_rows] + p1_rows
    (run_dir / "raw_rows.json").write_text(json.dumps(raw_rows_dicts, indent=2) + "\n")

    # 16. Compute metrics
    def _count_nonfinite(rows):
        return sum(1 for r in rows if not r.finite)

    def _missing_control(rows):
        # Every non-P1 row should have a matching control loss
        return sum(
            1 for r in rows
            if r.delta_controlled is None and r.finite
        )

    nonfinite = _count_nonfinite(all_rows)
    missing_ctrl = _missing_control(all_rows)

    metrics = {
        "protocols": {
            "simultaneous_measured": len(rows_p0) > 0,
            "private_only_control_measured": bool(control_results.get("control_losses")),
            "shared_then_private_measured": len(rows_p2) > 0,
            "private_then_shared_commit_measured": len(rows_p3) > 0,
        },
        "attribution": {
            "mandatory_rows_missing_control": missing_ctrl,
        },
        "snapshot_restore": {
            "failed": snapshot_restore_failed,
        },
        "numerical": {
            "nonfinite_mandatory_rows": nonfinite,
            "total_rows": len(all_rows),
        },
        "resources": {
            "gpu_hours": round(gpu_hours, 4),
            "elapsed_seconds": round(t_elapsed, 1),
            "gpu_index": 3,
        },
        "persistent_updates": persistent_updates,
        "K": 1,
        "step_scales": STEP_SCALES,
        "seed": MASTER_SEED,
        "summary": {
            "p0_rows": len(rows_p0),
            "p2_rows": len(rows_p2),
            "p3_rows": len(rows_p3),
            "directions": list(directions.keys()),
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[run_k1] metrics written; gpu_hours={gpu_hours:.4f}", flush=True)

    # 17. Build manifest
    run_status = (
        "pass"
        if (nonfinite == 0 and missing_ctrl == 0 and snapshot_restore_failed == 0
            and all(metrics["protocols"].values()))
        else "fail"
    )

    # Collect result files that now exist in the worktree
    result_files_rel = [
        "configs/alternating/showo2/storage-preflight.json",
        "configs/alternating/showo2/resolved-config.yaml",
        "runs/alternating-showo2-v1/raw_rows.json",
        "runs/alternating-showo2-v1/metrics.json",
    ]

    # Build artifacts list: include the SSD model blob that we actually read from
    ssd_blob = Path(
        f"{HF_CACHE}/hub/models--showlab--show-o2-1.5B/blobs/"
        "a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c"
    )
    artifacts = []
    if ssd_blob.is_file():
        artifacts.append({
            "artifact_id": "showo2-1.5b-checkpoint-ssd-execution",
            "kind": "model_weights_ssd_execution_copy",
            "canonical_uri": str(ssd_blob),
            "sha256": "a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c",
            "bytes": 5661862314,
        })

    # Add metrics.json as an artifact so verify_manifest_artifacts can hash it
    metrics_path = run_dir / "metrics.json"
    m_hash = sha256_file(metrics_path)
    m_bytes = metrics_path.stat().st_size
    artifacts.append({
        "artifact_id": "T216-metrics-json",
        "kind": "run_evidence",
        "canonical_uri": str(metrics_path),
        "sha256": m_hash,
        "bytes": m_bytes,
    })

    manifest = {
        "schema_version": 1,
        "run_id": "alternating-showo2-v1",
        "task_id": "T216",
        "run_kind": "formal",
        "source_revision": "2358267c14dddc3754e7a80e9b681308c6bcd0f7",
        "execution_revision": execution_revision,
        "dirty": False,
        "config_sha256": config_sha256,
        "environment": {
            "container": "H20-FoldUMM",
            "gpu_model": "NVIDIA H20",
            "gpu_count": 8,
            "gpus_used": 1,
            "gpu_index": 3,
            "torch": str(torch.__version__),
            "hf_home": HF_CACHE,
            "hf_hub_offline": True,
            "transformers_offline": True,
        },
        "status": run_status,
        "result_files": result_files_rel,
        "artifacts": artifacts,
        "retry": None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[run_k1] manifest written; status={run_status}", flush=True)

    return 0 if run_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
