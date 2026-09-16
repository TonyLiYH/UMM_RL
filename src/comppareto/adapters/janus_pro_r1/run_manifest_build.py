"""Assemble ``runs/janus-pro-r1-stack-v1/manifest.json`` (T720).

CPU-only, no model inference or training -- mirrors
``comppareto.data.run_artifacts``'s pattern exactly: hash real files on
disk, then call :func:`comppareto.oracle.manifest.build_run_manifest` to
assemble a schema-valid ``manifest.json``
(``schemas/run-manifest.schema.json``).

Path scope note (why this script does *not* reference ``/dockerdata``):
the GPU smoke runs themselves execute inside the Taiji container against
``/dockerdata/t720-janus-pro-r1/...`` (the verified local-SSD execution
copy -- see ``configs/janus-pro-r1/admission/storage-preflight.json`` and
``src/comppareto/adapters/janus_pro_r1/env.py``'s module docstring), but
``/dockerdata`` is a container-exclusive mount not reachable from the shell
that runs ``scripts/validate_task_submission.sh`` (confirmed directly:
``ls /dockerdata`` fails outside the container, while both
``/apdcephfs_cq7`` and ``/apdcephfs_cq9`` are mounted both inside and
outside it). Since ``scripts/verify_manifest_artifacts.py`` re-hashes every
``manifest.json`` artifact's ``canonical_uri`` from wherever it is
invoked, every artifact this script lists points at a CQ7-canonical or
in-repo path so the acceptance contract's ``artifact-hashes`` command can
actually re-verify it. The two multi-GB smoke checkpoint ``.pt`` files
(``sft_smoke_trainable_state.pt``, ``grpo_smoke_state.pt``) live only on
``/dockerdata`` and are therefore *not* listed as schema artifacts here;
their sha256/bytes/reload-check results are still real, reported in
``runs/janus-pro-r1-stack-v1/metrics.json`` and
``runs/janus-pro-r1-stack-v1/notes.md`` -- just not mechanically
re-hashable outside the container, which is documented rather than hidden.

Artifact-list scope note (mirrors the ``admission-showo2-2026-08-28``
precedent exactly): only *real external resources consumed or produced by
the run* are listed as ``artifacts`` (model weight shards, the data shard,
the two smoke-summary evidence files). The admission/report documents
themselves (``source-lock.yaml``, ``environment-{sft,rl}-lock.md``,
``storage-preflight.json``, ``artifact-verification.json``,
``metrics.json``) are listed only in ``result_files`` -- exactly as
``admission-showo2-2026-08-28/manifest.json`` does for its own equivalent
documents -- which also avoids a circular hashing dependency
(``artifact-verification.json`` is *produced by* hashing this very
manifest, so it cannot also be an input to it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from comppareto.oracle.manifest import build_run_manifest

RUN_ID = "janus-pro-r1-stack-v1"
TASK_ID = "T720"


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _config_sha256(source_paths: list[Path]) -> str:
    entries = []
    for path in sorted(source_paths):
        sha, size = _sha256_and_size(path)
        entries.append({"path": str(path), "sha256": sha, "bytes": size})
    encoded = json.dumps(entries, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_head(repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--janus-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--internvl-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--data-shard", required=True, type=Path)
    parser.add_argument("--admission-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--status", required=True, choices=["pass", "fail", "blocked"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    raw_sources = {
        "janus-pro-7b-pytorch-model-00001": (
            args.janus_checkpoint_dir / "pytorch_model-00001-of-00002.bin",
            "model_weights",
        ),
        "janus-pro-7b-pytorch-model-00002": (
            args.janus_checkpoint_dir / "pytorch_model-00002-of-00002.bin",
            "model_weights",
        ),
        "internvl2_5-8b-model-00001": (
            args.internvl_checkpoint_dir / "model-00001-of-00004.safetensors",
            "model_weights",
        ),
        "internvl2_5-8b-model-00002": (
            args.internvl_checkpoint_dir / "model-00002-of-00004.safetensors",
            "model_weights",
        ),
        "internvl2_5-8b-model-00003": (
            args.internvl_checkpoint_dir / "model-00003-of-00004.safetensors",
            "model_weights",
        ),
        "internvl2_5-8b-model-00004": (
            args.internvl_checkpoint_dir / "model-00004-of-00004.safetensors",
            "model_weights",
        ),
        "internvl2_5-8b-tokenizer-model": (
            args.internvl_checkpoint_dir / "tokenizer.model",
            "tokenizer",
        ),
        "janus-pro-r1-data-shard-0000-of-0524": (args.data_shard, "raw_source"),
        "sft-smoke-summary": (args.evidence_dir / "sft-smoke-summary.json", "run_evidence"),
        "grpo-smoke-summary": (args.evidence_dir / "grpo-smoke-summary.json", "run_evidence"),
    }
    # config_sha256 covers only the admission pinning documents (the actual
    # "configuration" this run was executed against) -- NOT
    # storage-preflight.json/artifact-verification.json/metrics.json, which
    # are outputs of the run, not inputs to it (same distinction the
    # admission-showo2-2026-08-28 precedent draws between its single
    # resolved-config artifact and its separately-listed report/metrics
    # result_files).
    config_source_paths = [
        args.admission_dir / "source-lock.yaml",
        args.admission_dir / "environment-sft-lock.md",
        args.admission_dir / "environment-rl-lock.md",
    ]

    artifacts: list[dict[str, Any]] = []
    for artifact_id, (path, kind) in raw_sources.items():
        sha, size = _sha256_and_size(path)
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "kind": kind,
                "canonical_uri": str(path),
                "sha256": sha,
                "bytes": size,
            }
        )

    config_sha256 = _config_sha256(config_source_paths)
    execution_revision = _git_head(args.repo_root)

    result_files = [
        "configs/janus-pro-r1/admission/source-lock.yaml",
        "configs/janus-pro-r1/admission/environment-sft-lock.md",
        "configs/janus-pro-r1/admission/environment-rl-lock.md",
        "configs/janus-pro-r1/admission/storage-preflight.json",
        "configs/janus-pro-r1/admission/artifact-verification.json",
        "reports/T720/first-report.md",
        "reports/T720/reuse-map.md",
        "reports/T720/result-summary.md",
        "reports/T720/claim-check.md",
        "reports/T720/failure-ledger.md",
        "runs/janus-pro-r1-stack-v1/manifest.json",
        "runs/janus-pro-r1-stack-v1/metrics.json",
        "runs/janus-pro-r1-stack-v1/notes.md",
        "runs/janus-pro-r1-stack-v1/evidence/sft-smoke-summary.json",
        "runs/janus-pro-r1-stack-v1/evidence/grpo-smoke-summary.json",
    ]

    manifest = build_run_manifest(
        run_id=RUN_ID,
        task_id=TASK_ID,
        run_kind="formal",
        source_revision=args.source_revision,
        execution_revision=execution_revision,
        dirty=False,
        config_sha256=config_sha256,
        environment={
            "container": "H20-FoldUMM",
            "gpu_model": "NVIDIA H20",
            "gpu_index_used": 1,
            "gpus_used": 1,
            "role": "janus-pro-r1-sft-grpo-stack-smoke",
            "sft_venv": "/dockerdata/t720-janus-pro-r1/venvs/sft",
            "rl_venv": "/dockerdata/t720-janus-pro-r1/venvs/rl",
            "hf_home": "/dockerdata/t720-janus-pro-r1/hf_home",
            "hf_hub_offline": True,
            "transformers_offline": True,
        },
        status=args.status,
        result_files=result_files,
        artifacts=artifacts,
        retry=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
