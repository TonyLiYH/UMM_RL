"""Assemble the ``runs/data-admission-posttraining-v1/`` manifest.json triple.

Computes real sha256/byte-size artifact entries for the raw source files
and the produced manifests, then calls
:func:`comppareto.oracle.manifest.build_run_manifest` to assemble a
schema-valid ``manifest.json`` (``schemas/run-manifest.schema.json``).
This module performs no model inference or training; it only hashes files
and writes JSON/Markdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from comppareto.oracle.manifest import build_run_manifest

RUN_ID = "data-admission-posttraining-v1"
TASK_ID = "T260"


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _config_sha256(source_paths: list[Path]) -> str:
    """Hash the sha256 list of every source/manifest file, sorted by path.

    A pure function of file contents -- deterministic across reruns given
    the same frozen inputs.
    """
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
    parser.add_argument("--coco-zip", required=True, type=Path)
    parser.add_argument("--llava-json", required=True, type=Path)
    parser.add_argument("--diffusiondb-parquet", required=True, type=Path)
    parser.add_argument("--diffusiondb-cache", required=True, type=Path)
    parser.add_argument("--configs-dir", required=True, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--status", required=True, choices=["pass", "fail", "blocked"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    raw_sources = {
        "coco-annotations-zip": (args.coco_zip, "raw_source"),
        "llava-instruct-150k-json": (args.llava_json, "raw_source"),
        "diffusiondb-metadata-parquet": (args.diffusiondb_parquet, "raw_source"),
        "diffusiondb-metadata-jsonl-cache": (args.diffusiondb_cache, "derived_cache"),
    }
    manifest_files = {
        "sources-yaml": (args.configs_dir / "sources.yaml", "config"),
        "diagnostic-manifest": (args.configs_dir / "diagnostic.jsonl", "manifest"),
        "pilot-train-manifest": (args.configs_dir / "pilot_train.jsonl", "manifest"),
        "pilot-validation-manifest": (args.configs_dir / "pilot_validation.jsonl", "manifest"),
        "pilot-meta-manifest": (args.configs_dir / "pilot_meta.jsonl", "manifest"),
        "evaluation-only-manifest": (args.configs_dir / "evaluation_only.yaml", "manifest"),
        "metrics-json": (args.metrics_json, "metrics"),
    }

    artifacts: list[dict[str, Any]] = []
    hashed_paths: list[Path] = []
    for artifact_id, (path, kind) in {**raw_sources, **manifest_files}.items():
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
        hashed_paths.append(path)

    config_sha256 = _config_sha256([p for p, _ in manifest_files.values()])
    execution_revision = _git_head(args.repo_root)

    result_files = [
        "configs/data/posttraining-v1/sources.yaml",
        "configs/data/posttraining-v1/diagnostic.jsonl",
        "configs/data/posttraining-v1/pilot_train.jsonl",
        "configs/data/posttraining-v1/pilot_validation.jsonl",
        "configs/data/posttraining-v1/pilot_meta.jsonl",
        "configs/data/posttraining-v1/evaluation_only.yaml",
        "reports/T260/first-report.md",
        "reports/T260/source-license-audit.md",
        "reports/T260/split-and-decontamination.md",
        "reports/T260/mixture-and-accounting.md",
        "reports/T260/result-summary.md",
        "reports/T260/claim-check.md",
        "reports/T260/failure-ledger.md",
        "runs/data-admission-posttraining-v1/manifest.json",
        "runs/data-admission-posttraining-v1/metrics.json",
        "runs/data-admission-posttraining-v1/notes.md",
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
            "python_implementation": "CPython",
            "role": "data-admission-manifest-build",
            "gpu_hours": 0,
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
