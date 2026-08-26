from __future__ import annotations

import json
from pathlib import Path

from comppareto.repo_state.runs import validate_run_manifest


SCHEMA = Path("schemas/run-manifest.schema.json")
REVISION = "b" * 40
CONFIG_HASH = "c" * 64
ARTIFACT_HASH = "d" * 64


def valid_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "T110-test-run",
        "task_id": "T110",
        "run_kind": "formal",
        "source_revision": REVISION,
        "execution_revision": REVISION,
        "dirty": False,
        "config_sha256": CONFIG_HASH,
        "environment": {"python": "3.13.0"},
        "status": "pass",
        "result_files": ["results/summary.json"],
        "artifacts": [
            {
                "artifact_id": "summary",
                "kind": "metrics",
                "canonical_uri": "storage://umm-rl/T110/summary.json",
                "sha256": ARTIFACT_HASH,
                "bytes": 128
            }
        ],
        "retry": None
    }


def write_manifest(root: Path, payload: dict[str, object]) -> Path:
    run_dir = root / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    (root / "results").mkdir()
    (root / "results" / "summary.json").write_text("{}\n")
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_accepts_committed_t1a_manifest() -> None:
    result = validate_run_manifest(
        Path("runs/t1_synthetic/t1_manifest.json"),
        SCHEMA,
        Path("."),
    )

    assert result.errors == ()


def test_rejects_missing_task_id(tmp_path: Path) -> None:
    payload = valid_manifest()
    del payload["task_id"]
    path = write_manifest(tmp_path, payload)

    result = validate_run_manifest(path, SCHEMA, tmp_path)

    assert any("'task_id' is a required property" in error for error in result.errors)


def test_rejects_dirty_formal_run(tmp_path: Path) -> None:
    payload = valid_manifest()
    payload["dirty"] = True
    path = write_manifest(tmp_path, payload)

    result = validate_run_manifest(path, SCHEMA, tmp_path)

    assert "T110-test-run: formal run must have dirty=false" in result.errors


def test_rejects_missing_result_reference(tmp_path: Path) -> None:
    payload = valid_manifest()
    payload["result_files"] = ["results/missing.json"]
    path = write_manifest(tmp_path, payload)

    result = validate_run_manifest(path, SCHEMA, tmp_path)

    assert "T110-test-run: missing result file results/missing.json" in result.errors


def test_rejects_zero_sized_artifact(tmp_path: Path) -> None:
    payload = valid_manifest()
    payload["artifacts"][0]["bytes"] = 0  # type: ignore[index]
    path = write_manifest(tmp_path, payload)

    result = validate_run_manifest(path, SCHEMA, tmp_path)

    assert any("0 is less than the minimum of 1" in error for error in result.errors)


def test_rejects_retry_without_failure_reason(tmp_path: Path) -> None:
    payload = valid_manifest()
    payload["retry"] = {"parent_run_id": "failed-run"}
    path = write_manifest(tmp_path, payload)

    result = validate_run_manifest(path, SCHEMA, tmp_path)

    assert any(
        "'failure_reason' is a required property" in error
        for error in result.errors
    )
