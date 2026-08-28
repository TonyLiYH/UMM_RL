"""Filesystem, cache, and hash checks before expensive model execution."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping


NETWORK_FILESYSTEMS = {
    "nfs",
    "nfs4",
    "lustre",
    "ceph",
    "cephfs",
    "cifs",
    "smbfs",
    "fuse.sshfs",
}
LOCAL_FILESYSTEMS = {"ext4", "xfs", "btrfs", "zfs", "apfs", "overlay"}


def classify_filesystem(filesystem_type: str) -> str:
    normalized = filesystem_type.strip().lower()
    if normalized in NETWORK_FILESYSTEMS:
        return "network"
    if normalized in LOCAL_FILESYSTEMS:
        return "local"
    return "unknown"


def parse_filesystem_type(output: str, *, platform_name: str) -> str:
    lines = [line for line in output.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("filesystem command returned no data row")
    fields = lines[-1].split()
    if platform_name == "Linux":
        if len(fields) < 2:
            raise ValueError("Linux df output lacks filesystem type")
        return fields[1]
    if platform_name == "Darwin":
        return "apfs"
    raise ValueError(f"unsupported platform for filesystem detection: {platform_name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_file_hashes(source: Path, destination: Path) -> dict[str, object]:
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "source_sha256": source_hash,
        "destination_sha256": destination_hash,
        "source_bytes": source.stat().st_size,
        "destination_bytes": destination.stat().st_size,
        "matches": (
            source_hash == destination_hash
            and source.stat().st_size == destination.stat().st_size
        ),
    }


def validate_offline_environment(
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    values = dict(os.environ if environment is None else environment)
    errors: list[str] = []
    cache = values.get("HF_HOME")
    if not cache:
        errors.append("HF_HOME must be set")
    elif not Path(cache).is_dir():
        errors.append(f"HF_HOME is not a directory: {cache}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if values.get(name) != "1":
            errors.append(f"{name} must be 1")
    return errors


def verify_manifest_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for artifact in manifest.get("artifacts", []):
        path = Path(str(artifact["canonical_uri"]))
        result: dict[str, Any] = {
            "artifact_id": artifact["artifact_id"],
            "canonical_uri": str(path),
            "expected_sha256": artifact["sha256"],
            "expected_bytes": artifact["bytes"],
        }
        if not path.is_file():
            result["status"] = "missing"
            result["actual_sha256"] = None
            result["actual_bytes"] = None
        else:
            actual_hash = sha256_file(path)
            actual_bytes = path.stat().st_size
            result["actual_sha256"] = actual_hash
            result["actual_bytes"] = actual_bytes
            result["status"] = (
                "pass"
                if (
                    actual_hash == artifact["sha256"]
                    and actual_bytes == artifact["bytes"]
                )
                else "mismatch"
            )
        results.append(result)
    passed = sum(item["status"] == "pass" for item in results)
    return {
        "checked": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "artifacts": results,
    }
