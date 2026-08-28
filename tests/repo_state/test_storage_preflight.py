from __future__ import annotations

from pathlib import Path

from comppareto.repo_state.storage_preflight import (
    classify_filesystem,
    compare_file_hashes,
    parse_filesystem_type,
    verify_manifest_artifacts,
    validate_offline_environment,
)


def test_classifies_network_and_local_filesystems() -> None:
    assert classify_filesystem("nfs4") == "network"
    assert classify_filesystem("lustre") == "network"
    assert classify_filesystem("ext4") == "local"
    assert classify_filesystem("xfs") == "local"
    assert classify_filesystem("overlay") == "local"
    assert classify_filesystem("mysteryfs") == "unknown"


def test_parses_linux_and_macos_filesystem_output() -> None:
    linux = """Filesystem Type 1K-blocks Used Available Use% Mounted on
/dev/nvme0n1p1 ext4 100 20 80 20% /data
"""
    macos = """Filesystem   512-blocks Used Available Capacity iused ifree %iused Mounted on
/dev/disk3s1 100 20 80 20% 1 2 1% /System/Volumes/Data
"""

    assert parse_filesystem_type(linux, platform_name="Linux") == "ext4"
    assert parse_filesystem_type(macos, platform_name="Darwin") == "apfs"


def test_compare_file_hashes_detects_match_and_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    same = tmp_path / "same.bin"
    different = tmp_path / "different.bin"
    source.write_bytes(b"weights")
    same.write_bytes(b"weights")
    different.write_bytes(b"other")

    assert compare_file_hashes(source, same)["matches"]
    assert not compare_file_hashes(source, different)["matches"]


def test_offline_environment_requires_cache_and_offline_flags(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()

    errors = validate_offline_environment(
        {
            "HF_HOME": str(cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    assert errors == []
    assert validate_offline_environment({"HF_HOME": str(cache)}) == [
        "HF_HUB_OFFLINE must be 1",
        "TRANSFORMERS_OFFLINE must be 1",
    ]


def test_verify_manifest_artifacts_reports_hash_size_and_missing(tmp_path: Path) -> None:
    good = tmp_path / "good.bin"
    changed = tmp_path / "changed.bin"
    good.write_bytes(b"good")
    changed.write_bytes(b"changed")
    import hashlib

    manifest = {
        "artifacts": [
            {
                "artifact_id": "good",
                "canonical_uri": str(good),
                "sha256": hashlib.sha256(b"good").hexdigest(),
                "bytes": 4,
            },
            {
                "artifact_id": "changed",
                "canonical_uri": str(changed),
                "sha256": hashlib.sha256(b"original").hexdigest(),
                "bytes": 7,
            },
            {
                "artifact_id": "missing",
                "canonical_uri": str(tmp_path / "missing.bin"),
                "sha256": "0" * 64,
                "bytes": 1,
            },
        ]
    }

    report = verify_manifest_artifacts(manifest)

    assert report["checked"] == 3
    assert report["passed"] == 1
    assert report["failed"] == 2
    assert report["artifacts"][0]["status"] == "pass"
    assert report["artifacts"][1]["status"] == "mismatch"
    assert report["artifacts"][2]["status"] == "missing"
