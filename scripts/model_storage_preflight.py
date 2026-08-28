#!/usr/bin/env python3
"""Emit a JSON preflight record for a model cache or checkpoint directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from comppareto.repo_state.storage_preflight import (
    classify_filesystem,
    parse_filesystem_type,
    validate_offline_environment,
)


def filesystem_type(path: Path) -> str:
    platform_name = platform.system()
    command = ["df", "-T", str(path)] if platform_name == "Linux" else ["df", str(path)]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_filesystem_type(completed.stdout, platform_name=platform_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    path = args.path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    fs_type = filesystem_type(path)
    usage = shutil.disk_usage(path)
    errors = validate_offline_environment(os.environ)
    if classify_filesystem(fs_type) != "local":
        errors.append(
            f"filesystem must be classified local; found {fs_type} "
            f"({classify_filesystem(fs_type)})"
        )
    if usage.free < args.minimum_free_bytes:
        errors.append(
            f"free bytes {usage.free} below required {args.minimum_free_bytes}"
        )

    payload = {
        "path": str(path),
        "filesystem_type": fs_type,
        "filesystem_class": classify_filesystem(fs_type),
        "capacity_bytes": usage.total,
        "free_bytes": usage.free,
        "minimum_free_bytes": args.minimum_free_bytes,
        "environment": {
            name: os.environ.get(name)
            for name in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        },
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
