from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_synthetic_cli_writes_passing_manifest(tmp_path: Path) -> None:
    output = tmp_path / "t1_manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "comppareto.synthetic",
            "--config",
            "configs/t1_synthetic.json",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output.read_text())
    assert manifest["status"] == "pass"
    assert set(manifest["checks"]) == {
        "exact_elimination",
        "conditional_rescaling",
        "common_descent",
        "indefinite_rejection",
        "selector_validation",
        "attainable_gain",
        "negotiation_rescaling",
    }
    assert all(item["passed"] for item in manifest["checks"].values())
    assert len(manifest["config_sha256"]) == 64
    assert manifest["environment"]["python"]
    assert manifest["environment"]["numpy"]
    assert manifest["environment"]["scipy"]
    assert "revision" in manifest["git"]
    assert isinstance(manifest["git"]["dirty"], bool)
