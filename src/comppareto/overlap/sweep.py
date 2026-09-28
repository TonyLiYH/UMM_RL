"""Sweep runner: generate >=``seeds_per_regime`` seeded cases per overlap regime,
verify each against the independent references in ``comppareto.overlap.verify``,
and emit the run manifest + failure ledger required by
``tasks/T110-overlap-family.md``'s "Required deliverables".

Runnable as ``python -m comppareto.overlap.sweep --config <yaml> --out <dir>``.
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

import numpy
import scipy
import yaml

from comppareto.overlap.manifest import build_run_manifest, case_record
from comppareto.overlap.regimes import REGIMES, OverlapRegime
from comppareto.overlap.verify import DEFAULT_TOLERANCES, generate_case, verify_case


def enumerate_case_keys(config: dict) -> list[tuple[OverlapRegime, int]]:
    seeds_per_regime = config["seeds_per_regime"]
    return [(regime, case_index) for regime in REGIMES for case_index in range(seeds_per_regime)]


def _generation_kwargs(config: dict) -> dict:
    return {
        "max_tasks": config.get("max_tasks", 5),
        "private_dim_range": tuple(config.get("private_dim_range", [1, 6])),
        "mu_range": tuple(config.get("mu_range", [0.05, 2.0])),
        "condition_number_range": tuple(config.get("condition_number_range", [1.0, 1000.0])),
        "gradient_scale_range": tuple(config.get("gradient_scale_range", [0.1, 10.0])),
        "global_dim_cycle": config.get("global_dim_cycle", 30),
        "global_dim_floor": config.get("global_dim_floor", 2),
    }


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_dirty(repo_root: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return bool(status.strip())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_snapshot() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def run_sweep(config: dict, source_revision: str) -> dict:
    config_seed = config["config_seed"]
    tolerances = {**DEFAULT_TOLERANCES, **config.get("tolerance_overrides", {})}
    generation_kwargs = _generation_kwargs(config)

    case_keys = enumerate_case_keys(config)
    records: list[dict] = []
    failures: list[dict] = []
    per_regime_totals: dict[str, int] = {regime: 0 for regime in REGIMES}
    per_regime_passed: dict[str, int] = {regime: 0 for regime in REGIMES}
    global_dims_seen: dict[str, set[int]] = {regime: set() for regime in REGIMES}

    start = time.perf_counter()
    for regime, case_index in case_keys:
        per_regime_totals[regime] += 1
        try:
            case = generate_case(config_seed, case_index, regime, **generation_kwargs)
            global_dims_seen[regime].add(case.global_dim)
            result = verify_case(case, tolerances=tolerances)
        except Exception as exc:  # noqa: BLE001 - any generation/verification exception is a case failure
            failures.append(
                {
                    "regime": regime,
                    "case_index": case_index,
                    "config_seed": config_seed,
                    "status": "exception",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        record = case_record(result, source_revision)
        records.append(record)
        if record["all_passed"]:
            per_regime_passed[regime] += 1
        else:
            failures.append(
                {
                    "regime": regime,
                    "case_index": case_index,
                    "config_seed": config_seed,
                    "global_dim": case.global_dim,
                    "num_tasks": case.num_tasks,
                    "status": "check-failed",
                    "output_hash": record["output_hash"],
                }
            )
    elapsed = time.perf_counter() - start

    summary = {
        "config_seed": config_seed,
        "source_revision": source_revision,
        "total_cases": len(case_keys),
        "passed_cases": sum(per_regime_passed.values()),
        "failed_cases": len(failures),
        "per_regime": {
            regime: {
                "total": per_regime_totals[regime],
                "passed": per_regime_passed[regime],
                "failed": per_regime_totals[regime] - per_regime_passed[regime],
                "global_dims_covered": sorted(global_dims_seen[regime]),
            }
            for regime in REGIMES
        },
        "elapsed_seconds": elapsed,
    }
    return {"case_records": records, "summary": summary, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[3]
    execution_revision = _git_head(repo_root)
    source_revision = args.source_revision or execution_revision
    dirty = _git_dirty(repo_root)
    config_sha256 = _sha256_file(args.config)

    outcome = run_sweep(config, source_revision)

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    written = {
        "case-records.json": outcome["case_records"],
        "summary.json": outcome["summary"],
        "failure_ledger.json": outcome["failures"],
    }
    for name, payload in written.items():
        with open(args.out / name, "w") as f:
            json.dump(payload, f, indent=2)

    result_files = [str((args.out / name).relative_to(repo_root)) for name in written]
    artifacts = [
        {
            "artifact_id": Path(name).stem,
            "kind": "overlap-run-artifact",
            "canonical_uri": str((args.out / name).relative_to(repo_root)),
            "sha256": _sha256_file(args.out / name),
            "bytes": (args.out / name).stat().st_size,
        }
        for name in written
    ]
    # Pass/fail gate: "Zero unexplained selector or objective mismatches" ->
    # any failed case blocks a "pass" run status (every failing seed is kept
    # in the ledger either way, per "store every failing seed").
    status = "pass" if outcome["summary"]["failed_cases"] == 0 else "fail"
    run_id = args.run_id or args.out.name

    manifest = build_run_manifest(
        run_id=run_id,
        task_id="T110",
        run_kind="formal" if not dirty else "diagnostic",
        source_revision=source_revision,
        execution_revision=execution_revision,
        dirty=dirty,
        config_sha256=config_sha256,
        environment=_environment_snapshot(),
        status=status,
        result_files=result_files,
        artifacts=artifacts,
        retry=None,
    )
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(outcome["summary"], indent=2))


if __name__ == "__main__":
    main()
