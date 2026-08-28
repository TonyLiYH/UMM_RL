"""Sweep runner: enumerate the section-8 grid, execute every case, and emit the
deterministic run manifest + failure ledger required by sections 10-11 of
``docs/theory/oracle-spec.md``.

Runnable as ``python -m comppareto.oracle.sweep --config <yaml> --out <dir>``.
Uses only already-declared dependencies (``numpy``, ``PyYAML``) since
``pyproject.toml`` is outside T155's ``allowed_paths``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy
import yaml

from comppareto.oracle.case import CaseSpec, run_case
from comppareto.oracle.manifest import build_run_manifest, case_record

COVERAGE_ATTRS = ("family", "optimizer", "horizon", "stability_regime")


def enumerate_cases(config: dict) -> list[CaseSpec]:
    families = config["families"]
    optimizers = config["optimizers"]
    momentum_betas = config["momentum_betas"]
    horizons = config["horizons"]
    stability_regimes = config["stability_regimes"]
    seeds_per_cell = config["seeds_per_cell"]
    noise_kinds = config["noise_kinds"]
    private_dims_pool = config["private_dims"]
    condition_numbers = config["condition_numbers"]
    coupling_rank_modes = config["coupling_rank_modes"]
    cosine_targets = config["gradient_cosine_targets"]
    scale_targets = config["gradient_scale_targets"]
    mu = config["mu"]
    config_seed = config["config_seed"]
    max_private_dim = max(private_dims_pool)

    cases: list[CaseSpec] = []
    case_index = 0
    for family, fam_cfg in families.items():
        num_tasks = fam_cfg["num_tasks"]
        num_blocks = fam_cfg["num_blocks"]
        block_width = fam_cfg["block_width"]
        for optimizer in optimizers:
            for horizon in horizons:
                for stability_regime in stability_regimes:
                    for seed_idx in range(seeds_per_cell):
                        noise_kind = noise_kinds[seed_idx % len(noise_kinds)]
                        beta = momentum_betas[seed_idx % len(momentum_betas)] if optimizer == "momentum" else None
                        private_dims = tuple(
                            private_dims_pool[(case_index + t) % len(private_dims_pool)] for t in range(num_tasks)
                        )
                        condition_number = condition_numbers[case_index % len(condition_numbers)]
                        rank_mode = coupling_rank_modes[case_index % len(coupling_rank_modes)]
                        # "full" is clipped to min(p_i, d_i) per task inside generate_coupling.
                        coupling_rank = 1 if rank_mode == "low" else max_private_dim + num_blocks * block_width
                        noise_rho = 0.3 if case_index % 2 == 0 else 0.7
                        cases.append(
                            CaseSpec(
                                case_index=case_index,
                                config_seed=config_seed,
                                family=family,
                                num_tasks=num_tasks,
                                num_blocks=num_blocks,
                                block_width=block_width,
                                private_dims=private_dims,
                                condition_number=condition_number,
                                coupling_rank=coupling_rank,
                                mu=mu,
                                optimizer=optimizer,
                                beta=beta,
                                horizon=horizon,
                                stability_regime=stability_regime,
                                noise_kind=noise_kind,
                                noise_sigma=0.1,
                                noise_rho=noise_rho,
                                gradient_cosine_target=cosine_targets[case_index % len(cosine_targets)],
                                gradient_scale_target=scale_targets[case_index % len(scale_targets)],
                                keep_full_detail=False,
                            )
                        )
                        case_index += 1
    return cases


def select_detailed_subset(cases: list[CaseSpec], budget: int) -> tuple[set[int], dict[str, set]]:
    """Cover every family/optimizer/horizon/stability_regime value at least once.

    For each (attribute, value) target still uncovered, pick whichever case
    realizing that value also closes the most *other* currently-open targets
    -- e.g. the first ``full_overlap`` case chosen is the one that happens to
    pair ``optimizer=momentum``, ``horizon=3``, ``regime=unstable`` in the
    same case, closing four targets in one pick instead of one. This keeps
    the curated subset near the low end of oracle-spec.md section 11's 5-10
    case target instead of the ~10 a naive single-attribute-at-a-time greedy
    needs for this 6x2x4x2 grid.
    """

    covered: dict[str, set] = {attr: set() for attr in COVERAGE_ATTRS}
    chosen: set[int] = set()
    all_values = {attr: sorted({getattr(s, attr) for s in cases}, key=str) for attr in COVERAGE_ATTRS}
    targets = [(attr, value) for attr in COVERAGE_ATTRS for value in all_values[attr]]

    for attr, value in targets:
        if value in covered[attr] or len(chosen) >= budget:
            continue
        candidates = [s for s in cases if getattr(s, attr) == value]
        best = max(candidates, key=lambda s: sum(1 for a in COVERAGE_ATTRS if getattr(s, a) not in covered[a]))
        chosen.add(best.case_index)
        for a in COVERAGE_ATTRS:
            covered[a].add(getattr(best, a))
    return chosen, covered


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
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def run_sweep(config: dict, source_revision: str) -> dict:
    cases = enumerate_cases(config)
    full_coverage = {attr: set() for attr in COVERAGE_ATTRS}
    for spec in cases:
        for attr in COVERAGE_ATTRS:
            full_coverage[attr].add(getattr(spec, attr))

    detailed_budget = config.get("detailed_subset_size", 8)
    detailed_indices, achieved_coverage = select_detailed_subset(cases, detailed_budget)
    cases = [replace(spec, keep_full_detail=spec.case_index in detailed_indices) for spec in cases]

    records = []
    failures = []
    start = time.perf_counter()
    for spec in cases:
        try:
            result = run_case(spec)
        except (FloatingPointError, OverflowError) as exc:
            failures.append(
                {
                    "case_index": spec.case_index,
                    "family": spec.family,
                    "optimizer": spec.optimizer,
                    "horizon": spec.horizon,
                    "stability_regime": spec.stability_regime,
                    "status": "unstable-overflow",
                    "reason": str(exc),
                }
            )
            continue
        record = case_record(result, source_revision)
        records.append(record)
        if not record["all_passed"]:
            failures.append(
                {
                    "case_index": spec.case_index,
                    "family": spec.family,
                    "optimizer": spec.optimizer,
                    "horizon": spec.horizon,
                    "stability_regime": spec.stability_regime,
                    "status": "check-failed",
                    "output_hash": record["output_hash"],
                }
            )
    elapsed = time.perf_counter() - start

    coverage_gaps = {attr: sorted(full_coverage[attr] - achieved_coverage[attr]) for attr in COVERAGE_ATTRS}
    coverage_gaps = {k: v for k, v in coverage_gaps.items() if v}

    summary = {
        "config_seed": config["config_seed"],
        "source_revision": source_revision,
        "total_cases": len(cases),
        "passed_cases": sum(1 for r in records if r["all_passed"]),
        "failed_cases": len(failures),
        "detailed_subset": sorted(detailed_indices),
        "detailed_subset_coverage_gaps": coverage_gaps,
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
            "kind": "oracle-run-artifact",
            "canonical_uri": str((args.out / name).relative_to(repo_root)),
            "sha256": _sha256_file(args.out / name),
            "bytes": (args.out / name).stat().st_size,
        }
        for name in written
    ]
    status = "pass" if outcome["summary"]["failed_cases"] == 0 else "fail"
    run_id = args.run_id or args.out.name

    manifest = build_run_manifest(
        run_id=run_id,
        task_id="T155",
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
