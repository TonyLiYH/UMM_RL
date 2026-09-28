# T110 Sweep Results

**Run ID**: t1b-overlap-20260928  
**Date**: 2026-09-28  
**Source revision**: d82adf4ffd83c013900e7125d10f5a5d5231addc  
**Config**: configs/t1b/overlap-family.yaml (sha256: 996359c7fc402cf12600d12c95acd4015553fa969b5b6256afc98084113ea6a6)

## Summary

| Regime   | Total | Passed | Failed | Dims covered |
|----------|-------|--------|--------|--------------|
| disjoint | 100   | 100    | 0      | 2–32 (all 31) |
| partial  | 100   | 100    | 0      | 2–32 (all 31) |
| full     | 100   | 100    | 0      | 2–32 (all 31) |
| **TOTAL**| **300** | **300** | **0** | — |

Pass/fail gate: **PASS** (zero unexplained selector or objective mismatches).

Failure ledger: empty (`[]`).

## Environment

- Python 3.11.6, NumPy 2.4.6, SciPy 1.17.1
- Platform: Linux-6.6.119-49.20.tl4.x86_64-x86_64-with-glibc2.38
- Elapsed: 1.13 seconds (pure CPU, single core)

## Checks verified per case

For each of the 300 cases (3 regimes × 100 seeds, global_dim cycling [2, 32]):

1. **Block lifting — scatter vs. matmul**: `QuadraticTask.lifted_gradient()` matched hand-coded scatter (index assignment) of local gradient blocks into global coordinates; lifted Hessian `selector.T @ h_xx @ selector` matched scatter of h_xx. Tolerance: 1e-12.

2. **Objective changes — Schur identity + linear-CG cross-check**: `compensated_change` closed-form identity verified numerically; `private_response` cross-checked against `scipy.sparse.linalg.cg` solve of the private stationarity system (rtol=1e-12, genuinely distinct from the closed-form `numpy.linalg.solve` used internally). Tolerance: 1e-8.

3. **Safe-set support-locality**: for task pair (i, j) with S_i \ S_j nonempty, a perturbation supported only on S_i \ S_j leaves task j's `compensated_change` unchanged (tolerance: 1e-9). Vacuous when S_i \ S_j is empty (always under `full`; never under `disjoint`).

## Notable findings

- Nonlinear CG (`scipy.optimize.minimize(method="CG")`) produced ~9% false failures on ill-conditioned private curvature (condition numbers up to 1000) during development. The closed-form Schur identity remained exact to ~1e-16 in those cases — the issue was the verification solver, not `quadratic.py`. Replaced with linear CG (`scipy.sparse.linalg.cg`), which has a finite-termination guarantee for exact SPD systems.

- scipy 1.17 removed the `tol` kwarg from `scipy.sparse.linalg.cg`; the correct parameters are `rtol` and `atol`.

- The safe-set invariance check is genuinely discriminating: a dedicated sanity-check test confirms that perturbations on shared coordinates (not S_i \ S_j) do change the compensated objective of the affected task.

## Artifact provenance

| File | SHA-256 |
|------|---------|
| case-records.json | f43df022b503fc6907e313be31a308e552f0d5dd575ebdcc45c931c60d014a7d |
| summary.json | b5a58f6024b9e6d078a97e50fe1665be743bb7e46d10927ad94dd99a8bccee91 |
| failure_ledger.json | 4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945 |

Full manifest: `runs/t1b-overlap-20260928/manifest.json`

## Validator note

`scripts/validate_task_submission.sh T110` exits 1 because `tasks/contracts/T110.acceptance.yaml` is missing — this is a known infrastructure gap. The path `tasks/contracts/` is outside T110's `allowed_paths` and cannot be created by this agent. All substantive deliverables (code, tests, config, run manifest, report, task file) are present and correct. This gap must be resolved by the reviewer or the infrastructure owner before the validator can pass.
