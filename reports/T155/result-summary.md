# T155 result summary

Task: `tasks/T155-exact-finite-response-oracle.md`. Branch: `agent/T155-exact-oracle`.

This report reflects the state after `reports/T155/local-review.md`'s
**second review** R7-R9 remediation (execution_revision
`432ea002fb71a66bbecc22e2a1d59b74777a1769`), superseding the R1-R6 state
described in the review history below.

## What was built

- First report (mathematical specification), published before implementation: `docs/theory/oracle-spec.md`.
- Implementation, fourteen modules under `src/comppareto/oracle/`: `selectors.py`, `tasks.py`, `noise.py`, `sgd.py`, `momentum.py`, `hypergradient.py`, `crosscheck.py`, `generation.py`, `seeds.py`, `stability.py`, `case.py`, `manifest.py`, `pareto.py`, `highprecision.py`, plus the sweep runner `sweep.py`.
- Test suite, **111 tests** under `tests/oracle/` (10 files: `test_selectors.py`, `test_sgd.py`, `test_momentum.py`, `test_generation.py`, `test_stability.py`, `test_crosscheck_tolerances.py`, `test_sweep.py`, `test_case.py`, `test_pareto.py`, `test_highprecision.py`, plus `conftest.py`/`_helpers.py`), all passing (up from 100, driven by R7's `decimal.Decimal` reference tests in `test_highprecision.py` and R8's SciPy-SLSQP independent-check tests in `test_pareto.py`).
- Resolved sweep configuration: `configs/oracle/baseline.yaml` (`config_seed: 20260827155`).
- Run evidence: `runs/oracle-20260827-baseline/` (`config.yaml`, `notes.md`, `manifest.json`, `case-records.json`, `summary.json`, `failure_ledger.json`, and, new in R7, `high_precision_recheck.json`). `manifest.json` is the schema-valid envelope required by `schemas/run-manifest.schema.json` (`run_kind: formal`, `dirty: false`, **`status: pass`**); the flat per-case array lives in `case-records.json`, one of `manifest.json`'s `result_files`.
- No new dependency added; `pyproject.toml` (outside `allowed_paths`) is unchanged. Only `numpy`, `scipy`, `PyYAML` used.

## Sweep metrics

- Grid: 6 graph families x 2 optimizers (SGD; momentum, beta in {0.5, 0.9}) x 4 horizons (K in {1,3,5,10}) x 2 stability regimes x 3 seeds per cell = **288 cases**.
- **286/288 cases passed every check (99.31%)**; 2 failed (see failure-ledger.md) -- the same 2 cases as every prior round, now independently confirmed as pure floating-point cancellation by R7's Decimal reference (see below), rather than merely "not disproven" by the earlier inconclusive longdouble recheck.
- `summary.json` now reports the gate-relevant breakdown separately from the raw `failed_cases` count: `stable_failed_cases: 0`, `pareto_failed_cases: 0`, `high_precision_failed_cases: 0` (all three must be zero for `manifest.json status: pass`; `failed_cases: 2` remains the raw, unmodified ledger count).
- Elapsed: **25.2s** single CPU core (versus 24.8s before this round -- R8's added per-case SciPy-SLSQP solve and R7's two-case Decimal(precision=100) recheck cost is within noise of the existing active-set enumeration cost, still well within the section-10 "low tens of seconds to a few minutes" budget).
- Output size: `manifest.json` ~1.6KB (schema envelope only); `case-records.json` **8.76MB** (up from 7.7MB, driven by R8's added `scipy_qp`/`independent_check` fields inside every case's `pareto_reference`); new artifact `high_precision_recheck.json` **2.75KB**.
- Detailed-trajectory subset (section 11): **6 cases** (`case_index` 33, 60, 96, 144, 210, 240), with zero coverage gaps across family/optimizer/horizon/stability_regime (`summary.json.detailed_subset_coverage_gaps == {}`).
- Selector contract: 100% of generated `P_i` pass the single-lift/no-duplicate-coordinate contract (enforced at construction in `generate_tasks`, exercised across all 288 cases with no exception raised).
- All six graph families realize their defining structural property in every case (`test_build_incidence_matches_family_property`, parametrized over all 6 families, and re-verified live inside every sweep case via `build_incidence`'s own post-generation assertion).
- Common-descent/Pareto reference (R3, gated in R8): every one of the 288 case records carries a `pareto_reference` computed from the tasks' real lifted exact gradients (not random probe directions), containing the exact active-set solution, a diagnostic-only Frank-Wolfe cross-check, and (new in R8) `scipy_qp` -- an independent SciPy-SLSQP constrained-QP solve -- plus an `independent_check` evaluating five preregistered scale-aware thresholds (simplex feasibility, weight nonnegativity, KKT residual, objective gap, combined-gradient discrepancy, all 1e-6, normalized by `max(diag(Gram), 1)`). All 288 cases pass `independent_check` (max residuals across the sweep: kkt=3.5e-9, objective_gap=1.9e-16, combined_gradient=4.1e-9); this check now gates `CaseResult.all_passed` and `summary.json.pareto_failed_cases`.
- Independent high-precision reference (new in R7): `src/comppareto/oracle/highprecision.py::recheck_known_failures` reconstructs the full affine-transition/sensitivity/quadratic-model/loss-change chain for both known failures using `decimal.Decimal` (precision=100, platform-independent, literal per-step recurrence -- not the matrix-power closed form the longdouble mirror uses, for cross-path independence). Both are confirmed `pure_cancellation: true`: case 41's Decimal relative error is 2.37e-13 (vs. float64's 2.73e-9, ~11,500x smaller); case 287's is 4.72e-16 (vs. float64's 1.29e-8, ~27,000,000x smaller). Persisted in full in `high_precision_recheck.json`.

## Per-check pass rates (288 cases x 1-8 tasks per case; check applies per task)

| Check | Tolerance | Result |
|---|---|---|
| analytic state vs. independent unroll | rel <=1e-10 | pass on every task in every case (typical error ~1e-15 to 1e-16) |
| analytic hypergradient vs. independently implemented reverse-mode differentiation | rel <=1e-9 | pass on every task in every case (typical error ~1e-15 to 1e-16) |
| central finite-difference directional derivative, h in [1e-6, 1e-2] | rel <=1e-6 | pass on every task in every case within the required envelope |
| exact quadratic loss-change identity vs. direct evaluation | rel <=1e-9 | pass except the 2 tasks in the 2 failing cases (see failure-ledger.md); both confirmed pure floating-point cancellation by the R7 Decimal(precision=100) reference, not a formula/implementation mismatch |
| independent constrained-QP Pareto/min-norm-point cross-check (SciPy SLSQP vs. exact active-set enumeration) | 5 preregistered scale-aware thresholds, all 1e-6 | pass on all 288 cases (max residuals: kkt=3.5e-9, objective_gap=1.9e-16, combined_gradient=4.1e-9) |

## Independent runs of the test suite

`PYTHONPATH=src python3 -m pytest tests/oracle/ -v` -> 111 passed, 0 failed. This includes the bugs caught and fixed during development, including: the original hypergradient sign bug (recorded in `docs/theory/oracle-spec.md`'s section-4 correction and `hypergradient.py`'s `gradient_at_point` rename); the R3 scale-dependent tolerance bug in `src/comppareto/oracle/pareto.py::min_norm_point_active_set` (an absolute KKT/active-consistency acceptance gate spuriously rejected the true global optimum on case_index 47, whose Gram-matrix magnitudes reach ~1.2e11; fixed by selecting the minimum-objective lambda-feasible candidate instead, which is provably the exact optimum for this convex QP with no residual gate needed); and, new in R8, an SLSQP false-convergence bug (the unnormalized objective on this oracle's up-to-1e13 Gram-matrix scale caused `ftol` to falsely report convergence before reaching the true optimum; fixed by normalizing the objective/gradient by `scale = max(diag(Gram), 1)` before solving). All three are direct evidence that the mandated independent cross-check discipline (section 7) functions as designed.

`PYTHONPATH=src python3 -m pytest tests/ -q` (full repo suite) -> **156 passed, 0 failed**. The previously-flagged pre-existing failure, `tests/repo_state/test_cli.py::test_cli_validates_repository` (a stale hardcoded `manifests=1` assumption, outside T155's `allowed_paths`), no longer reproduces on the current merged `origin/main` tree and is resolved; it is not a T155 fix and is noted here only to close out the earlier flag.

## Process note

An earlier draft of this work-in-progress added a status paragraph to `PROGRESS.md`. Per `reports/README.md` ("Only the local review side updates `PROGRESS.md`, accepts a task, stops a route, or opens successor tasks"), that edit has been reverted; the equivalent status information is instead reported here for the local reviewer to fold into `PROGRESS.md` at their discretion.

## Conclusion

**Supports gate.** Per the task's pass/fail gate ("for every accepted **stable** seeded case ... all failed or unstable seeds remain in the ledger"), the gate is evaluated over stable-regime cases; `summary.json.stable_failed_cases == 0`. The 2 documented loss-change exceptions (case_index 41, 287; see failure-ledger.md and claim-check.md) are both unstable-regime and both now independently confirmed as pure floating-point cancellation (R7's Decimal reference), not an unexplained or systematic mismatch, and remain retained in the ledger unmodified. The independent Pareto/QP cross-check (R8) passes on all 288 cases and is folded into the gate. `manifest.json status: pass`. The Adam-like extension (section 12) is deferred, as declared in the first report, and is not part of this submission.
