# T155 result summary

Task: `tasks/T155-exact-finite-response-oracle.md`. Branch: `agent/T155-exact-oracle`.

This report reflects the state after `reports/T155/local-review.md` R1-R6
remediation (execution_revision `b8a1f6b6fe8b861b8825bb82299dee03f8e4a667`).

## What was built

- First report (mathematical specification), published before implementation: `docs/theory/oracle-spec.md`.
- Implementation, fourteen modules under `src/comppareto/oracle/`: `selectors.py`, `tasks.py`, `noise.py`, `sgd.py`, `momentum.py`, `hypergradient.py`, `crosscheck.py`, `generation.py`, `seeds.py`, `stability.py`, `case.py`, `manifest.py`, `pareto.py`, `highprecision.py`, plus the sweep runner `sweep.py`.
- Test suite, **100 tests** under `tests/oracle/` (10 files: `test_selectors.py`, `test_sgd.py`, `test_momentum.py`, `test_generation.py`, `test_stability.py`, `test_crosscheck_tolerances.py`, `test_sweep.py`, `test_case.py`, `test_pareto.py`, `test_highprecision.py`, plus `conftest.py`/`_helpers.py`), all passing.
- Resolved sweep configuration: `configs/oracle/baseline.yaml` (`config_seed: 20260827155`).
- Run evidence: `runs/oracle-20260827-baseline/` (`config.yaml`, `notes.md`, `manifest.json`, `case-records.json`, `summary.json`, `failure_ledger.json`). Per R1, `manifest.json` is now the schema-valid envelope required by `schemas/run-manifest.schema.json` (`run_kind: formal`, `dirty: false`); the flat per-case array lives in `case-records.json`, one of `manifest.json`'s `result_files`.
- No new dependency added; `pyproject.toml` (outside `allowed_paths`) is unchanged. Only `numpy`, `scipy`, `PyYAML` used.

## Sweep metrics

- Grid: 6 graph families x 2 optimizers (SGD; momentum, beta in {0.5, 0.9}) x 4 horizons (K in {1,3,5,10}) x 2 stability regimes x 3 seeds per cell = **288 cases**.
- **286/288 cases passed every check (99.31%)**; 2 failed (see failure-ledger.md), unchanged from the original submission after R5's higher-precision recheck.
- Elapsed: **24.8s** single CPU core (up from 9.26s in the original submission; the increase is R3's added per-case active-set enumeration over up to `2^8-1=255` subsets for the 8-task cases -- still within the section-10 "low tens of seconds to a few minutes" budget).
- Output size: `manifest.json` **1.6KB** (schema envelope only); `case-records.json` **7.7MB** (up from the original inlined 2.7MB, driven by R2's per-task `detail` payloads and R3's `pareto_reference` -- still within the section-10 "a few MB" order-of-magnitude estimate).
- Detailed-trajectory subset (section 11): **6 cases** (`case_index` 33, 60, 96, 144, 210, 240), with zero coverage gaps across family/optimizer/horizon/stability_regime (`summary.json.detailed_subset_coverage_gaps == {}`).
- Selector contract: 100% of generated `P_i` pass the single-lift/no-duplicate-coordinate contract (enforced at construction in `generate_tasks`, exercised across all 288 cases with no exception raised).
- All six graph families realize their defining structural property in every case (`test_build_incidence_matches_family_property`, parametrized over all 6 families, and re-verified live inside every sweep case via `build_incidence`'s own post-generation assertion).
- Common-descent/Pareto reference (R3): every one of the 288 case records carries a `pareto_reference` computed from the tasks' real lifted exact gradients (not random probe directions), with the exact active-set solver and the independent Frank-Wolfe solver in `src/comppareto/oracle/pareto.py` cross-checked against each other on every case.

## Per-check pass rates (288 cases x 1-8 tasks per case; check applies per task)

| Check | Tolerance | Result |
|---|---|---|
| analytic state vs. independent unroll | rel <=1e-10 | pass on every task in every case (typical error ~1e-15 to 1e-16) |
| analytic hypergradient vs. independently implemented reverse-mode differentiation | rel <=1e-9 | pass on every task in every case (typical error ~1e-15 to 1e-16) |
| central finite-difference directional derivative, h in [1e-6, 1e-2] | rel <=1e-6 | pass on every task in every case within the required envelope |
| exact quadratic loss-change identity vs. direct evaluation | rel <=1e-9 | pass except the 2 tasks in the 2 failing cases (see failure-ledger.md) |

## Independent runs of the test suite

`PYTHONPATH=src python3 -m pytest tests/oracle/ -v` -> 100 passed, 0 failed, in ~8-10s. This includes the bugs caught and fixed during development, including during this R1-R6 remediation: the original hypergradient sign bug (recorded in `docs/theory/oracle-spec.md`'s section-4 correction and `hypergradient.py`'s `gradient_at_point` rename), and a newly-found scale-dependent tolerance bug in `src/comppareto/oracle/pareto.py::min_norm_point_active_set` (an absolute KKT/active-consistency acceptance gate spuriously rejected the true global optimum on case_index 47, whose Gram-matrix magnitudes reach ~1.2e11; fixed by selecting the minimum-objective lambda-feasible candidate instead, which is provably the exact optimum for this convex QP with no residual gate needed -- see commit `b8a1f6b` and `runs/oracle-20260827-baseline/notes.md`). Both are direct evidence that the mandated independent cross-check discipline (section 7) functions as designed.

`PYTHONPATH=src python3 -m pytest tests/ -q` (full repo suite) -> 129 passed, 1 failed. The 1 failure, `tests/repo_state/test_cli.py::test_cli_validates_repository`, is a pre-existing bug outside T155's `allowed_paths`, not a regression from this remediation: it hardcodes `run_manifests=pass manifests=1`, an assumption that became stale the moment the original T155 submission (`e1cb034`) added a second run manifest (`runs/oracle-20260827-baseline/manifest.json`) alongside the pre-existing `runs/t1_synthetic/t1_manifest.json` -- confirmed by running the CLI against the `125482d` revision (before this session's fixes), which was *already* failing this same test, just at an earlier assertion (`returncode == 0`), because the un-regenerated `manifest.json` was still the pre-R1 flat array. Fixing the count assertion requires editing `tests/repo_state/test_cli.py`, which is outside `allowed_paths` for T155; flagged for local-reviewer decision rather than resolved unilaterally.

## Process note

An earlier draft of this work-in-progress added a status paragraph to `PROGRESS.md`. Per `reports/README.md` ("Only the local review side updates `PROGRESS.md`, accepts a task, stops a route, or opens successor tasks"), that edit has been reverted; the equivalent status information is instead reported here for the local reviewer to fold into `PROGRESS.md` at their discretion.

## Conclusion

**Supports gate**, with 2 documented, non-tolerance-relaxed loss-change exceptions (case_index 41, 287; see failure-ledger.md and claim-check.md) and 1 pre-existing, out-of-scope test-assertion staleness (`test_cli_validates_repository`'s manifest count) flagged for local-reviewer decision. The Adam-like extension (section 12) is deferred, as declared in the first report, and is not part of this submission.
