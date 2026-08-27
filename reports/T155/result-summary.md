# T155 result summary

Task: `tasks/T155-exact-finite-response-oracle.md`. Branch: `agent/T155-exact-oracle`.

## What was built

- First report (mathematical specification), published before implementation: `docs/theory/oracle-spec.md`.
- Implementation, twelve modules under `src/comppareto/oracle/`: `selectors.py`, `tasks.py`, `noise.py`, `sgd.py`, `momentum.py`, `hypergradient.py`, `crosscheck.py`, `generation.py`, `seeds.py`, `stability.py`, `case.py`, `manifest.py`, plus the sweep runner `sweep.py`.
- Test suite, 52 tests under `tests/oracle/` (8 files: `test_selectors.py`, `test_sgd.py`, `test_momentum.py`, `test_generation.py`, `test_stability.py`, `test_crosscheck_tolerances.py`, `test_sweep.py`, plus `conftest.py`/`_helpers.py`), all passing.
- Resolved sweep configuration: `configs/oracle/baseline.yaml` (`config_seed: 20260827155`).
- Run evidence: `runs/oracle-20260827-baseline/` (`config.yaml`, `notes.md`, `manifest.json`, `summary.json`, `failure_ledger.json`).
- No new dependency added; `pyproject.toml` (outside `allowed_paths`) is unchanged. Only `numpy`, `scipy`, `PyYAML` used.

## Sweep metrics

- Grid: 6 graph families x 2 optimizers (SGD; momentum, beta in {0.5, 0.9}) x 4 horizons (K in {1,3,5,10}) x 2 stability regimes x 3 seeds per cell = **288 cases**.
- **286/288 cases passed every check (99.31%)**; 2 failed (see failure-ledger.md).
- Elapsed: **9.26s** single CPU core (section-10 estimate: "low tens of seconds to a few minutes" -- within budget, no throttling triggered).
- Output size: `manifest.json` **2.7MB** (section-10 estimate: "a few MB for the full sweep" -- within budget).
- Detailed-trajectory subset (section 11): **6 cases** (`case_index` 33, 60, 96, 144, 210, 240), with zero coverage gaps across family/optimizer/horizon/stability_regime (`summary.json.detailed_subset_coverage_gaps == {}`).
- Selector contract: 100% of generated `P_i` pass the single-lift/no-duplicate-coordinate contract (enforced at construction in `generate_tasks`, exercised across all 288 cases with no exception raised).
- All six graph families realize their defining structural property in every case (`test_build_incidence_matches_family_property`, parametrized over all 6 families, and re-verified live inside every sweep case via `build_incidence`'s own post-generation assertion).

## Per-check pass rates (288 cases x 1-8 tasks per case; check applies per task)

| Check | Tolerance | Result |
|---|---|---|
| analytic state vs. independent unroll | rel <=1e-10 | pass on every task in every case (typical error ~1e-15 to 1e-16) |
| analytic hypergradient vs. hand-coded reverse-mode AD | rel <=1e-9 | pass on every task in every case (typical error ~1e-15 to 1e-16) |
| central finite-difference directional derivative, h in [1e-6, 1e-2] | rel <=1e-6 | pass on every task in every case within the required envelope |
| exact quadratic loss-change identity vs. direct evaluation | rel <=1e-9 | pass except the 2 tasks in the 2 failing cases (see failure-ledger.md) |

## Independent runs of the test suite

`PYTHONPATH=src python3 -m pytest tests/oracle/ -v` -> 52 passed, 0 failed, in ~8-9s. This includes the two bugs caught and fixed during development (recorded in `docs/theory/oracle-spec.md`'s section-4 correction and in `src/comppareto/oracle/hypergradient.py`'s `gradient_at_point` rename), which is direct evidence that the mandated three-independent-method cross-check discipline (section 7) functions as designed.

## Process note

An earlier draft of this work-in-progress added a status paragraph to `PROGRESS.md`. Per `reports/README.md` ("Only the local review side updates `PROGRESS.md`, accepts a task, stops a route, or opens successor tasks"), that edit has been reverted; the equivalent status information is instead reported here for the local reviewer to fold into `PROGRESS.md` at their discretion.

## Conclusion

**Supports gate**, with 2 documented, non-tolerance-relaxed exceptions flagged for local-reviewer decision (see failure-ledger.md and claim-check.md). The Adam-like extension (section 12) is deferred, as declared in the first report, and is not part of this submission.
