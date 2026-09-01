# T155 failure ledger

## Documented, explained unstable-regime failures (do not block the gate)

Per the task's precise pass/fail gate wording ("for every accepted **stable**
seeded case ... all failed or unstable seeds remain in the ledger"), the two
cases below are unstable-regime, documented, and -- as of the second local
review's R7 remediation -- independently confirmed as pure floating-point
cancellation rather than an unexplained or systematic mismatch. They
therefore remain in this ledger (unmodified, no tolerance relaxed, no seed
dropped) without blocking `manifest.json status: pass`; see
`runs/oracle-20260827-baseline/high_precision_recheck.json` for the full
decisive evidence.

### 1. Case_index 41 — `disjoint` / `momentum` (beta cycled) / K=5 / `unstable`

- **Check that failed**: `loss_change` (exact quadratic identity vs. direct evaluation), tolerance 1e-9 relative.
- **Observed relative error**: 2.7e-9 (~2.7x the tolerance).
- **Checks on the same case that passed**: `state` (~1e-15 relative), `hypergradient` (~1e-16 relative), `finite_difference` (all required h in [1e-6,1e-2] steps within envelope).
- **Root cause**: the `stability_regime=unstable` draw for this case realizes a spectral radius of approximately 1.64-1.67 (see `runs/oracle-20260827-baseline/manifest.json`, case_index 41, `spectral_radius` field). Compounded over K=5 SGD-with-momentum steps, this grows the private-state trajectory magnitude enough that the `exact_loss_change` identity's `grad^T d + 1/2 d^T Q d` evaluation becomes a difference of two large, close-in-magnitude terms, amplifying floating-point cancellation. The underlying `grad` and `Q` values are themselves exact to machine precision (confirmed by the passing `state`/`hypergradient` checks on the identical case) — the loss of precision is specific to this one subtraction, not to the state or sensitivity computation.
- **Decisive independent confirmation (second review, R7)**: the original longdouble-only recheck (first review, R5) was platform-dependent and, per the second review, did not decisively establish a cancellation-only explanation. It has been replaced by an independent, platform-independent `decimal.Decimal` (precision=100) reconstruction of the full affine-transition/sensitivity/quadratic-model/loss-change chain via the literal per-step recurrence (`src/comppareto/oracle/highprecision.py::recheck_momentum_loss_change`, a different code path from the state/sensitivity matrix-power closed form, for genuine cross-path independence). Result: Decimal relative error is **2.37e-13**, versus float64's 2.73e-9 — roughly **11,500x smaller**, and the ratio comfortably exceeds the `pure_cancellation` criterion (Decimal relative error <= float64 relative error / 10). `cond(Q) = 977`, trajectory amplification factor ~11.8x. Full breakdown (both cancelling terms, baseline loss magnitude, longdouble/decimal/float64 comparison) persisted in `runs/oracle-20260827-baseline/high_precision_recheck.json`, key `"41"`.
- **Why not reclassified as `unstable-overflow`**: `docs/theory/oracle-spec.md` section 9 carves out deliberately-unstable cases only for actual float64 overflow (inf/nan). No overflow occurred here — all values are finite, well-formed floats; the failure is amplified-but-finite roundoff, which the spec does not exempt.
- **Why not tolerance-relaxed**: `tasks/T155-exact-finite-response-oracle.md`'s failure-and-retry rules require local review before any numerical tolerance change; the 1e-9 tolerance remains unchanged.
- **Disposition**: left as a `check-failed` entry in `failure_ledger.json` and `manifest.json` (`all_passed: false`), retained rather than dropped, per "all failed or unstable seeds remain in the ledger." Because it is unstable-regime and now independently confirmed as pure cancellation (not unexplained/systematic), it does not block `manifest.json status: pass` per the task's gate wording.

### 2. Case_index 287 — `random_sparse` / `momentum` (beta cycled) / K=10 / `unstable`

- **Check that failed**: `loss_change`, tolerance 1e-9 relative.
- **Observed relative error**: 1.3e-8 (~13x the tolerance).
- **Checks on the same case that passed**: `state` (~1e-15 relative), `hypergradient` (~1e-16 relative), `finite_difference` (all required steps within envelope).
- **Root cause**: same mechanism as case 41, at larger scale — spectral radius ~1.64-1.67 compounded over K=10 (twice the horizon of case 41), producing a proportionally larger trajectory-magnitude amplification and a correspondingly larger cancellation error (1.3e-8 vs. 2.7e-9 — roughly consistent with error growing with horizon under a fixed unstable spectral radius).
- **Decisive independent confirmation (second review, R7)**: Decimal(precision=100) relative error is **4.72e-16**, versus float64's 1.29e-8 — roughly **27,000,000x smaller**, far exceeding the `pure_cancellation` threshold. `cond(Q) = 464`, trajectory amplification factor ~173.8x (consistent with the larger horizon driving a larger cancellation than case 41). Full breakdown in `high_precision_recheck.json`, key `"287"`.
- **Disposition**: identical to case 41 above — retained in the ledger, does not block `manifest.json status: pass`.

## Resolved issues (not outstanding — recorded for completeness)

### Detailed-subset coverage-selection bug (implementation bug in `sweep.py`, not a data/numeric failure)

- **Symptom**: two successive implementations of `select_detailed_subset` left at least one `(attribute, value)` combination uncovered in the section-11 detailed-trajectory subset (`random_sparse` family missing in the first attempt; `momentum` optimizer missing in the second, "two-phase greedy" attempt), because `family` is the outermost loop in `enumerate_cases` and case-index-ordered greedy selection over-invests budget in the first family's optimizer/horizon/regime combinations before ever reaching later families.
- **Fix**: rewrote `select_detailed_subset` to, for each of the 14 `(attribute, value)` targets, pick whichever remaining candidate case realizing that value also closes the most other currently-open targets. Verified by `tests/oracle/test_sweep.py::test_select_detailed_subset_covers_every_attribute_value` and by the final run's `summary.json.detailed_subset_coverage_gaps == {}`.
- **Why recorded here rather than as a claim exception**: this was an infrastructure defect in the selection helper, caught and fixed before the reported sweep numbers were finalized; it does not affect any of the 288 cases' correctness checks and does not represent an unresolved anomaly in the oracle itself.

### Pareto active-set false-rejection bug (first review, R3) and SLSQP false-convergence bug (second review, R8)

- Both are code-quality bugs in the independent Pareto reference infrastructure, not oracle correctness anomalies; fully described in `runs/oracle-20260827-baseline/notes.md`'s "R1-R6" and "R7-R9" remediation sections respectively. Neither affects any of the 288 cases' `state`/`hypergradient`/`finite_difference`/`loss_change` checks.

## Summary

- Total cases: 288.
- Failed: 2 (both `check-failed` on the `loss_change` check only; both `unstable`/`momentum`; no `unstable-overflow` cases occurred in this sweep). `stable_failed_cases: 0`.
- Both failures are documented, reproducible (deterministic seeds, same `config_seed`), independently confirmed as pure floating-point cancellation (not a formula/implementation mismatch) by the second review's R7 Decimal(precision=100) reference, and retained in the ledger unmodified.
- No seed was removed, no tolerance was changed, and no generation range was retuned after observing these failures.
- Because both failures are unstable-regime and explained (not unexplained/systematic), and because the independent Pareto/QP cross-check (R8) passes on all 288 cases, `manifest.json status: pass`.
