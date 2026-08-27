# T155 failure ledger

## Unresolved anomalies (require local-reviewer decision)

### 1. Case_index 41 — `disjoint` / `momentum` (beta cycled) / K=5 / `unstable`

- **Check that failed**: `loss_change` (exact quadratic identity vs. direct evaluation), tolerance 1e-9 relative.
- **Observed relative error**: 2.7e-9 (~2.7x the tolerance).
- **Checks on the same case that passed**: `state` (~1e-15 relative), `hypergradient` (~1e-16 relative), `finite_difference` (all required h in [1e-6,1e-2] steps within envelope).
- **Root cause**: the `stability_regime=unstable` draw for this case realizes a spectral radius of approximately 1.64-1.67 (see `runs/oracle-20260827-baseline/manifest.json`, case_index 41, `spectral_radius` field). Compounded over K=5 SGD-with-momentum steps, this grows the private-state trajectory magnitude enough that the `exact_loss_change` identity's `grad^T d + 1/2 d^T Q d` evaluation becomes a difference of two large, close-in-magnitude terms, amplifying floating-point cancellation. The underlying `grad` and `Q` values are themselves exact to machine precision (confirmed by the passing `state`/`hypergradient` checks on the identical case) — the loss of precision is specific to this one subtraction, not to the state or sensitivity computation.
- **Why not reclassified as `unstable-overflow`**: `docs/theory/oracle-spec.md` section 9 carves out deliberately-unstable cases only for actual float64 overflow (inf/nan). No overflow occurred here — all values are finite, well-formed floats; the failure is amplified-but-finite roundoff, which the spec does not exempt.
- **Why not tolerance-relaxed**: `tasks/T155-exact-finite-response-oracle.md`'s failure-and-retry rules require local review before any numerical tolerance change. No such review has occurred.
- **Disposition**: left as a `check-failed` entry in `failure_ledger.json` and `manifest.json` (`all_passed: false`), retained rather than dropped, per "all failed or unstable seeds remain in the ledger."
- **Recommendation for local review**: either (a) accept this as an expected floating-point limitation of the exact-identity check specifically under compounded instability at K>=5, and consider whether the `loss_change` tolerance should scale with a case's realized spectral radius / horizon instead of being fixed at 1e-9 uniformly, or (b) require a higher-precision (e.g. `numpy.float128` or a compensated-summation) evaluation path for `exact_loss_change` under the unstable regime before this can pass at 1e-9. Both are tolerance/implementation decisions reserved for local review, not made unilaterally here.

### 2. Case_index 287 — `random_sparse` / `momentum` (beta cycled) / K=10 / `unstable`

- **Check that failed**: `loss_change`, tolerance 1e-9 relative.
- **Observed relative error**: 1.3e-8 (~13x the tolerance).
- **Checks on the same case that passed**: `state` (~1e-15 relative), `hypergradient` (~1e-16 relative), `finite_difference` (all required steps within envelope).
- **Root cause**: same mechanism as case 41, at larger scale — spectral radius ~1.64-1.67 compounded over K=10 (twice the horizon of case 41), producing a proportionally larger trajectory-magnitude amplification and a correspondingly larger cancellation error (1.3e-8 vs. 2.7e-9 — roughly consistent with error growing with horizon under a fixed unstable spectral radius).
- **Disposition and recommendation**: identical to case 41 above.

## Resolved issues (not outstanding — recorded for completeness)

### Detailed-subset coverage-selection bug (implementation bug in `sweep.py`, not a data/numeric failure)

- **Symptom**: two successive implementations of `select_detailed_subset` left at least one `(attribute, value)` combination uncovered in the section-11 detailed-trajectory subset (`random_sparse` family missing in the first attempt; `momentum` optimizer missing in the second, "two-phase greedy" attempt), because `family` is the outermost loop in `enumerate_cases` and case-index-ordered greedy selection over-invests budget in the first family's optimizer/horizon/regime combinations before ever reaching later families.
- **Fix**: rewrote `select_detailed_subset` to, for each of the 14 `(attribute, value)` targets, pick whichever remaining candidate case realizing that value also closes the most other currently-open targets. Verified by `tests/oracle/test_sweep.py::test_select_detailed_subset_covers_every_attribute_value` and by the final run's `summary.json.detailed_subset_coverage_gaps == {}`.
- **Why recorded here rather than as a claim exception**: this was an infrastructure defect in the selection helper, caught and fixed before the reported sweep numbers were finalized; it does not affect any of the 288 cases' correctness checks and does not represent an unresolved anomaly in the oracle itself.

## Summary

- Total cases: 288.
- Failed: 2 (both `check-failed` on the `loss_change` check only; both `unstable`/`momentum`; no `unstable-overflow` cases occurred in this sweep).
- Both failures are documented, reproducible (deterministic seeds, same `config_seed`), and left unresolved pending local review, per the frozen protocol.
- No seed was removed, no tolerance was changed, and no generation range was retuned after observing these failures.
