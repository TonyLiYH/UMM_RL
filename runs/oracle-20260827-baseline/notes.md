# run notes: oracle-20260827-baseline

Full T155 section-8 sweep grid: 6 graph families x 2 optimizers (SGD;
momentum with beta in {0.5, 0.9} cycled by seed index) x 4 horizons
(K in {1,3,5,10}) x 2 stability regimes x 3 seeds per cell = 288 cases.
Noise kind (gaussian / block_correlated) and momentum beta are both cycled
by `seed_idx % len(...)` within each cell rather than a further Cartesian
multiplier, per oracle-spec.md section 8's "noise model: both, every cell"
read together with section 10's 300-350 case grid estimate (a full Cartesian
product over noise kind would have pushed the count past that estimate).
Per-family (num_tasks, num_blocks, block_width) is fixed in
`configs/oracle/baseline.yaml` to realize the full m in {2,3,4,6,8} and
B in {4,8,16,32,64} ranges across the six families (one value per family,
covering both ends), per the resolved ranges already published in
`docs/theory/oracle-spec.md` section 8.

## Result

- 288 cases, 286 passed, 2 failed (both `check-failed`, none `unstable-overflow`).
- Elapsed 9.26s single core, well under the section-10 estimate of "low tens
  of seconds to a few minutes."
- manifest.json 2.7MB, matching section 10's "a few MB for the full sweep"
  estimate; no throttling needed.
- Detailed-subset (full trajectories/Jacobians per oracle-spec.md section 11):
  6 cases (case_index 33, 60, 96, 144, 210, 240), covering every family, both
  optimizers, all four horizons, and both stability regimes with zero gaps
  (`detailed_subset_coverage_gaps: {}` in summary.json).

## The 2 failures (case_index 41, 287 in the first sweep attempt before the
detailed-subset fix below; identical failures persist after the fix since
the case grid itself did not change)

Both are `momentum`, `stability_regime=unstable`, larger horizon (K=5 and
K=10 respectively). In both, `state` and `hypergradient` checks pass at
1e-15/1e-16 relative error (machine precision), and every finite-difference
step in the required h in [1e-6, 1e-2] envelope passes -- but the
`loss_change` check (the exact quadratic identity of section 6) misses the
1e-9 relative tolerance by roughly 2-13x (2.7e-9 and 1.3e-8 respectively).

Root cause, established by inspecting the full per-check breakdown (not
guessed): the deliberately-unstable regime gives these tasks realized
spectral radius ~1.64-1.67; compounded over K=5-10 steps this grows the
private-state trajectory magnitude substantially, which amplifies
floating-point cancellation specifically in the `exact_loss_change`
identity's `grad^T d + 1/2 d^T Q d` evaluation (a difference of large
terms), even though the underlying `grad` and `Q` are themselves exact to
machine precision. This is the same mechanism oracle-spec.md section 9
anticipated ("instability changes the magnitude of the trajectory, not the
exactness of the algebra") -- but that section's carve-out is specifically
for float64 *overflow*, not merely amplified-but-finite roundoff, so these
two cases are correctly left as gate failures, not reclassified as
`unstable-overflow`.

Per the task file's failure-and-retry rules ("numerical tolerance changes
require local review"), the 1e-9 loss-change tolerance was **not** relaxed
and no seed was dropped. Both cases remain in `failure_ledger.json` and
`manifest.json` (with `all_passed: false`) exactly as the frozen protocol
requires ("all failed or unstable seeds remain in the ledger"). This is
flagged for local-reviewer decision in `reports/T155/`.

## Detailed-subset selection bug found and fixed during this run

The first implementation of `select_detailed_subset` in
`src/comppareto/oracle/sweep.py` did a single greedy pass over case index
order, which -- because the case grid enumerates `family` as the outermost
loop -- exhausted the `detailed_subset_size: 10` budget covering
optimizer/horizon/regime combinations *within the first family alone*,
leaving 1-2 families entirely unrepresented in the detailed subset
(`random_sparse` was missing in the first run, then `momentum` in the
naive two-phase fix attempt). Fixed by switching to a targeted approach:
for each (attribute, value) target still uncovered, pick whichever
candidate case realizing that value also closes the most other open
targets simultaneously. This reduced the detailed subset from 10 forced
picks down to 6 with zero coverage gaps -- comfortably within
oracle-spec.md section 11's "5-10 representative cases" target.
