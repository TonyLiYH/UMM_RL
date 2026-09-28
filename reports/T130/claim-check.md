# T130 Claim Check

**Task:** T130-indefinite-trust-region

## Research claim being investigated

The T130 task states: "Positive private curvature alone is insufficient to guarantee
that a proposed step is safe. A naive acceptance rule that checks only the
compensated (Schur-based) predicted change can accept steps that genuinely increase
the objective -- a false accept."

## Verdict: CONFIRMED, with qualifications

### What was verified

**Claim 1: Construction of indefinite-schur tasks with individually-PD blocks.**
Confirmed. All six counterexamples construct without error (private block PD by
construction), and `np.linalg.eigvalsh` confirms each Schur complement's most
negative eigenvalue matches the analytic value to within 1e-9 (e.g., -24.0 for
the 2D unit-coupling case).

**Claim 2: The naive (predicted-only) rule produces false accepts on indefinite cases.**
Confirmed. For every "unsafe" and "reducible" case: `compensated_change < 0`
(the naive rule would accept) but `direct_change(step, 0) > 0` (a fresh measurement
shows the objective actually increases with the private coordinate frozen). Measured
false-accept cases: 4 out of 6 cases in the suite.

**Claim 3: The corrected (dual-signal) contract produces zero false accepts.**
Confirmed. For every case where `evaluate_step` returns `verdict="accept"`, both
`predicted_change < 0` and `measured_change < 0`. No case where the corrected
contract accepted but the objective genuinely increased.

**Claim 4: The reduction strategy recovers a genuinely improving step.**
Confirmed for `reducible_mixed_step`. Projecting out the negative-curvature
eigenvector (`[0.6, 0.8]`) from the unsafe mixed step yields the reduced step
`[-0.8, 0.6]`, for which both predicted and measured changes equal exactly -4.5.

### Positive finding about existing code

`trust_region_optimum` already raises `CurvatureError` when fed `task.schur()`
for all indefinite-schur cases. The existing guard is correct, provided the caller
uses `schur()` rather than `h_xx`. Passing `h_xx` alone (which is PD) lets the
same step through silently -- the "silent convex projection" danger the task names.

### Qualifications

The "false accept" phenomenon demonstrated here is specifically about the gap
between the compensated-change model (which optimistically assumes instant private
re-optimization) and a fresh frozen-coordinate measurement. In the
`indefinite_but_genuinely_safe` case, the same indefinite Schur complement does
NOT produce a false accept because a strongly favorable gradient overrides the
curvature signal -- showing the contract is not spuriously conservative.
