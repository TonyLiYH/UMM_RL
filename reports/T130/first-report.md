# T130 First Report: Indefinite Curvature and Trust-Region Rejection

**Task:** T130-indefinite-trust-region
**Branch:** agent/T130-indefinite-trust-region
**Date:** 2026-09-28
**Status:** running

## What this task is about

`QuadraticTask` requires the regularized private curvature block `h_phiphi + mu*I`
to be positive definite, but it never checks the shared block `h_xx` or the
Schur complement `h_xx - h_xphi @ (h_phiphi + mu*I)^{-1} @ h_xphi^T`, which is
the true effective curvature governing how a step in the shared coordinate changes
the objective once the private coordinate is allowed to respond.

It is therefore possible to construct tasks where both the private block and `h_xx`
are individually positive definite while the Schur complement has a negative
eigenvalue. This is the "indefinite effective curvature" scenario T130 is designed
to study.

The task asks for:
1. A counterexample suite with analytically known negative-curvature directions.
2. A measured-objective acceptance contract that does not rely solely on the
   compensated (Schur-based) predicted change.
3. Regression tests quantifying the false-accept rate of the naive vs. corrected rule.

## Approach

The construction uses the rank-k perturbation identity. With `h_xx = a*I` and
coupling columns `c_j` (mutually orthogonal), the Schur complement has closed-form
eigenvalues `a - ||c_j||^2 / p_j` (where `p_j` are diagonal entries of the
regularized private curvature). Choosing `||c_j||^2 / p_j > a` makes the Schur
complement indefinite while keeping both `h_xx` and the private block individually
PD. All eigenvalues were cross-checked numerically against `np.linalg.eigh`.

The acceptance contract separates two signals:
- **Compensated (predicted) change**: `task.compensated_change(step)` -- assumes
  the private coordinate has already jumped to its analytic optimum. This is what
  a naive rule trusts.
- **Frozen (measured) change**: `task.direct_change(step, zeros)` -- evaluates the
  exact quadratic with the private coordinate held at its current value (zero /
  not-yet-re-optimized). This is a freshly measured, conservative signal.

A step is accepted only when both signals agree the objective decreases. When they
disagree, the contract attempts a reduction (projecting out negative-curvature
eigenvectors of the Schur complement) before declaring outright rejection.

## CPU estimate

Suite construction and evaluation: < 1 second.
Full test suite (`python3 -m pytest`): ~2-3 seconds for T130-related tests.

## Files produced

- `src/comppareto/trust_region_guard.py` -- the acceptance contract
- `src/comppareto/trust_region_counterexamples.py` -- the counterexample suite
- `tests/test_trust_region_guard.py` -- 36 regression tests
- `configs/t1b/indefinite_trust_region.json` -- config
- `reports/T130/` -- this and companion reports
