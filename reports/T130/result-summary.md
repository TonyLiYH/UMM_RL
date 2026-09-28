# T130 Result Summary

**Task:** T130-indefinite-trust-region
**Branch:** agent/T130-indefinite-trust-region
**Date:** 2026-09-28

## Counterexample suite: 6 cases

| Name | Category | Schur min eigenvalue | Predicted change | Measured change | Verdict |
|------|----------|---------------------|-----------------|-----------------|---------|
| pure_curvature_2d | unsafe | -24.0 (analytic) | -12.0 | +0.5 | reject |
| safe_control_2d | safe | +0.75 (analytic) | -0.406 | -0.375 | accept |
| two_negative_directions_dir_a | unsafe | -24.0 (analytic) | -12.0 | +0.5 | reject |
| two_negative_directions_dir_b | unsafe | -24.0 (analytic) | -12.0 | +0.5 | reject |
| reducible_mixed_step | reducible | -24.0 (analytic) | -127.38 | +0.62 | reduce |
| indefinite_but_genuinely_safe | safe | -24.0 (analytic) | -325.0 | -12.5 | accept |

## False-accept rates

- Naive rule (predicted-only): 4 false accepts out of 4 unsafe/reducible cases (100%)
- Corrected contract (evaluate_step): 0 false accepts out of 6 cases (0%)

## Reduction result

For `reducible_mixed_step`: the unsafe mixed step `3.2*v1 + 1.0*v2` is rejected, but
projecting out the `v1 = [0.6, 0.8]` component yields the reduced step `[-0.8, 0.6]`
(v2 alone), for which both predicted and measured changes equal exactly -4.5. The
reduction recovers a genuinely improving step.

## Regression test coverage

36 tests in `tests/test_trust_region_guard.py`, all passing. Covers:
- Task construction (private blocks PD), schur eigenvalue accuracy (< 1e-9 error)
- `trust_region_optimum` raises `CurvatureError` on indefinite schur (existing guard)
- `trust_region_optimum` does NOT raise on `h_xx` alone (silent convex projection risk)
- Naive false-accept behavior, evaluate_step verdict correctness, reduction values
- Zero false-accept rate of corrected contract, projection orthogonality/idempotence
- Input validation
