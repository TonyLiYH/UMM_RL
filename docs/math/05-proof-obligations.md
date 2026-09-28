# Proof obligations and experiment links

## Established

1. Exact local Schur-complement elimination under positive private curvature.
2. The reduced curvature satisfies
   \(S_i\preceq H_{\theta\theta}^{(i)}\).
3. Strict common descent is equivalent to strict separation of the origin from
   the task-gradient convex hull.
4. Bounded first-order private compensation weakly enlarges the jointly
   feasible shared-direction set.

## Completed in the fixed quadratic scalarization

### O1. Alternating quadratic iteration matrices

Derive exact iteration matrices for:

- simultaneous shared/private gradient descent;
- shared-then-private Gauss--Seidel updates;
- private-then-shared commit-style updates;
- exact reduced-space/Schur updates.

The matrices, stability conditions, equal-spectrum result, strict-ordering
family, and counterexamples are now in
[`proofs/alternating-quadratic-convergence.md`](proofs/alternating-quadratic-convergence.md).

### O2. Strict speedup conditions

Find non-vacuous conditions on coupling, damping, step sizes, and approximation
error under which

\[
\rho(T_{\mathrm{reduced}})
<
\rho(T_{\mathrm{SP}}).
\]

The project retains cases where exact response ties SP and where stable
overrelaxation makes SP faster than a fixed-step reduced method. Persistent SP
and persistent PS are isospectral under matched fixed block maps.

### O3. Finite-step/native-optimizer gap

Bound

\[
\|u_i^K(d)-u_i^*(d)\|_{C_i}
\]

and translate the gap into response-value and descent-margin error. Full AdamW
is a later extension; SGD or a smooth damped optimizer is the first theorem
scope.

### O4. Stochastic attribution

Separate:

- sampling error;
- private-response approximation error;
- randomness-coupling error;
- actual shared-update effect.

The same data, RNG protocol, and private-step budget must be used in control
and treatment branches.

The exact gain identity, commit-response bounds, common-random-number variance,
and selection-bias boundary are in
[`proofs/controlled-gain-and-commit-error.md`](proofs/controlled-gain-and-commit-error.md).

## Remaining before a general speed claim

1. Analyze virtual commit and commit-then-SP for nonquadratic losses.
2. Extend from fixed scalarization to local Jacobians or switched systems for
   MGDA, PCGrad, and CAGrad.
3. Bound nonnormal finite-time amplification, not only spectral radius.
4. Compare total compute through \(c/(-\log\rho)\) or a finite-budget analogue.
5. Calibrate response sensitivity and stochastic error from data.

## Experiments that can proceed now

The first experiments do not require O1--O4 to be fully solved. They test:

1. whether SP improves over simultaneous raw training;
2. whether commit improves over SP;
3. whether a small-subspace rerun reference improves prediction;
4. whether any improvement survives a compute-matched private-only control;
5. whether raw zero/near-zero directions become nonzero under response-aware
   gradients.

No persistent joint-training claim is authorized until the corresponding task
Gate is opened.
