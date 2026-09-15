# Proof obligations and experiment links

## Established

1. Exact local Schur-complement elimination under positive private curvature.
2. The reduced curvature satisfies
   \(S_i\preceq H_{\theta\theta}^{(i)}\).
3. Strict common descent is equivalent to strict separation of the origin from
   the task-gradient convex hull.
4. Bounded first-order private compensation weakly enlarges the jointly
   feasible shared-direction set.

## Must be completed before a convergence-speed claim

### O1. Alternating quadratic iteration matrices

Derive exact iteration matrices for:

- simultaneous shared/private gradient descent;
- shared-then-private Gauss--Seidel updates;
- private-then-shared commit-style updates;
- exact reduced-space/Schur updates.

Compare spectral radii under declared positive-definite quadratic families.

### O2. Strict speedup conditions

Find non-vacuous conditions on coupling, damping, step sizes, and approximation
error under which

\[
\rho(T_{\mathrm{reduced}})
<
\rho(T_{\mathrm{SP}}).
\]

The project must also retain counterexamples where SP is equal or better.

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

