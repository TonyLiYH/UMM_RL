# Compensation-aware Pareto post-training

This document separates exact best-response theory from finite-step training estimators. Symbols are coordinate-checked: task-local shared gradients live in the coordinates selected by \(P_i\); a single multiplication by \(P_i^\top\) lifts them into global coordinates.

## 1. Partial-overlap parameter model

Let the global shared state be \(\theta\in\mathbb R^p\). Task \(i\) reads the task-local shared coordinates

\[
x_i=P_i\theta\in\mathbb R^{p_i},
\]

where \(P_i\in\{0,1\}^{p_i\times p}\) selects parameter blocks without duplicating a coordinate. Task-private parameters are \(\phi_i\), and native optimizer state—momenta, adaptive moments, replay state, or rollout state—is \(\omega_i\). The task loss is

\[
\ell_i(x_i,\phi_i;\xi_i).
\]

The bipartite overlap graph links task \(i\) to global block \(b\) exactly when \(P_i\) selects that block. A block touched by only one task needs no multi-objective negotiation.

## 2. Four distinct optimization objects

### 2.1 Raw joint-state objective

At training state \(t\), the raw local shared gradient is

\[
g^{\mathrm{raw}}_{i,t}=\nabla_{x_i}\mathbb E\ell_i(x_i,\phi_{i,t};\xi_i).
\]

It ignores any private response.

### 2.2 Finite native private response

Let \(A_i^K\) be exactly \(K\) task-native private updates with the shared state held fixed. Write the complete private optimizer state as \(s_{i,t}=(\phi_{i,t},\omega_{i,t})\):

\[
s_{i,t}^{K}
=A_i^K(x_i;s_{i,t},\zeta_{i,t}),
\qquad
\phi_{i,t}^{K}=\pi_\phi s_{i,t}^{K}.
\]

The optimizer state may include momenta, adaptive moments, step counters, schedules, gradient-scaler state, replay state, or other task-native memory.

There are two distinct operational values.

The **rerun-response** value reruns the complete private response at a candidate shared state:

\[
F_{i,t}^{K,\mathrm{rerun}}(\theta)=
\mathbb E_{\zeta_{i,t},\xi_{i,t}^{meta}}
\ell_i\!\left(
P_i\theta,
\pi_\phi A_i^K(P_i\theta;s_{i,t},\zeta_{i,t});
\xi_{i,t}^{meta}
\right).
\]

Its derivative, when the transition and randomness admit differentiation, is the finite-unroll hypergradient.

The **commit-response** counterfactual first computes the private response at the current state and then holds it fixed while evaluating a candidate shared state:

\[
F_{i,t}^{K,\mathrm{commit}}(\theta';\theta)=
\mathbb E_{\zeta_{i,t},\xi_{i,t}^{meta}}
\ell_i\!\left(
P_i\theta',
\pi_\phi A_i^K(P_i\theta;s_{i,t},\zeta_{i,t});
\xi_{i,t}^{meta}
\right).
\]

Its derivative with respect to \(\theta'\) at \(\theta'=\theta\) is the post-adaptation stop-gradient.

For backward-compatible shorthand, a generic time-indexed operational value is

\[
F_{i,t}^{K}(\theta;\phi_{i,t},\omega_{i,t})=
\mathbb E_{\zeta_{i,t},\xi_{i,t}^{meta}}
\ell_i(P_i\theta,\phi_{i,t}^{K};\xi_{i,t}^{meta}).
\]

but it is meaningful only after its rerun/commit semantics are fixed. It is conditional on the current private parameters, optimizer state, adaptation data, meta-batch, and whether the inner transition is virtual or persistent.

The corresponding derivatives must not be conflated:

- **post-adaptation stop-gradient:** the exact local derivative of the commit-response counterfactual;
- **finite-unroll hypergradient:** the exact local derivative of the rerun-response counterfactual when all relevant state transitions are differentiable and differentiation may be exchanged with expectation.

Thus these are not merely approximations of different accuracy. They answer different operational questions. Every formal run must state which protocol it implements.

### 2.3 Exact regularized best response

For local theory, define

\[
J_i(x_i,\phi_i)=
\mathbb E\ell_i(x_i,\phi_i)+
\frac{\mu_i}{2}\|\phi_i-\phi_i^0\|^2,
\qquad \mu_i>0,
\]

and

\[
\phi_i^*(x_i)=\arg\min_{\phi_i}J_i(x_i,\phi_i),
\qquad
F_i^*(\theta)=J_i(P_i\theta,\phi_i^*(P_i\theta)).
\]

The stationary condition is

\[
\nabla_{\phi_i}\mathbb E\ell_i(x_i,\phi_i^*)+
\mu_i(\phi_i^*-\phi_i^0)=0.
\]

The raw-loss private gradient need not be zero. Only the gradient of \(J_i\) with respect to \(\phi_i\) is zero.

### 2.4 Implicit estimator

When the exact response is locally unique, implicit differentiation estimates derivatives of \(F_i^*\) without unrolling the complete inner trajectory. It is an approximation to the exact regularized object, not automatically an approximation to an arbitrary finite native optimizer.

## 3. Correct local Schur-complement model

Expand \(J_i\) at \((x_i,\phi_i^*)\). Let \(\delta x_i=P_i d\), private displacement \(u_i\), local shared gradient \(a_i=\nabla_{x_i}J_i\), and Hessian blocks of the unregularized expected loss be \(H_{xx},H_{x\phi},H_{\phi x},H_{\phi\phi}\). Because \(\nabla_{\phi_i}J_i=0\), the quadratic change is

\[
q_i(d,u_i)=
a_i^\top P_i d+
\frac12(P_i d)^\top H_{xx}(P_i d)+
(P_i d)^\top H_{x\phi}u_i+
\frac12u_i^\top(H_{\phi\phi}+\mu_iI)u_i.
\]

Assume \(C_i=H_{\phi\phi}+\mu_iI\succ0\). Then

\[
u_i^*(d)=-C_i^{-1}H_{\phi x}P_i d,
\]

and

\[
\widetilde q_i(d)=
a_i^\top P_i d+
\frac12d^\top P_i^\top S_iP_i d,
\]

\[
S_i=H_{xx}-H_{x\phi}C_i^{-1}H_{\phi x}.
\]

The task-local value gradient is \(a_i\in\mathbb R^{p_i}\); its global lift is exactly \(\bar a_i=P_i^\top a_i\in\mathbb R^p\). No second lift is applied.

## 4. Proposition A: local compensation under a PSD surrogate

### Assumptions

1. The local model is the second-order expansion of \(J_i\), including the proximal term.
2. \(C_i\succ0\).
3. The cross blocks are transposes, as for a symmetric Hessian or explicitly symmetrized curvature surrogate.

### Statement

For every shared step \(d\), \(S_i\preceq H_{xx}\) and \(\widetilde q_i(d)\le q_i(d,0)\).

### Proof

Since \(C_i\succ0\),

\[
H_{x\phi}C_i^{-1}H_{\phi x}
=(C_i^{-1/2}H_{\phi x})^\top(C_i^{-1/2}H_{\phi x})
\]

is positive semidefinite. Hence \(S_i\preceq H_{xx}\). Direct minimization over \(u_i\) gives \(\widetilde q_i(d)=\min_{u_i}q_i(d,u_i)\le q_i(d,0)\). More precisely,

\[
q_i(d,u_i)-\widetilde q_i(d)
=
\frac12
\left(u_i-u_i^*(d)\right)^\top
C_i
\left(u_i-u_i^*(d)\right).
\]

Thus a finite private displacement \(u_i^K\) has the exact quadratic compensation gap

\[
q_i(d,u_i^K)-\widetilde q_i(d)
=
\frac12\left\|u_i^K-u_i^*(d)\right\|_{C_i}^2.
\]

If the complete damped block Hessian is additionally positive semidefinite, then its Schur complement satisfies \(S_i\succeq0\), which is the extra condition needed for convexity of the compensated shared quadratic.

### Failure cases

- Indefiniteness of the complete block Hessian does not destroy the unique minimizer over \(u_i\) when \(C_i\succ0\), but it can make \(S_i\) indefinite, make the shared trust-region model nonconvex, and invalidate a convex negotiation interpretation.
- A Gauss–Newton/Fisher approximation changes the modeled curvature and must be named explicitly.
- Strong damping can make the proposition true but the approximation uninformative.
- Finite native updates need not attain \(u_i^*(d)\); their realized effect is tested separately.

The algorithm therefore combines a PSD/damped model with measured-loss trust-region acceptance.

## 5. Proposition B: conditional loss-scale invariance

For a fixed task-independent metric \(M\succ0\), radius \(\rho\), and local model, define

\[
r_i=\max_{\|d\|_M\le\rho}-\widetilde q_i(d),
\qquad
R_i(d)=\frac{-\widetilde q_i(d)}{r_i+\epsilon_i}.
\]

Negotiate

\[
\max_{d,\tau}\quad \tau-\frac{\lambda}{2}\|d\|_M^2
\quad\text{s.t.}\quad
R_i(d)\ge\tau\ \forall i,
\quad\|d\|_M\le\rho.
\]

### Assumptions

1. Rescaling task \(i\) by \(c_i>0\) rescales the complete local model \(\widetilde q_i\) and \(\epsilon_i\) by \(c_i\).
2. \(M,\rho,\lambda\) and all other tasks remain fixed.
3. The optimizer solves the same constrained problem to the same tolerance.

### Statement and proof

Because the trust set is fixed,

\[
r_i'=\max_{\lVert d\rVert_M\le\rho}-c_i\widetilde q_i(d)=c_i r_i.
\]

Each ratio \(R_i(d)\) is therefore unchanged because numerator, attainable gain, and \(\epsilon_i\) all scale by \(c_i\). The feasible set and objective are unchanged, so the solution set is unchanged. The scaling of \(r_i\) is a consequence of its definition rather than an independent assumption.

### Failure cases

- If \(\widetilde q_i\) comes from a regularized objective, complete-model rescaling requires the proximal term and its coefficient to scale with the task objective. Rescaling only the raw loss while keeping an absolute proximal coefficient fixed does not satisfy the proposition.
- A shared Adam metric estimated from mixed task gradients generally changes when one task is rescaled.
- A fixed absolute \(\epsilon_i\) breaks invariance.
- Clipping, gradient normalization, finite precision, and stochastic reward calibration can break invariance.

Accordingly, the method is described as **conditionally loss-scale invariant**, not scale-free. The invariant variant uses a task-independent or frozen reference metric; mixed Adam metrics are an ablation without the claim.

## 6. Proposition C: deterministic common-descent certificate

At one fixed state, let \(h_i\in\mathbb R^{p_i}\) be the exact gradient of the selected differentiable value object in task-local coordinates, and let \(s_i>0\) be fixed task scales. Lift once:

\[
\bar h_i=P_i^\top h_i/s_i.
\]

Define

\[
h^*=\arg\min_{h\in\operatorname{conv}\{\bar h_1,\ldots,\bar h_K\}}
\|h\|_{M^{-1}}^2,
\qquad d=-M^{-1}h^*.
\]

### Assumptions

1. Every objective is differentiable at the same fixed state.
2. \(h_i\) are exact gradients of those objectives, not biased stochastic estimates.
3. \(M\succ0\) and \(s_i\) are fixed.

### Statement

If \(h^*\ne0\), then for every task

\[
\bar h_i^\top d\le-\|h^*\|_{M^{-1}}^2<0.
\]

If \(h^*=0\), no strict common direction is certified by these gradients; the state is Pareto stationary for the selected objectives and block space.

### Proof

The projection variational inequality gives \((\bar h_i-h^*)^\top M^{-1}h^*\ge0\). Rearranging and substituting \(d=-M^{-1}h^*\) gives the result.

### Stochastic boundary

For minibatch gradients, this is not a deterministic safety guarantee. The diagnostic reports a confidence interval for each directional derivative using repeated microbatches. A step is “statistically certified” only if every upper confidence bound is below zero; otherwise it is uncertified and handled by measured-loss acceptance. No stochastic convergence theorem is claimed in the initial project.

## 7. Proposed algorithm family

### Exact diagnostic

Use Hessian-vector products and conjugate gradients on selected blocks to estimate the regularized Schur complement. This validates the model on synthetic problems and selected real-model layers.

### Finite unroll

For each task and current state:

1. snapshot private parameters and optimizer state;
2. run \(K\in\{1,3,5\}\) private native steps with shared parameters fixed;
3. evaluate a disjoint meta-batch;
4. compute either stop-gradient or unrolled local gradients;
5. lift once using \(P_i^\top\) and negotiate the shared step;
6. accept or shrink using fresh measured losses;
7. follow a preregistered protocol for rolling back or committing the private transition.

### Scalable approximation

Use a diagonal or low-rank PSD curvature approximation and refresh negotiation weights every \(m\) steps. A frozen reference metric preserves the conditional scale claim; Adam-derived metrics are reported separately. Approximation error, additional data use, gradient evaluations, FLOPs, GPU-hours, and wall-clock overhead are all measured.

## 8. Evidence ledger

| Item | Status | Required evidence |
|---|---|---|
| Schur-complement derivation | Proven under stated local assumptions | Symbolic/numerical tests |
| PSD local compensation inequality | Proven above | Synthetic PSD and indefinite counterexamples |
| Conditional loss-scale invariance | Proven above | Rescaling and metric-failure tests |
| Deterministic common-descent certificate | Proven above | Convex-hull numerical tests |
| Stochastic certificate | Diagnostic proposal only | Calibration and coverage study |
| Novel convergence rate | Not claimed | New proof required before any claim |
| Better real-model prediction | Empirical hypothesis | D0 held-out study |
| Better joint capability | Empirical hypothesis | E1–E3 comparisons |

## 9. Next theory target: data-aware robust descent

The fixed-state propositions above do not include finite-response, curvature, sampling, or distribution-shift error. The next proposed value object is

\[
F_{i,t}^{K}(\theta;q_{i,t})=
\mathbb E_{z\sim q_{i,t}}
\ell_i\!\left(P_i\theta,A_i^K(P_i\theta;\phi_{i,t},\omega_{i,t}),z\right).
\]

For an estimated global task signal, the target analysis separates

\[
\widehat h_i-h_i=e_i^{\mathrm{inner}}+e_i^{\mathrm{curv}}
+e_i^{\mathrm{sample}}+e_i^{\mathrm{shift}}.
\]

The intended new result is a computable high-probability common-descent condition whose margin dominates these errors and the second-order remainder. It is an open theorem target, not an established result. The full research-state rationale and candidate experiments are recorded in `docs/handoffs/2026-08-27-research-state.md`.

If \(F_i^K\) itself is the selected operational objective, finite \(K\) is not an inner-solve error. In that case the deterministic approximation term means error in differentiating or representing the prescribed finite trajectory. A response-horizon or inner error is meaningful only when comparing \(F_i^K\) with a separately defined longer-horizon, fixed-point, or exact best-response object.

For a block-diagonal metric \(M=\operatorname{diag}(M_b)\), the preferred certificate uses task-block error radii \(\epsilon_{i,b}\) and the dual pairing

\[
\left|(\widehat h_i-h_i)^\top d\right|
\le
\sum_{b\in\mathcal B_i}
\epsilon_{i,b}\lVert d_b\rVert_{M_b}.
\]

Consequently, define

\[
\underline\gamma_i(d)
=
-\widehat h_i^\top d
-
\sum_{b\in\mathcal B_i}
\epsilon_{i,b}\lVert d_b\rVert_{M_b}.
\]

If \(\underline\gamma_i(d)>0\) and \(F_i^K\) is locally \(L_i\)-smooth in the corresponding norm, then any step satisfying

\[
0<\eta<
\frac{2\underline\gamma_i(d)}
{L_i\lVert d\rVert_M^2}
\]

decreases task \(i\) on the event defining the error radii. A simultaneous certificate must place all tasks in one joint event with probability at least \(1-\delta\), and it must remain valid when \(d\) is selected from the same estimated signals. Sample splitting, cross-fitting, or a uniform confidence set is therefore required; a pointwise interval for a data-dependent direction is not sufficient.
