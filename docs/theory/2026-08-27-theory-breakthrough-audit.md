# Theory breakthrough audit — 2026-08-27

## 1. Executive decision

CompPareto still has a credible route to a mathematically distinctive contribution, but the contribution should not be framed as a generic multi-objective bilevel formulation, a generic stochastic MGDA convergence result, or a generic robust multi-objective method. Those areas are already occupied by mature and rapidly advancing work.

The strongest target is:

> **A finite-horizon, optimizer-state-aware, graph-localized posterior certificate for simultaneous Pareto descent, coupled to optimal allocation of samples and differentiation compute.**

The intended novelty is the intersection of four structures:

1. the operational objective is a prescribed finite task-native response, including momentum or Adam state;
2. derivative approximation error is bounded from observed trajectory residuals rather than unknown asymptotic constants;
3. uncertainty is localized to the task-block overlap graph;
4. the certificate determines acceptance, step size, response horizon, rank, CG iterations, and sample allocation.

No single ingredient above is new in isolation. Their combination is the candidate contribution.

## 2. Corrections required before a new theorem

### 2.1 Stop-gradient and finite unrolling are different protocols

Let \(s_i=(\phi_i,\omega_i)\) be the complete private optimizer state and let

\[
s_{i,K}=A_i^K(x_i;s_{i,0},\zeta_{0:K-1}).
\]

The rerun-response value

\[
F_i^{K,\mathrm{rerun}}(x)
=
\mathbb E L_i(x,\pi_\phi A_i^K(x;s_{i,0}))
\]

has a finite-unroll derivative. The commit-response counterfactual

\[
F_i^{K,\mathrm{commit}}(x';x)
=
\mathbb E L_i(x',\pi_\phi A_i^K(x;s_{i,0}))
\]

has a stop-gradient derivative with respect to \(x'\) at \(x'=x\). These are different operational questions, not merely estimators of different accuracy. Every theorem and experiment must name the protocol it certifies.

### 2.2 Finite \(K\) is not an inner error when \(F_i^K\) is the objective

If the selected target is the finite-response value \(F_i^K\), an exact differentiation of the prescribed \(K\) steps has no inner-solve error. The relevant deterministic error is

\[
e_i^{\mathrm{diff}}=\widehat h_i^K-h_i^K.
\]

An inner or horizon error is meaningful only relative to a separately defined reference:

\[
e_i^{\mathrm{horizon}}
=h_i^K-h_i^{K_{\mathrm{ref}}}
\quad\text{or}\quad
h_i^K-h_i^*.
\]

The latter requires assumptions that make a fixed point or best response meaningful.

### 2.3 Schur compensation and convexity require different assumptions

For the quadratic model, \(C_i\succ0\) alone gives a unique private minimizer and

\[
q_i(d,u_i)-\widetilde q_i(d)
=
\frac12\|u_i-u_i^*(d)\|_{C_i}^2.
\]

Positive semidefiniteness of the full damped block Hessian is only needed to obtain \(S_i\succeq0\), hence convexity in the shared step. An indefinite full Hessian does not by itself destroy private elimination.

### 2.4 The robust bound must use paired norms and a joint event

If

\[
\|\widehat h_i-h_i\|_{M^{-1}}\le\epsilon_i,
\]

then

\[
|(\widehat h_i-h_i)^\top d|
\le\epsilon_i\|d\|_M.
\]

The confidence event must hold for all tasks simultaneously and remain valid for a direction \(d\) selected from estimated data. This requires an independent certification batch, cross-fitting, or a uniform confidence set.

## 3. Closest verified literature and novelty boundary

| Work | Verified contribution | Consequence for CompPareto |
|---|---|---|
| Ehrhardt & Roberts, *Analyzing Inexact Hypergradients for Bilevel Learning*, arXiv:2301.04764 | Computable posterior bounds from lower-solve and linear-system residuals under a strongly convex lower problem | A posteriori hypergradient error alone is not new; extend it to finite native optimizer trajectories and Pareto acceptance |
| Bogensperger et al., *An Adaptively Inexact Method for Bilevel Learning Using Primal-Dual Style Differentiation*, arXiv:2412.06436 | Uses posterior hypergradient error to adapt lower solves, differentiation tolerance, and upper line search | Adaptive computation from hypergradient error is occupied; the new object must be the worst robust Pareto margin over a task-block graph |
| Liu & Vicente, *The Stochastic Multi-Gradient Algorithm*, arXiv:1907.04472 | The stochastic MGDA direction can be biased even when each task gradient is unbiased | Do not transfer deterministic convex-hull geometry directly to minibatch directions |
| MoCo, ICLR 2023 | Gradient tracking corrects stochastic multi-objective direction bias | Tracking compensated hypergradients is a baseline, not a new claim |
| Chen, *Improved Convergence Rate for Stochastic Multi-Gradient Descent*, arXiv:2607.18174 | Uses Lipschitz continuity of the Pareto-stationarity measure to obtain a \(\widetilde O(T^{-1})\) squared-stationarity rate with growing batches | A generic stochastic MGDA convergence theorem is a weak and crowded target |
| Yang et al., *Distributionally Robust Multi-Objective Optimization*, arXiv:2605.05660 | Objective-wise distributional robustness, biased gradients, clipping, and nonconvex Pareto-stationarity sample complexity | A Wasserstein or robust-MGDA wrapper is not enough |
| Jiang & Huang, *Efficient Hessian-Free Methods for Multi-Objective Bilevel Optimization with Nonconvex Lower Level*, arXiv:2608.12704 | Nonconvex lower-level MOBL, Moreau-envelope reformulation, Hessian-free single-loop algorithms, stochastic convergence, and Chebyshev scalarization | “Nonconvex stochastic MOBL” is no longer a defensible headline |
| Lu, *Bilevel Optimization with Coupled Decision-Dependent Distributions*, ICML 2023 | Upper and lower distributions depend on decisions across bilevel levels | Decision-dependent bilevel distributions are prior art and should remain an E3 extension |
| He et al., *Decision-Dependent Stochastic Optimization: The Role of Distribution Dynamics*, arXiv:2503.07324 | Models decisions and evolving distributions as coupled nonlinear dynamics with expectation and high-probability guarantees | Dynamic rollout shift needs explicit distribution dynamics, not an unconstrained error label |
| Tran & Vicente, *Stochastic block coordinate and function alternation for multi-objective optimization and learning*, arXiv:2605.12432 | Alternates objectives and variable blocks with convergence guarantees | Block alternation is prior art; CompPareto must exploit task-block incidence in uncertainty or certification |
| Huang & Chen, *Regularity-Aware Stochastic MGDA*, arXiv:2607.15412 | Establishes \(1/2\)-Hölder continuity in general and Lipschitz continuity under regularity of the common-descent direction map | Avoid relying on stable MGDA weights near degenerate active sets; certify the achieved margin or stationarity measure instead |

The literature audit found no verified work that simultaneously covers finite task-native optimizer state, partial task-block overlap, computable trajectory error, stochastic simultaneous descent, and certificate-driven resource allocation. This is the available novelty window.

## 4. Recommended theorem stack

### Theorem A — finite-horizon optimizer-response posterior error

Write the complete task-private state transition as

\[
s_{k+1}=T_{i,k}(s_k,x;\zeta_k),
\qquad x=P_i\theta.
\]

Let

\[
Z_k=\frac{\partial s_k}{\partial x},
\qquad
Z_{k+1}=J_kZ_k+B_k,
\]

where \(J_k=D_sT_{i,k}\) and \(B_k=D_xT_{i,k}\). An approximate sensitivity \(\widehat Z_k\) has the computable tangent residual

\[
r_k=\widehat Z_{k+1}-J_k\widehat Z_k-B_k.
\]

With \(E_k=\widehat Z_k-Z_k\),

\[
E_K
=
\Phi_{K,0}E_0+
\sum_{j=0}^{K-1}\Phi_{K,j+1}r_j,
\qquad
\Phi_{K,j+1}=J_{K-1}\cdots J_{j+1}.
\]

Hence

\[
\|E_K\|
\le
\|\Phi_{K,0}\|\|E_0\|
+
\sum_{j=0}^{K-1}\|\Phi_{K,j+1}\|\|r_j\|.
\]

For terminal loss \(L_i(x,s_K)\), the corresponding finite-response hypergradient error is bounded by

\[
\|\widehat h_i^K-h_i^K\|
\le
\|\nabla_sL_i\|_*
\left(
\|\Phi_{K,0}\|\|E_0\|
+
\sum_{j=0}^{K-1}\|\Phi_{K,j+1}\|\|r_j\|
\right)
+
e_i^{\mathrm{readout}}.
\]

This result is finite-horizon and local. It does not require convexity, convergence of Adam, or existence of an inner optimum. The nontrivial research work is to obtain useful, inexpensive upper estimates of the propagation gains and non-smooth switching defects.

### Theorem B — graph-localized simultaneous descent

Let the metric be block diagonal:

\[
M=\operatorname{diag}(M_b),
\]

and let task \(i\) touch blocks \(\mathcal B_i\). Suppose a joint event of probability at least \(1-\delta\) gives

\[
\|e_{i,b}\|_{M_b^{-1}}\le\epsilon_{i,b}
\quad
\text{for all }(i,b)\text{ in the overlap graph}.
\]

For any data-dependent candidate direction covered by this event,

\[
h_i^\top d
\le
\widehat h_i^\top d
+
\sum_{b\in\mathcal B_i}\epsilon_{i,b}\|d_b\|_{M_b}.
\]

Define

\[
\underline\gamma_i(d)
=
-\widehat h_i^\top d
-
\sum_{b\in\mathcal B_i}\epsilon_{i,b}\|d_b\|_{M_b}.
\]

If \(\underline\gamma_i(d)>0\) for every task and \(F_i^K\) is locally \(L_i\)-smooth in the chosen norm, then

\[
0<\eta<
\min_i
\frac{2\underline\gamma_i(d)}
{L_i\|d\|_M^2}
\]

guarantees simultaneous descent on the joint event.

The graph-specific gain is that uncertainty only accumulates on active task-block edges. In sparse overlap regimes, the width can depend on task support size rather than total model dimension. This must be demonstrated with a valid covariance or martingale model; parameter overlap is not automatically a stochastic dependency graph.

### Theorem C — certificate-width-optimal resource allocation

For a fixed direction, suppose the sampling contribution to the squared width is

\[
V_i(n)
=
\sum_{b\in\mathcal B_i}\frac{a_{i,b}}{n_b},
\qquad
a_{i,b}=\sigma_{i,b}^2\|d_b\|_{M_b}^2.
\]

With per-query costs \(c_b\) and budget \(N\), solve

\[
\min_{n_b>0}\max_i V_i(n)
\quad\text{s.t.}\quad
\sum_b c_bn_b\le N.
\]

This is convex. If \(y^*\in\Delta_m\) are optimal dual task weights, the continuous optimum satisfies

\[
n_b^*
=
\frac{
N\sqrt{
\frac1{c_b}\sum_{i:b\in\mathcal B_i}y_i^*a_{i,b}
}}
{
\sum_{b'}
\sqrt{
c_{b'}\sum_{j:b'\in\mathcal B_j}y_j^*a_{j,b'}
}
}.
\]

The practical version should jointly allocate adaptation/meta samples, finite-response horizon \(K_i\), CG or adjoint iterations, low-rank approximation rank, and repeated microbatches for certification. The allocation objective is the widest normalized robust margin, not generic gradient variance.

### Optional Theorem D — exact component decomposition

If the overlap graph separates into components, the metric and trust sets are block separable, and private state does not cross components, fixed-\(\tau\) retained-gain feasibility decomposes exactly by connected component. With a single global trust radius, a scalar dual multiplier remains as the only global coupling.

This is useful for parallel depth and communication, but it is an auxiliary result rather than the main novelty.

## 5. What should not be the headline

| Route | Decision | Reason |
|---|---|---|
| Generic MOBL Pareto convergence | Drop as headline | Occupied by WC-MHGD and newer nonconvex Hessian-free work |
| Stochastic MGDA rate | Drop as headline | 2026 results significantly strengthen the general theory |
| Wasserstein robust MGDA | Baseline or E3 extension | DR-MOO already covers objective-wise distributional robustness |
| Global Adam contraction | Do not assume | Not credible for a general nonconvex neural private subsystem |
| Shadowing or modified ODE analysis | Explanatory appendix only | Weak connection to a computable one-step acceptance decision |
| Treewidth solver for MGDA | Stretch | Mathematically interesting, but task count may be too small for practical impact |
| Viability/safe-set language | Use only with an executable projection rule | Terminology without computable constants will look decorative |

## 6. Exact-oracle validation programme

### 6.1 Linear-Gaussian finite-response oracle

Required controls:

- task-block graph: disjoint, chain, star, random sparse, fully shared;
- private response: SGD, momentum, diagonal Adam-like state;
- coupling rank and private curvature;
- stochastic gradient covariance by task and block;
- exact finite-unroll hypergradient;
- exact trajectory Jacobian and propagation gain;
- optional bounded distribution perturbation.

Required outputs:

- exact versus approximate finite-response hypergradient;
- posterior-bound coverage and tightness;
- simultaneous certificate coverage;
- false-safe and false-reject rates;
- samples, CG iterations, and unroll steps needed for a target width;
- regret to oracle resource allocation;
- disappearance of graph advantage under full overlap.

### 6.2 Controlled scene-graph bridge

Use capability factors such as count, relation, attribute, text, and composition as objectives or slices. The key test is whether the task-block uncertainty profile measured on the controlled renderer predicts the D0 large-model certificate width and failure modes.

## 7. Decision gates

The new theory route should be stopped or demoted if any condition holds:

1. posterior bounds have poor coverage on the exact finite-response oracle;
2. bounds are so loose that almost no useful step is certified;
3. overlap-localized widths do not improve over global widths on sparse graphs;
4. the advantage persists under full overlap, indicating an accounting error;
5. adaptive allocation does not approach oracle allocation or beat uniform allocation;
6. constants cannot be measured or conservatively calibrated on D0;
7. the same certificate applied to identical-\(A_i^K\) MGDA/Chebyshev baselines eliminates the apparent method advantage.

## 8. Recommended project sequence

1. Complete the already authorized T110, T120, T130, and T210 work.
2. Freeze rerun/commit finite-response semantics.
3. Add an exact finite-horizon optimizer-state oracle, initially for SGD and momentum, then an Adam-like diagonal state.
4. Prove and test the trajectory-residual posterior bound.
5. Add graph-localized simultaneous confidence bounds.
6. Add the minimax sample/compute allocation program.
7. Integrate the certificate into T340 before any E1 method claim.
8. Keep decision-dependent distribution dynamics for E3 unless static or bounded-shift D0 results already pass.

## 9. Recommended paper-level positioning

The strongest defensible future statement is:

> We study shared updates after finite task-native private optimizer responses. We derive trajectory-based posterior errors for the resulting optimizer-state-aware hypergradients, localize those errors through the task-parameter overlap graph, and use the resulting simultaneous descent certificate to allocate samples and differentiation compute. The method is validated first against an exact post-adaptation oracle and then in unified multimodal post-training under compute-matched baselines.

This positioning is more specific and more defensible than claiming a new general theory of multi-objective bilevel optimization.
