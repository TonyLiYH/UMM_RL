# Breakthrough design note — finite-response robust Pareto control

**Date:** 2026-10-05  
**Status:** local research proposal; not yet an accepted task or empirical claim  
**Scope:** theory route for T160/T170 and the later D0 interface

## Executive conclusion

The repository has a sound and unusually well-controlled starting point: it
defines the *prescribed finite native response* as the object of interest and
already contains an exact SGD/momentum oracle with independent checks. The
current proposed headline — a posterior finite-response hypergradient bound
followed by graph-localized robust descent — is promising, but it should be
sharpened in two ways:

1. **Make the theorem selective rather than asymptotic.** The output should be
   a certified accept/reject/continue decision for a proposed shared step, not
   another generic convergence theorem for multi-objective bilevel learning.
2. **Treat uncertainty as a joint support-aware set, not as independent task
   error bars.** Parameter overlap is a support structure; it is not a
   probabilistic independence assumption.

The recommended contribution is therefore:

> **Cross-fitted, finite-horizon robust Pareto descent with trajectory-aware
> error radii and support-localized adaptive resource allocation.**

The mathematical object is a computable robust margin. It combines (i) exact
finite-horizon response semantics, (ii) residual-based differentiation error,
(iii) simultaneous statistical confidence, and (iv) an optional bounded
distribution-shift term. The margin chooses the direction, step size, and
whether more samples or more response/unroll computation are worth paying for.

This is materially more distinctive than “we apply MGDA to a bilevel model.”
It also fits the existing oracle and does not require a global contraction
theorem for Adam or a generic nonconvex MOBL convergence result.

## 1. Repository assessment

### 1.1 What is already strong

- The distinction between raw, commit, and rerun-response gradients is
  explicit. This prevents a common but serious semantic error: treating a
  stop-gradient commit signal as the derivative of the rerun value.
- The exact finite-response oracle is accepted and includes independent
  reverse-mode, finite-difference, high-precision, and Pareto-QP checks.
- The common-descent convex-hull criterion and local Schur reduction are
  correct under their stated assumptions.
- The compute-matched private-only control makes attribution falsifiable.

### 1.2 What should not be the headline

The following are now crowded or too weak as standalone contributions:

- a generic multi-objective bilevel formulation;
- a generic stochastic MGDA convergence theorem;
- “nonconvex MOBL with Hessian-free hypergradients” without a new operational
  object;
- a Wasserstein/robust-MOO wrapper with no finite-response or compute decision;
- a global convergence claim for an arbitrary native optimizer state.

Recent work has already strengthened these baselines: WC-MHGD gives finite-time
Pareto-stationarity results for multi-objective bilevel learning; recent SMG
work reports an approximately inverse-iteration squared-stationarity rate with
growing batches; and MOMEHA addresses nonconvex lower levels with a Moreau/
hessian-free construction. These results make a new asymptotic theorem a poor
novelty bet. See the literature map at the end of this note.

## 2. Proposed theorem: a cross-fitted finite-response certificate

Fix a task (i), a shared state (	heta), private optimizer state (s_i),
and a declared finite response map

\[
s_i^K(x;\zeta)=A_i^K(x;s_i,\zeta),
\qquad
F_i^K(x)=\mathbb E\,\ell_i\bigl(x,\pi_s A_i^K(x;s_i,\zeta);z\bigr).
\]

The theorem is about (F_i^K), not an unknown infinite-horizon best response.
Let (h_i^K=\nabla F_i^K(\theta)) denote the exact rerun-response gradient
when it exists.

Partition shared coordinates into parameter blocks (b=1,\ldots,B). Let

\[
S_i=\{b:\text{task }i\text{ reads or updates block }b\}
\]

be the support induced by the selector/route. On a selection split, compute an
estimated response signal \(\widehat h_i\). On an independent certification
split, construct a simultaneous event

\[
\mathcal E=\left\{
  \|e_{ib}\|_{M_{ib}^{-1}}\le r_{ib}
  \ \text{for every task }i\text{ and block }b\in S_i
\right\},
\qquad e_{ib}=\widehat h_{ib}-h_{ib}^K.
\]

The construction may use empirical-Bernstein, self-normalized, or bootstrap
radii, but the event must be joint. No independence between overlapping tasks
is assumed. A safe implementation can use a union bound over task-block pairs;
a sharper implementation can use a joint covariance ellipsoid.

For a direction (d), define the support-local uncertainty penalty

\[
U_i(d)=\sum_{b\in S_i} r_{ib}\|d_b\|_{M_{ib}}+\beta_i\|d\|_M,
\]

where \(\beta_i\) is a separately audited deterministic approximation radius
for tangent truncation, curvature approximation, clipping, or another named
algorithmic error. The first term is statistical and support-local; the second
must not be silently mixed into a confidence interval.

Assume (F_i^K) has (L_i)-Lipschitz gradient in the trust region. Then, on
the joint event \(\mathcal E\), for every direction selected without using
the certification split,

\[
\boxed{
F_i^K(\theta+\eta d)-F_i^K(\theta)
\le
\eta\left[
\widehat h_i^\top d+U_i(d)
\right]
 +\frac{L_i\eta^2}{2}\|d\|_2^2.
}
\]

Consequently, a sufficient simultaneous descent certificate is

\[
\boxed{
\widehat h_i^\top d+U_i(d)
 +\frac{L_i\eta}{2}\|d\|_2^2
\le -\gamma_i
\quad\forall i,
}
\]

for declared margins \(\gamma_i>0\). The certificate is valid for the
*selected* direction because selection and certification are cross-fitted or
otherwise covered by a uniform event.

### Proof sketch and exact boundary

The first-order error term is

\[
(h_i^K-\widehat h_i)^\top d
 =\sum_{b\in S_i} e_{ib}^\top d_b,
\]

and Cauchy–Schwarz in the paired norms gives

\[
e_{ib}^\top d_b
\le r_{ib}\|d_b\|_{M_{ib}}.
\]

Smoothness supplies the quadratic remainder. The proof does not require
stochastic independence, convexity, or exact private optimality. It does
require that the confidence event is simultaneous and that the target is the
declared finite response.

This is the theorem T160 should implement first. It is deliberately modest:
the unknown quantity is the finite-response gradient, while the approximation
radius is explicit and testable.

## 3. New algorithmic object: robust response MGDA

Use the certificate itself to select a direction. For a trust radius
\(\|d\|_2\le1\), solve

\[
\begin{aligned}
\max_{d,t}\quad &t\\
\text{s.t.}\quad
&t\le -\widehat h_i^\top d-U_i(d)-\frac{L_i\eta}{2}\|d\|_2^2,
\quad i=1,\ldots,m,\\
&\|d\|_2\le1.
\end{aligned}
\]

The right-hand sides are concave in (d), so the hypograph constraints give a
convex program for fixed radii and (L_i\). It returns:

- (t>0): a jointly certified step exists;
- (t\le0): do not claim conflict; either gather more evidence, increase
  response compute, shrink the step, or accept a soft Pareto trade-off.

This is not just MGDA with a penalty. The uncertainty penalty changes with the
direction and with the task-block support. A direction that looks good in a
high-variance block is correctly discounted even if its raw inner product is
large.

### Required baselines

Every future experiment must compare this program against:

1. raw MGDA on the same selection gradients;
2. normalized-sum/Chebyshev with tuned weights;
3. robust MGDA with global error radii;
4. support-local robust MGDA with uniform allocation;
5. oracle robust MGDA using exact finite-response gradients and known radii.

The full-overlap regime is a mandatory negative control: support localization
should lose its advantage when every task touches every block.

## 4. Resource allocation is part of the theory, not a tuning detail

For a fixed candidate direction, an empirical-Bernstein-style radius often has
the form

\[
r_{ib}(n_{ib})
\approx
\frac{a_{ib}}{\sqrt{n_{ib}}}+\frac{c_{ib}}{n_{ib}},
\]

where (n_{ib}) is the number of task-block certification observations.
Ignoring the lower-order (1/n) term first gives the convex minimax problem

\[
\min_{n_{ib}\ge n_{\min}}
\max_i\sum_{b\in S_i}
\frac{a_{ib}\|d_b\|_{M_{ib}}}{\sqrt{n_{ib}}}
\quad\text{s.t.}\quad
\sum_{i,b} c_{ib}n_{ib}\le C.
\]

This has a useful KKT structure. For a fixed active mixture of worst tasks with
dual weights \(\alpha_i\), define

\[
\bar a_{ib}=\alpha_i a_{ib}\|d_b\|_{M_{ib}}.
\]

Away from lower bounds,

\[
\boxed{
n_{ib}\propto
\left(\frac{\bar a_{ib}}{c_{ib}}\right)^{2/3}.
}
\]

Thus the optimal design allocates more certification queries to blocks that
are both (i) influential for the proposed direction and (ii) uncertain for the
currently worst task. This is stronger than temperature balancing and gives a
direct experimental prediction: adaptive allocation should reduce the maximum
certificate width at lower total query cost than uniform allocation on sparse
graphs, but not under full overlap.

The exact implementation should retain the (1/n) term, lower bounds, integer
rounding, and task-query versus edge/block-query accounting as separate oracle
models. The continuous KKT solution is a reference, not the final claim.

## 5. Trajectory residuals: what T160 should actually certify

For a finite response state recurrence

\[
s_{k+1}=T_k(s_k,x),
\qquad
J_{k+1}=\partial_xT_k+\partial_sT_kJ_k,
\]

an approximate tangent \(\widehat J_k\) has per-step residual

\[
R_k=\widehat J_{k+1}
 -\left(\partial_xT_k+\partial_sT_k\widehat J_k\right).
\]

Let \(G_{j:k}=\prod_{r=j}^{k-1}\|\partial_sT_r\|\). Then the propagated
tangent error admits the a posteriori bound

\[
\|J_K-\widehat J_K\|
\le
\sum_{j=0}^{K-1}G_{j+1:K}\|R_j\|,
\]

with the convention (G_{K:K}=1). Multiplying by the terminal state-gradient
and adding any direct-derivative residual gives a finite-response
hypergradient radius \(\beta_i\).

This is the right role for inexact-hypergradient literature: import the
residual/propgation-gain pattern, but specialize it to the *declared optimizer
trajectory* and retain the measured trajectory residual in every artifact.
Do not call finite (K) itself an error unless comparing against a separately
declared (K_{\rm ref}) or fixed point.

T160 should test three regimes separately:

1. exact recurrence (radius near numerical zero);
2. truncated tangent/CG/low-rank approximation (finite calibrated radius);
3. unstable or clipped/switching trajectory (unsupported or rejected, never
   silently certified).

## 6. Optional extension: decision-dependent distributions

The first theorem should use a frozen certification distribution. For later RL
or rollout settings, add a separate term rather than hiding distribution shift
inside gradient noise. One safe form is:

\[
\left|F_i(\theta+\eta d;q_{\theta+\eta d})
 -F_i(\theta+\eta d;q_\theta)\right|
\le \kappa_i\,D(q_{\theta+\eta d},q_\theta),
\]

and assume or estimate

\[
D(q_{\theta+\eta d},q_\theta)\le \chi_i\eta\|d\|.
\]

The resulting term \(\kappa_i\chi_i\eta\|d\|\) is a first-order performative
penalty. It can be included in (U_i(d)), but only if the divergence and its
calibration protocol are declared. This extension should remain outside the
first static-data theorem until D0/E1 demonstrate that it matters.

## 7. Falsification gates

The route should be demoted if any of these occurs:

1. coverage fails on the accepted T155 oracle;
2. the certificate is so conservative that it almost never accepts an oracle
   descent step;
3. support-local widths do not beat global widths on sparse overlap;
4. the graph advantage survives full overlap;
5. adaptive allocation does not beat or approach the continuous KKT reference;
6. the same gains appear for raw MGDA after matching all certification cost;
7. the claimed \(\beta_i\) is measured only on the selection batch and not on
   an independent validation/certification path;
8. a real-model D0 signal does not predict held-out controlled gains better
   than raw first- and second-order Taylor baselines.

## 8. Concrete next actions

These are research actions, not yet a change of authoritative task status:

1. Keep T110/T120/T130 as independent validation of the existing algebra.
2. Use this note as the design specification for opening T160 once its
   dependencies are accepted.
3. Implement the certificate first on T155's exact SGD/momentum trajectories;
   do not start with Adam-like state or real-model unrolling.
4. Add a paired selection/certification split to the oracle sweep.
5. Measure coverage, acceptance, width, false-safe rate, and allocation regret
   before any D0 method comparison.
6. Only after the exact oracle passes, open the graph-localized allocation route
   and then connect it to D0.

## 9. Literature map and novelty boundary

The following works are the closest current references checked on 2026-10-05:

- Ehrhardt & Roberts, *Analyzing Inexact Hypergradients for Bilevel Learning*,
  arXiv:2301.04764: a posteriori inexact-hypergradient analysis. Import the
  residual logic, but do not present it as new by itself.
- Bogensperger et al., *An Adaptively Inexact Method for Bilevel Learning Using
  Primal-Dual Style Differentiation*, arXiv:2412.06436: adaptive lower-solve and
  differentiation tolerances. Our distinction is finite native optimizer-state
  response plus a taskwise robust acceptance rule.
- *Multi-Objective Bilevel Learning*, arXiv:2511.07824: weighted-Chebyshev
  multi-hyper-gradient descent with finite-time Pareto-stationarity results.
  This is a direct baseline and a reason not to claim generic MOBL theory.
- Chen, *Improved Convergence Rates for Stochastic Multi-Gradient Descent*,
  arXiv:2607.18174: a recent stronger stochastic-MGDA rate. Use it as a
  convergence baseline, not as the novelty target.
- Jiang & Huang, *Efficient Hessian-Free Methods for Multi-Objective Bilevel
  Optimization with Nonconvex Lower Level*, arXiv:2608.12704: a recent
  nonconvex-lower-level MOBL route. This closes another obvious headline.
- Wood & Dall'Anese, *Online Saddle Point Tracking with Decision-Dependent
  Data*, PMLR 211 (2023), and related decision-dependent-distribution work:
  motivation and tools for the later rollout-shift extension.

The candidate novelty is the intersection, not any one citation:

\[
\text{finite native response}
 +\text{trajectory residual radius}
 +\text{joint support-local confidence}
 +\text{certificate-guided allocation}.
\]

The paper should claim this only after all four pieces are implemented,
independently checked, and shown to change compute-matched decisions.
