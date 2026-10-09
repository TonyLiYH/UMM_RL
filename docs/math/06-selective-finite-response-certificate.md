# Selective finite-response control with anytime-valid evidence

**Status:** Proof target / design specification (not yet proven or empirically
accepted)

This note upgrades the static robust certificate in
[`docs/theory/2026-10-05-breakthrough-design.md`](../theory/2026-10-05-breakthrough-design.md)
into a sequential decision object. The key change is that the method does not
only ask whether a direction is good under the current estimate; it decides
whether to **commit, reject, or buy more evidence** while preserving a joint
error guarantee under optional continuation.

## 1. Why a sequential layer is needed

The proposed finite-response bound is valid for a direction selected without
using the certification data. In an actual training loop, however, the system
may repeatedly ask for more microbatches, more private-response steps, or more
accurate tangent solves after seeing an inconclusive margin. Reusing a fixed
confidence interval after this adaptive stopping rule is not automatically
valid.

The theory target is therefore an **anytime-valid selective certificate**. It
should remain valid when the algorithm adaptively chooses the next query and
stops at a data-dependent time.

## 2. Decision-relevant uncertainty set

For task (i), let \(h_i^K\) be the exact rerun-response gradient and let
\(\widehat h_{i,n}\) be its estimate after \(n\) certification observations.
Use a joint, support-aware confidence set

\[
\mathcal C_n(\delta)=
\left\{h:
\|h_{i,S_i}-\widehat h_{i,n,S_i}\|_{\Sigma_{i,n}^{-1}}
\le r_{i,n}(\delta),\ \forall i\right\},
\]

where (S_i) is the task's active shared-block support. A blockwise union
bound is an allowed baseline, but the preferred object is a covariance-aware
ellipsoid because overlap creates correlated estimation errors even when the
parameter selectors are sparse.

The confidence sequence requirement is

\[
\Pr\left( (h_i^K)_i\in\mathcal C_n(\delta)\text{ for all }n\ge n_0\right)
\ge 1-\delta.
\]

This event must cover adaptive query choices and optional continuation. It may
be constructed from a self-normalized martingale bound, an e-process, or a
time-uniform empirical-Bernstein construction. A fixed-(n) interval is not a
confidence sequence.

## 3. Robust margin and inverse feasible region

For a proposed direction \(d\), step \(\eta>0\), and a per-unit-step
smoothness remainder \(R_i(d,\eta)\), define the lower certified descent margin

\[
\Gamma_{i,n}(d,\eta):=
-\widehat h_{i,n}^{\top}d
-r_{i,n}(\delta)\|d_{S_i}\|_{\Sigma_{i,n}}
-\beta_{i,n}\|d\|_M
-R_i(d,\eta).
\]

The direction's simultaneous robust margin is

\[
\Gamma_n(d,\eta)=\min_i\Gamma_{i,n}(d,\eta).
\]

For example, the descent lemma gives
\(R_i(d,\eta)=L_i\eta\|d\|_2^2/2\). On the confidence-sequence event,
\(\Gamma_n(d,\eta)>0\) implies the
finite-response loss decreases for every task after the declared response and
step. If (Gamma_n(d,eta)\le0), this is **not** evidence of Pareto conflict:
the correct action is to reject, shrink, or collect information.

This gives an inverse feasible region for the decision: the set of gradient
realizations under which the proposed ((d,eta)) is certified safe. The
research novelty is to make this region operational for finite native
optimizer responses and to use its width to control compute.

## 4. Selective controller

At each state, the controller executes the following policy:

1. Use a selection split to propose (d) and (eta).
2. Query a certification edge \((i,b)\), or refine the approximation to the
   fixed \(K\)-step response and its tangent.
3. Update the confidence sequence and recompute (Gamma_n(d,eta)).
4. **Commit** if (Gamma_n(d,eta)\ge\gamma).
5. **Reject or shrink** if a declared budget is exhausted and the margin is
   non-positive.
6. Otherwise continue with the query that maximizes predicted reduction in
   certificate width per unit cost.

The last rule is a value-of-information heuristic in the first implementation
and a theorem target later. A simple directional proxy is

\[
\operatorname{VOI}(q;d)
\approx
\frac{\Gamma_{n+1}(d,\eta)-\Gamma_n(d,\eta)}{\operatorname{cost}(q)},
\]

with the improvement estimated from the current variance model. The oracle
benchmark must compare this policy with uniform sampling and with the exact
finite-budget allocation solver.

## 5. Theorem target

Let (\tau) be the stopping time at which the controller commits, rejects, or
exhausts its budget. Prove:

\[
\Pr\left[
\text{every committed step satisfies }
F_i^K(\theta+\eta d)-F_i^K(\theta)\le -\eta\gamma
\text{ for all }i
\right]\ge 1-\delta.
\]

The response horizon \(K\) is fixed for each certified decision. Extending
the native response to \(K+1\) changes the target \(F_i^K\); doing so requires
a new simultaneous event that covers the candidate horizons or a fresh
certification run. The proof should separate three terms:

1. statistical uncertainty controlled by the joint confidence sequence;
2. deterministic tangent/curvature approximation controlled by residuals;
3. smoothness or distribution-shift remainder controlled by a declared trust
   region.

No independence between overlapping tasks is required. The theorem must not
claim that parameter disjointness implies stochastic independence.

## 6. Resource allocation consequence

For a candidate direction, let (q) index an admissible query: a task
microbatch, an edge/block probe, or a refinement of the fixed-horizon response
or tangent. The decision-aware allocation problem is

\[
\min_{n\in\mathbb N^Q}
\max_i\operatorname{width}_i(n;d)
\quad\text{s.t.}\quad
\sum_q c_qn_q\le C.
\]

The target is not variance reduction in every coordinate. It is reduction of
the widest task margin that controls commit/reject/continue. This makes the
allocation graph- and direction-dependent. The continuous (n^{-1/2}) KKT
solution in T170 is the first reference; the sequential policy must be judged
by certified-step recall, false-safe rate, total cost, and regret to that
oracle.

## 7. Boundary conditions and falsifiers

The route is demoted if:

- optional continuation breaks empirical coverage;
- the controller commits unsafe steps on the exact finite-response oracle;
- confidence sets are valid but so wide that no useful step is ever certified;
- the value-of-information policy does not beat uniform allocation under sparse
  overlap, or retains an advantage under full overlap;
- the method's gain disappears when the same sequential controller is wrapped
  around raw MGDA and Chebyshev baselines.

This note intentionally does not assert a theorem. It defines the additional
mathematical object that can make the project more than a robustified MGDA
implementation: a finite-response, support-aware, anytime-valid controller
that allocates evidence only when the decision is uncertain.
