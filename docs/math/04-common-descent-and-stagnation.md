# Common descent and stagnation

## Proven geometric criterion

For exact nonzero task gradients \(a_1,\ldots,a_m\), a strict common descent
direction exists if and only if

\[
0\notin\operatorname{conv}\{a_1,\ldots,a_m\}.
\]

The proof follows from strict separation and is recorded in
[`proofs/finite-response-common-descent.md`](proofs/finite-response-common-descent.md).

For two tasks, failure occurs when the two nonzero gradients are negative
scalar multiples, or when a task gradient is zero and strict first-order
decrease is required for that task.

## Raw and response-aware stationarity

Raw apparent stationarity:

\[
0\in\operatorname{conv}\{g_i^{raw}\}.
\]

Finite-response geometry:

\[
0\in\operatorname{conv}\{g_i^{rerun}\}.
\]

If the first condition holds but the second does not, raw negotiation has a
**false response-relative stagnation point**: it stops under the raw objective
even though the finite-response values admit a common descent direction.

If both contain the origin, strict first-order common descent remains
unavailable under the selected response horizon. Progress then requires a soft
trade-off, a larger private capacity, a different sharing structure, a
non-local step, or a time-window rather than per-step constraint.

## Bounded private compensation

With first-order private budget \(\|u_i\|\le r_i\),

\[
\min_{\|u_i\|\le r_i}
\left(g_i^\top d+h_i^\top u_i\right)
=
g_i^\top d-r_i\|h_i\|.
\]

Therefore a shared direction can be made jointly improving by bounded private
steps exactly when

\[
g_i^\top d<r_i\|h_i\|
\quad\text{for all }i.
\]

This enlarges the feasible set relative to requiring
\(g_i^\top d<0\), but it does not show that a fixed native optimizer realizes
the optimal private displacement.

## Operational classification

| Observation | Interpretation | Next action |
|---|---|---|
| raw stops, response succeeds | false raw stagnation | use response-aware signal |
| both stop, estimates precise | genuine local trade-off under current structure | soft Pareto or change sharing |
| certificate fails from variance | evidence insufficient | allocate samples/compute |
| direction works only after unmatched private steps | attribution failure | use private-only control |

