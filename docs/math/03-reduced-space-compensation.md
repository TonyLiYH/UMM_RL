# Reduced-space private compensation

## Why eliminate private variables?

Elimination does not discard private parameters and does not itself remove
task conflict. It expresses the task-private response as a function of a
candidate shared displacement, then uses the response-adjusted task model to
choose the shared displacement.

The sequence is:

\[
d\ \text{candidate}
\rightarrow
u_i^*(d)\ \text{conditional private response}
\rightarrow
\widetilde Q_i(d)=Q_i(d,u_i^*(d)).
\]

After choosing \(d\), the recovered \(u_i^*(d)\) is still applied.

## Local quadratic model

At the current state, let

\[
\begin{aligned}
Q_i(d,u_i)
={}&g_i^\top d+h_i^\top u_i
+\frac12d^\top H_{\theta\theta}^{(i)}d\\
&+d^\top H_{\theta\phi}^{(i)}u_i
+\frac12u_i^\top H_{\phi\phi}^{(i)}u_i.
\end{aligned}
\]

If damping is used, augment the local model explicitly by

\[
\frac{\lambda_i}{2}\|u_i\|^2.
\]

Then the damped private block

\[
C_i=H_{\phi\phi}^{(i)}+\lambda_i I
\]

is positive definite and the conditional minimizer of the damped model is

\[
u_i^*(d)
=
-C_i^{-1}
\left(h_i+H_{\phi\theta}^{(i)}d\right).
\]

Substitution yields

\[
\widetilde Q_i(d)
=
\widetilde g_i^\top d
+\frac12d^\top S_i d
+c_i,
\]

where

\[
\widetilde g_i
=
g_i-H_{\theta\phi}^{(i)}C_i^{-1}h_i,
\]

\[
S_i
=
H_{\theta\theta}^{(i)}
-
H_{\theta\phi}^{(i)}C_i^{-1}H_{\phi\theta}^{(i)}.
\]

The complete derivation is in
[`proofs/schur-complement-reduction.md`](proofs/schur-complement-reduction.md).

## Interpretation

- \(C_i^{-1}h_i\) models unfinished private optimization.
- \(C_i^{-1}H_{\phi\theta}^{(i)}d\) is the private displacement induced by a
  candidate shared move.
- \(\widetilde g_i\) is a local reduced-space gradient.
- \(S_i\) is the Schur-complement effective shared curvature.

If \(h_i=0\), then \(\widetilde g_i=g_i\): exact local private stationarity
removes the first-order correction, while \(S_i\) can still change the
finite-step curvature.

## Boundary

The reduced model can rank candidate shared directions before executing private
adaptation. SP instead chooses \(d\) from raw gradients and adapts privately
afterward. Reduced-space modeling is worthwhile only if this look-ahead
improves compute-matched outcomes.

Exact elimination plus a shared gradient step is not automatically faster than
a matched exact-response alternating method: they have the same asymptotic
spectrum on the fixed strongly convex quadratic. One-step quadratic solution
requires the stronger Schur--Newton choice \(S^{-1}\), whose cost must be
counted.
