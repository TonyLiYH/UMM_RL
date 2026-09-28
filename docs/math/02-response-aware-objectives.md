# Response-aware objectives

## Raw shared gradient

\[
g_i^{raw}
=
P_i^\top\nabla_{x_i}L_i(P_i\theta,\phi_i).
\]

It answers the partial-derivative question with the current private state held
fixed.

## Commit-response gradient

After computing

\[
\bar s_i=A_i^K(P_i\theta;s_i),
\]

hold \(\bar s_i\) fixed and differentiate a candidate shared input:

\[
g_i^{commit}
=
P_i^\top
\partial_{x_i}L_i(x_i,\pi_\phi\bar s_i)
\big|_{x_i=P_i\theta}.
\]

Commit changes the evaluation point but does not differentiate the optimizer
trajectory.

## Rerun-response gradient

The full finite-response value is

\[
F_i^K(\theta)
=
L_i\!\left(
P_i\theta,
\pi_\phi A_i^K(P_i\theta;s_i)
\right).
\]

When differentiable,

\[
g_i^{rerun}
=
\nabla_\theta F_i^K(\theta)
=
g_i^{commit}
+
\left(\frac{\partial s_i^K}{\partial\theta}\right)^\top
\nabla_{s_i}L_i.
\]

The second term is the optimizer-trajectory response. Full AdamW unrolling can
be numerically fragile; a failed automatic-differentiation implementation is
not evidence that the mathematical response map is intrinsically
nondifferentiable.

## Definition: actual finite change

For a shared direction \(d\) and step \(\eta\),

\[
\Delta F_i^K
=
F_i^K(\theta+\eta d)-F_i^K(\theta).
\]

The first-order prediction is

\[
\widehat{\Delta F_i^K}
=
\eta(g_i^{rerun})^\top d.
\]

Raw and commit gradients can also be used as predictors, but they correspond
to different counterfactuals. Measuring \(\Delta F_i^K\) validates the gradient
object; it does not replace gradient-based parameter updates.

