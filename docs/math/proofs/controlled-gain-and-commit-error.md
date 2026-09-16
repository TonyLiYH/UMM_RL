# Controlled gain and commit-response error

## 1. Exact gain decomposition

Fix a private snapshot \(s\), response operator \(A^K\), shared state
\(\theta\), and displacement \(\delta\). Define

\[
s_0^K=A^K(\theta;s),
\qquad
s_\delta^K=A^K(\theta+\delta;s).
\]

Let

\[
\Delta^{\mathrm{total}}
=L(\theta+\delta,s_\delta^K)-L(\theta,s),
\]

\[
\Delta^{\mathrm{private}}
=L(\theta,s_0^K)-L(\theta,s),
\]

\[
\Delta^{\mathrm{controlled}}
=L(\theta+\delta,s_\delta^K)-L(\theta,s_0^K).
\]

Adding and subtracting \(L(\theta,s_0^K)\) gives

\[
\boxed{
\Delta^{\mathrm{total}}
=
\Delta^{\mathrm{private}}
+
\Delta^{\mathrm{controlled}}.
}
\]

Total improvement therefore does not prove that the shared step helped.

## 2. Rerun and commit values

Write \(A(\theta)=A^K(\theta;s)\). The rerun value is

\[
F(\theta')=L(\theta',A(\theta')),
\]

while the commit counterfactual at base point \(\theta\) is

\[
C(\theta';\theta)=L(\theta',A(\theta)).
\]

They agree at the base: \(F(\theta)=C(\theta;\theta)\).

## 3. Value and gradient error

Assume

\[
\|A(\theta+\delta)-A(\theta)\|
\leq\beta_A\|\delta\|
\]

and that \(L\) is \(G_s\)-Lipschitz in private state. Then

\[
\boxed{
|F(\theta+\delta)-C(\theta+\delta;\theta)|
\leq G_s\beta_A\|\delta\|.
}
\]

If \(A,L\) are differentiable,

\[
\boxed{
\nabla F(\theta)-g^{\mathrm{commit}}
=
J_A(\theta)^\top\nabla_sL(\theta,A(\theta)).
}
\]

Hence, if \(\|J_A\|\leq\beta_A\) and \(\|\nabla_sL\|\leq G_s\),

\[
\boxed{
\|\nabla F(\theta)-g^{\mathrm{commit}}\|
\leq\beta_AG_s.
}
\]

Commit is reliable when either the private trajectory is insensitive to the
shared state or the adapted private state is nearly stationary for the
evaluation loss.

## 4. Safe directional condition

Suppose

\[
\|\widehat g^{\mathrm{commit}}-\nabla F(\theta)\|
\leq\epsilon
\]

and \(F\) has \(L_F\)-Lipschitz gradient. For
\(\theta^+=\theta+\eta d\),

\[
F(\theta^+)-F(\theta)
\leq
\eta\widehat g^{\mathrm{commit}\top}d
+
\eta\epsilon\|d\|
+
\frac{L_F\eta^2}{2}\|d\|^2.
\]

A sufficient descent condition is

\[
\boxed{
-\widehat g^{\mathrm{commit}\top}d
>
\epsilon\|d\|
+
\frac{L_F\eta}{2}\|d\|^2.
}
\]

The response component of \(\epsilon\) can be bounded by \(\beta_AG_s\), or
calibrated by occasional directional reruns or paired finite differences.

## 5. Paired stochastic estimation

Let

\[
Z_1(\omega)
=L(\theta+\delta,A^K(\theta+\delta;s,\omega);\xi),
\]

\[
Z_0(\omega)
=L(\theta,A^K(\theta;s,\omega);\xi).
\]

With common random numbers,

\[
\widehat\Delta_{\mathrm{CRN}}
=
\frac1n\sum_{j=1}^n[Z_1(\omega_j)-Z_0(\omega_j)].
\]

It is unbiased for the coupled estimand, with variance

\[
\frac1n
\left(
\operatorname{Var}Z_1+\operatorname{Var}Z_0
-2\operatorname{Cov}(Z_1,Z_0)
\right).
\]

Independent randomness removes the covariance term and targets a difference
of marginal expectations. The coupled and independent estimands are not
operationally identical when randomness is part of the response protocol.

Same-batch adaptation/evaluation estimates a same-batch operational value, not
held-out post-adaptation performance. Direction selection also creates
optimism: selecting the best estimated direction makes its in-sample gain
upward biased. Use disjoint selection and evaluation batches or nested
resampling.

## 6. Counterexample: commit points uphill

Consider

\[
L(x,y)=\frac12x^2+\frac45xy+\frac12y^2
\]

at \(y=0\), followed by one private gradient step with \(\alpha=5/3\). Then

\[
\bar y=-\frac43x,
\qquad
g_{\mathrm{commit}}=-\frac1{15}x.
\]

But the true one-step rerun value is

\[
F^1(x;0)=\frac{29}{90}x^2,
\qquad
\nabla F^1(x;0)=\frac{29}{45}x.
\]

For \(x\neq0\), commit and rerun point in opposite directions. Stable private
optimization alone does not guarantee commit descent for the rerun value.

