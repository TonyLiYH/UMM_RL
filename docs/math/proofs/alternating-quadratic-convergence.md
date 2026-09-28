# Alternating quadratic convergence

**Status: Proven for a fixed strongly convex quadratic scalarization.**

This proof does not establish a convergence rate for MGDA, PCGrad, CAGrad, or
another state-dependent negotiator. Those methods generally produce nonlinear
or piecewise-smooth dynamics even when every task loss is quadratic.

## 1. Model and finite private response

Consider

\[
\mathcal L(\theta,\phi)
=
\frac12
\begin{bmatrix}\theta\\\phi\end{bmatrix}^{\!\top}
\begin{bmatrix}A&B\\B^\top&C\end{bmatrix}
\begin{bmatrix}\theta\\\phi\end{bmatrix}
-
\begin{bmatrix}a\\b\end{bmatrix}^{\!\top}
\begin{bmatrix}\theta\\\phi\end{bmatrix}
+c_0,
\]

with block Hessian \(H\succ0\). Let

\[
x=\theta-\theta_*,
\qquad
y=\phi-\phi_*.
\]

Then

\[
g=Ax+By,\qquad h=B^\top x+Cy,
\]

and

\[
S=A-BC^{-1}B^\top\succ0.
\]

At fixed \(x\), apply \(K\) private iterations

\[
y^{k+1}=y^k-Q(B^\top x+Cy^k).
\]

Define

\[
R=(I-QC)^K,
\qquad
E=\sum_{j=0}^{K-1}(I-QC)^jQ.
\]

Then

\[
EC=I-R,\qquad E=(I-R)C^{-1},
\]

and

\[
\boxed{y^K=Ry-EB^\top x}.
\]

Equivalently,

\[
y^K=-C^{-1}B^\top x
+R\left(y+C^{-1}B^\top x\right).
\]

Thus finite-response error relative to exact private minimization is controlled
exactly by \(R\). For private SGD, \(Q=\alpha I\) and
\(R=(I-\alpha C)^K\).

## 2. Exact iteration matrices

Let

\[
x^+=x-\Gamma(Ax+By),\qquad J=I-\Gamma A.
\]

Define

\[
X=
\begin{bmatrix}J&-\Gamma B\\0&I\end{bmatrix},
\qquad
Y=
\begin{bmatrix}I&0\\-EB^\top&R\end{bmatrix}.
\]

The simultaneous protocol is

\[
\boxed{
T_{\mathrm{sim}}
=
\begin{bmatrix}J&-\Gamma B\\-EB^\top&R\end{bmatrix}
=
I-\begin{bmatrix}\Gamma&0\\0&E\end{bmatrix}H.
}
\]

Shared then private is

\[
\boxed{
T_{\mathrm{SP}}=YX
=
\begin{bmatrix}
J&-\Gamma B\\
-EB^\top J&R+EB^\top\Gamma B
\end{bmatrix}.
}
\]

Persistent private then shared is

\[
\boxed{
T_{\mathrm{PS}}=XY
=
\begin{bmatrix}
J+\Gamma BEB^\top&-\Gamma BR\\
-EB^\top&R
\end{bmatrix}.
}
\]

Persistent PS is distinct from virtual commit, which restores the private
snapshot.

Exact private elimination gives

\[
\bar{\mathcal L}(x)=\mathcal L_*+\frac12x^\top Sx.
\]

With shared preconditioner \(\Gamma_R\),

\[
x^+=(I-\Gamma_RS)x,\qquad
y^+=-C^{-1}B^\top x^+.
\]

Therefore

\[
\boxed{
T_{\mathrm{red}}
=
\begin{bmatrix}M&0\\-C^{-1}B^\top M&0\end{bmatrix},
\quad M=I-\Gamma_RS,
}
\]

and

\[
\rho(T_{\mathrm{red}})=\rho(I-\Gamma_RS).
\]

The Schur--Newton choice \(\Gamma_R=S^{-1}\) solves the exact quadratic in one
iteration. This comes from solving the Schur system, not from compensation
alone.

## 3. Stability

### Proposition 1: simultaneous necessary and sufficient condition

Let \(D=\operatorname{diag}(\Gamma,E)\succ0\). Since

\[
D^{-1/2}T_{\mathrm{sim}}D^{1/2}
=
I-D^{1/2}HD^{1/2},
\]

\[
\boxed{
\rho(T_{\mathrm{sim}})<1
\iff
\lambda_{\max}(D^{1/2}HD^{1/2})<2.
}
\]

### Proposition 2: sufficient block condition

If

\[
2\Gamma^{-1}-A\succ0,\qquad
2E^{-1}-C\succ0,
\]

then

\[
\rho(T_{\mathrm{SP}})<1,\qquad
\rho(T_{\mathrm{PS}})<1.
\]

For a shared displacement \(\Delta x=-\Gamma g\),

\[
\Delta\mathcal L_{\mathrm{shared}}
=
-\frac12\Delta x^\top(2\Gamma^{-1}-A)\Delta x.
\]

The private block has the analogous formula. A nonstationary cycle therefore
strictly decreases the quadratic Lyapunov function.

These conditions are sufficient and require \(E\succ0\). For arbitrary
noncommuting \(Q,C\), positivity of \(E\) must be checked. Private SGD
\(Q=\alpha I\) is the clean first scope.

### Proposition 3: reduced condition

\[
\boxed{
\rho(T_{\mathrm{red}})<1
\iff
0<\lambda_j(\Gamma_R^{1/2}S\Gamma_R^{1/2})<2
\quad\forall j.
}
\]

Although \(S\preceq A\), this does not imply
\(\kappa(S)\leq\kappa(A)\). For example,

\[
A=I_2,\quad C=1,\quad
B=(\sqrt{1-\varepsilon},0)^\top
\]

gives \(S=\operatorname{diag}(\varepsilon,1)\).

## 4. SP and persistent PS are isospectral

### Theorem 1

\[
\boxed{
\operatorname{spec}(T_{\mathrm{SP}})
=
\operatorname{spec}(T_{\mathrm{PS}}),
\qquad
\rho(T_{\mathrm{SP}})=\rho(T_{\mathrm{PS}}).
}
\]

### Proof

\[
T_{\mathrm{SP}}=YX,\qquad T_{\mathrm{PS}}=XY.
\]

For same-size square matrices,

\[
\det(I-zXY)=\det(I-zYX).
\]

Hence their characteristic polynomials agree.

Reordering the same persistent block updates therefore does not improve the
asymptotic spectral factor in this model. Finite-time trajectories may still
differ through nonnormality, initialization, sampling, optimizer state, and
persistence semantics.

## 5. Exact-response equivalence

If

\[
R=0,\qquad E=C^{-1},\qquad\Gamma_R=\Gamma,
\]

then

\[
\boxed{
\rho(T_{\mathrm{SP}})
=
\rho(T_{\mathrm{PS}})
=
\rho(T_{\mathrm{red}})
=
\rho(I-\Gamma S).
}
\]

Exact elimination does not automatically accelerate a matched exact-response
alternating baseline.

## 6. A strict ordering family

Let

\[
H=\begin{bmatrix}I&B\\B^\top&I\end{bmatrix},
\qquad \gamma=\|B\|_2<1,
\]

and choose

\[
\Gamma=I,\quad E=(1-r)I,\quad R=rI,\quad 0<r<1.
\]

Then

\[
\rho_{\mathrm{red}}=\gamma^2,
\]

\[
\rho_{\mathrm{SP}}
=
\rho_{\mathrm{PS}}
=
r+(1-r)\gamma^2,
\]

\[
\rho_{\mathrm{sim}}
=
\frac{r+\sqrt{r^2+4(1-r)\gamma^2}}{2}.
\]

Hence

\[
\boxed{
\rho_{\mathrm{red}}
<
\rho_{\mathrm{SP}}
=
\rho_{\mathrm{PS}}
<
\rho_{\mathrm{sim}}
<1.
}
\]

This is a fixed-update comparison. If the alternating baseline can obtain
\(r=0\) at matched cost, it matches the reduced method asymptotically.

## 7. Counterexamples

### Exact response need not strictly accelerate

For

\[
A=C=1,\quad B=4/5,\quad\eta=\alpha=1,
\]

\[
\rho_{\mathrm{sim}}=4/5,
\qquad
\boxed{
\rho_{\mathrm{SP}}
=
\rho_{\mathrm{PS}}
=
\rho_{\mathrm{red}}
=
16/25.
}
\]

### Stable overrelaxed SP can beat a fixed-step reduced method

For the same \(A,C,B,\eta\), but \(\alpha=5/3\),

\[
\rho_{\mathrm{SP}}
=
\rho_{\mathrm{PS}}
=
2/5
<
16/25
=
\rho_{\mathrm{red}}.
\]

The reduced method can retune its shared step. The example shows that response
accuracy alone does not imply speed domination.

### Virtual commit can be unstable

For the overrelaxed example, virtual adaptation followed by a shared commit
step while restoring the private state has

\[
T_{\mathrm{virtual}}
=
\begin{bmatrix}16/15&8/15\\0&1\end{bmatrix},
\]

so \(\rho(T_{\mathrm{virtual}})=16/15>1\).

If the commit-selected shared step is followed by a fresh SP private update,

\[
T_{\mathrm{commit\text{-}SP}}
=
\begin{bmatrix}
16/15&8/15\\
-64/45&-62/45
\end{bmatrix},
\]

whose spectral radius is approximately \(1.01306>1\), while ordinary SP has
spectral radius \(0.4\).

## 8. Raw, commit, and rerun on the quadratic

Define

\[
r_\phi=y+C^{-1}B^\top x.
\]

Then

\[
g_{\mathrm{raw}}=Sx+Br_\phi,
\]

\[
g_{\mathrm{commit}}=Sx+BRr_\phi.
\]

If \(R^\top C=CR\), then

\[
g_{\mathrm{rerun}}=Sx+BR^2r_\phi.
\]

Let

\[
q_K=\|C^{1/2}RC^{-1/2}\|_2<1,
\qquad
M_B=\|BC^{-1/2}\|_2.
\]

Then

\[
\|g_{\mathrm{commit}}-Sx\|
\leq M_Bq_K\|r_\phi\|_C,
\]

\[
\|g_{\mathrm{rerun}}-Sx\|
\leq M_Bq_K^2\|r_\phi\|_C,
\]

\[
\|g_{\mathrm{commit}}-g_{\mathrm{rerun}}\|
\leq M_Bq_K(1+q_K)\|r_\phi\|_C.
\]

These are gradient-approximation results, not optimization-speed results.

## 9. Multi-task fixed-scalarization extension

For

\[
\mathcal L_w=\sum_iw_iL_i(P_i\theta,\phi_i),
\qquad w_i>0,
\]

\[
A=\sum_iw_iP_i^\top A_iP_i,
\]

\[
B=
\begin{bmatrix}
w_1P_1^\top B_1&\cdots&w_mP_m^\top B_m
\end{bmatrix},
\]

\[
C=\operatorname{diag}(w_1C_1,\ldots,w_mC_m).
\]

If every task block Hessian is positive definite and
\(\bigcap_i\ker P_i=\{0\}\), then the aggregate Hessian is positive definite,
and

\[
S
=
\sum_iw_iP_i^\top
\left(A_i-B_iC_i^{-1}B_i^\top\right)P_i.
\]

If task \(i\) applies

\[
\phi_i^+=\phi_i-\alpha_i\nabla_{\phi_i}L_i,
\]

the equivalent aggregate preconditioner is

\[
Q_i=\frac{\alpha_i}{w_i}I.
\]

These fixed-matrix results do not transfer directly to MGDA, PCGrad, or
CAGrad. Their state-dependent negotiation map must be linearized separately.

## 10. Cost-aware comparison

For linearly convergent methods with factor \(\rho\in(0,1)\) and per-window
cost \(c\), asymptotic time-to-accuracy scales with

\[
\frac{c}{-\log\rho}.
\]

A smaller spectral radius alone does not establish lower GPU time.

