# Proof: bounded first-order private compensation

## Proposition

Let task \(i\)'s first-order change be

\[
q_i(d,u_i)=g_i^\top d+h_i^\top u_i,
\]

with private budget \(\|u_i\|_2\le r_i\). Then

\[
\min_{\|u_i\|_2\le r_i}q_i(d,u_i)
=
g_i^\top d-r_i\|h_i\|_2.
\]

Consequently, private compensation makes \(d\) strictly improving for task
\(i\) if and only if

\[
g_i^\top d<r_i\|h_i\|_2.
\]

## Proof

By Cauchy--Schwarz,

\[
h_i^\top u_i
\ge
-\|h_i\|_2\|u_i\|_2
\ge
-r_i\|h_i\|_2.
\]

If \(h_i\ne0\), equality is attained by

\[
u_i^*=-r_i\frac{h_i}{\|h_i\|_2}.
\]

If \(h_i=0\), every feasible \(u_i\) attains zero private first-order change,
which agrees with the formula. Adding the constant \(g_i^\top d\) proves the
minimum. Strict improvement is equivalent to the stated inequality.

For all tasks, define

\[
\mathcal D_0=\{d:g_i^\top d<0\ \forall i\},
\]

\[
\mathcal D_r=\{d:g_i^\top d<r_i\|h_i\|\ \forall i\}.
\]

Since \(r_i\|h_i\|\ge0\), \(\mathcal D_0\subseteq\mathcal D_r\). The inclusion
is strict whenever a direction satisfies all relaxed inequalities and at least
one corresponding raw directional derivative is nonnegative.

## Limitation

This result permits the best displacement inside a norm ball. It does not claim
that \(K\) steps of SGD, AdamW, or another native optimizer attain that
displacement.

