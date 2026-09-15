# Proof: common-descent convex-hull criterion

## Proposition

For vectors \(a_1,\ldots,a_m\in\mathbb R^p\), there exists \(d\) satisfying

\[
a_i^\top d<0\quad\text{for every }i
\]

if and only if

\[
0\notin\operatorname{conv}\{a_1,\ldots,a_m\}.
\]

## Proof

If \(0\) belongs to the convex hull, there are
\(\lambda_i\ge0\), \(\sum_i\lambda_i=1\), such that

\[
\sum_i\lambda_i a_i=0.
\]

If a strict common descent direction existed, then

\[
0
=
\left(\sum_i\lambda_i a_i\right)^\top d
=
\sum_i\lambda_i a_i^\top d
<0,
\]

a contradiction.

Conversely, the convex hull of finitely many vectors is compact and convex. If
it does not contain the origin, the strict separating-hyperplane theorem gives
a vector \(v\) and constant \(c>0\) such that

\[
v^\top a\ge c
\quad
\text{for every }a\in\operatorname{conv}\{a_i\}.
\]

In particular \(v^\top a_i\ge c\) for every task. Taking \(d=-v\) gives

\[
a_i^\top d\le-c<0.
\]

## Application

Use \(a_i=g_i^{raw}\) for raw stationarity and
\(a_i=g_i^{rerun}\) for finite-response stationarity. A change in convex-hull
membership is the model-independent definition of response-induced removal or
creation of a strict common direction.

