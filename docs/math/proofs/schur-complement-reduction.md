# Proof: local reduced-space compensation

## Proposition

For

\[
\begin{aligned}
Q(d,u)
={}&g^\top d+h^\top u
+\frac12d^\top A d+d^\top B u
+\frac12u^\top C u,
\end{aligned}
\]

where \(C\succ0\), the unique conditional minimizer is

\[
u^*(d)=-C^{-1}(h+B^\top d).
\]

The reduced objective is

\[
\widetilde Q(d)
=
c
+
\left(g-BC^{-1}h\right)^\top d
+
\frac12d^\top
\left(A-BC^{-1}B^\top\right)d,
\]

where \(c=-\tfrac12h^\top C^{-1}h\).

## Proof

Complete the square:

\[
\begin{aligned}
Q(d,u)
={}&g^\top d+\frac12d^\top A d\\
&+\frac12
\left(u+C^{-1}(h+B^\top d)\right)^\top
C
\left(u+C^{-1}(h+B^\top d)\right)\\
&-\frac12(h+B^\top d)^\top C^{-1}(h+B^\top d).
\end{aligned}
\]

Positive definiteness of \(C\) makes the square nonnegative and uniquely
minimized at \(u^*(d)\). Expanding the last term gives

\[
-\frac12h^\top C^{-1}h
-d^\top BC^{-1}h
-\frac12d^\top BC^{-1}B^\top d.
\]

Collecting constant, linear, and quadratic terms proves the result.

Moreover,

\[
A-\left(A-BC^{-1}B^\top\right)
=
BC^{-1}B^\top\succeq0,
\]

so the reduced curvature is no larger than the fixed-private curvature in
Loewner order.

## Interpretation boundary

This is exact for the declared quadratic model. Neural-network use requires a
named curvature approximation, damping rule, trust region, and measured-loss
acceptance test.

