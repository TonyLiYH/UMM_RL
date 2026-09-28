# Notation and model-independent problem

## Definition: shared/private objectives

For tasks \(i=1,\ldots,m\), let

\[
L_i(P_i\theta,\phi_i)
\]

be the task loss, where:

- \(\theta\in\mathbb R^p\) is the global parameter vector;
- \(P_i\theta\) selects the shared coordinates used by task \(i\);
- \(\phi_i\in\mathbb R^{q_i}\) is task-private;
- \(s_i=(\phi_i,\omega_i)\) additionally includes native optimizer state.

A global shared displacement is \(d\). A task-private displacement is \(u_i\).
Conflict requires negotiation only on coordinates selected by more than one
task.

## Definition: joint local improvement

A tuple \((d,u_1,\ldots,u_m)\) is a strict local joint-improvement step when

\[
L_i(P_i(\theta+d),\phi_i+u_i)
<
L_i(P_i\theta,\phi_i)
\quad\text{for every }i.
\]

At first order, with

\[
g_i=P_i^\top\nabla_{x_i}L_i,\qquad
h_i=\nabla_{\phi_i}L_i,
\]

the condition is

\[
g_i^\top d+h_i^\top u_i<0
\quad\text{for every }i.
\]

The private variables do not conflict with one another because \(u_i\) affects
only task \(i\). The shared displacement \(d\) is the coupled decision.

## Definition: finite private response

Let

\[
s_i^K(\theta)=A_i^K(P_i\theta;s_i^0,\zeta_i)
\]

denote exactly \(K\) native private updates at fixed shared parameters. The
finite-response value is

\[
F_i^K(\theta)
=
\mathbb E\,
L_i\!\left(P_i\theta,\pi_\phi s_i^K(\theta);\xi_i^{meta}\right).
\]

The expectation and randomness coupling must be declared by each experiment.

## Research questions

1. Does a shared update selected from raw gradients remain beneficial after
   equal-budget private adaptation?
2. Does private adaptation change the shared direction that should have been
   selected?
3. When does alternating optimization stagnate because of true Pareto conflict,
   and when is the stagnation caused by an inadequate gradient object?
4. Can a response-aware method improve total GPU time or gradient evaluations,
   rather than only reducing the number of outer iterations?

