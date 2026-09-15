# Alternating shared/private update protocols

## Definition: shared-then-private (SP)

At state \((\theta_t,s_{1,t},\ldots,s_{m,t})\), compute raw shared gradients

\[
g_{i,t}^{raw}
=
P_i^\top\nabla_{x_i}L_i(P_i\theta_t,\phi_{i,t}).
\]

A negotiator \(\mathcal N\), such as SUM, normalized SUM, MGDA, PCGrad, or
CAGrad, produces one shared direction:

\[
d_t=\mathcal N(g_{1,t}^{raw},\ldots,g_{m,t}^{raw}).
\]

Apply the shared update,

\[
\theta_{t+1}=\theta_t+\eta d_t,
\]

freeze \(\theta_{t+1}\), and run each private response:

\[
s_{i,t+1}=A_i^K(P_i\theta_{t+1};s_{i,t}).
\]

This is a Gauss--Seidel-style block update. It jointly trains all parameter
blocks over one window, but it does not jointly choose \(d_t\) and the private
responses: \(d_t\) is fixed before the responses are observed.

## Definition: compute-matched private-only control

To attribute improvement to the shared displacement rather than the extra
private steps, run the control branch from the same snapshot:

\[
s_{i,t}^{0,K}=A_i^K(P_i\theta_t;s_{i,t}),
\]

and the treatment branch:

\[
s_{i,t}^{d,K}=A_i^K(P_i(\theta_t+\eta d_t);s_{i,t}).
\]

Define

\[
\Delta_i^{controlled}(d_t)
=
L_i(P_i(\theta_t+\eta d_t),\pi_\phi s_{i,t}^{d,K})
-
L_i(P_i\theta_t,\pi_\phi s_{i,t}^{0,K}).
\]

This is the primary attribution quantity. Both branches consume the same
private adaptation budget.

## Definition: private-then-shared (PS/commit)

First form a virtual private response:

\[
\bar s_{i,t}=A_i^K(P_i\theta_t;s_{i,t}),
\]

then compute the post-adaptation stop-gradient:

\[
g_{i,t}^{commit}
=
P_i^\top
\partial_{x_i}L_i(x_i,\pi_\phi\bar s_{i,t})
\big|_{x_i=P_i\theta_t}.
\]

Negotiate and update the shared state from
\(\{g_{i,t}^{commit}\}\). The virtual private transition is restored unless the
protocol explicitly authorizes persistence.

## Definition: simultaneous baseline

Shared and private gradients are computed at the old state, and all blocks are
then updated. It differs from SP because the private update does not see
\(\theta_{t+1}\).

## Empirical question

The ordering ladder is:

\[
\text{simultaneous}
\rightarrow
\text{SP}
\rightarrow
\text{PS/commit}
\rightarrow
\text{rerun or reduced-space response}.
\]

Complex response modeling is justified only if it improves over SP after
matching private steps, data, gradient evaluations, and wall-clock budget.

