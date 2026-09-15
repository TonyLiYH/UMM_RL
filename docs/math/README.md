# Shared/private optimization mathematics

This directory is the proof-oriented source for the shared/private optimization
questions studied by CompPareto. It complements
[`docs/theory/formulation.md`](../theory/formulation.md), which remains the
method-level overview.

## Status labels

- **Definition:** fixes notation or an operational protocol.
- **Proven:** assumptions and a complete proof are recorded.
- **Proof sketch:** argument is informative but not yet a formal project claim.
- **Conjecture:** plausible theorem target requiring proof.
- **Empirical question:** must be decided by a controlled experiment.

## Reading order

1. [`00-notation-and-problem.md`](00-notation-and-problem.md)
2. [`01-alternating-shared-private-update.md`](01-alternating-shared-private-update.md)
3. [`02-response-aware-objectives.md`](02-response-aware-objectives.md)
4. [`03-reduced-space-compensation.md`](03-reduced-space-compensation.md)
5. [`04-common-descent-and-stagnation.md`](04-common-descent-and-stagnation.md)
6. [`05-proof-obligations.md`](05-proof-obligations.md)

Detailed proofs live in [`proofs/`](proofs/).

## Current scientific boundary

The repository has established local quadratic compensation and deterministic
same-state common-descent results. It has not established that a response-aware
method is universally faster than alternating optimization, nor that private
adaptation always removes task conflict.

The immediate controlled question is:

> Does the simple shared-then-private alternating protocol already capture most
> of the useful private compensation, or does choosing the shared direction
> from post-adaptation/response-aware information provide additional
> compute-matched benefit?

