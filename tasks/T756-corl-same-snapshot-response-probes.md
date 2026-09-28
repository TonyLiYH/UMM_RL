---
id: T756
title: CoRL same-snapshot response-opportunity probes
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T756-corl-same-snapshot-response-probes
depends_on: [T711, T755, T730, T740]
blocks: [T760, T770]
allowed_paths: ["tasks/T756-corl-same-snapshot-response-probes.md", "configs/corl/response-probes-v1/", "runs/corl-response-probes-v1/", "reports/T756/", "src/comppareto/response_probe/", "tests/response_probe/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-28
updated_at: 2026-09-28
---

# T756: CoRL same-snapshot response-opportunity probes

## Objective

Determine whether finite private adaptation changes useful shared-update
decisions beyond tuned raw directions and shared-then-private updates.

## Frozen counterfactual window

For every probe, freeze:

- trajectories, candidate order, behavior log probabilities, masks, rewards,
  advantages, and reward-evaluator state;
- parameters, optimizer moments, scheduler, and RNG;
- a disjoint audit batch.

Evaluate:

\[
L_{00},\quad L_{10},\quad L_{01},\quad L_{11},\quad L_c,
\]

for neither update, shared-only, private-only, shared-then-private, and
commit-endpoint-held-fixed states.

Report:

\[
\Delta^{private}=L_{01}-L_{00},
\quad
\Delta^{controlled}=L_{11}-L_{01},
\]

\[
I=L_{11}-L_{10}-L_{01}+L_{00},
\quad
R=L_{11}-L_c.
\]

## Candidate directions

At minimum compare tuned raw SUM, shared-then-private, raw direction
line-search with equal validation budget, commit, and a bounded rerun
reference. Use realized post-AdamW shared displacements.

## Advance gate

Advance only if response information improves held-out candidate decision
quality over the strongest cheap predictor by a preregistered practical margin
and produces favorable controlled shared gain. Better correlation alone is
insufficient.

## Authorization boundary

Planned until P1 pilot checkpoints, frozen counterfactual semantics, practical
margins, and resource budget are accepted.

