---
id: T760
title: CoRL traditional multi-task negotiator wave
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T760-corl-traditional-negotiator-wave
depends_on: [T730, T740, T755, T756]
blocks: [T770]
allowed_paths: ["tasks/T760-corl-traditional-negotiator-wave.md", "configs/corl/negotiators-v1/", "runs/corl-negotiators-v1/", "reports/T760/", "src/comppareto/negotiators/", "tests/negotiators/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T760: CoRL traditional multi-task negotiator wave

## Objective

Screen the smallest set of strong conventional alternatives under frozen task
objectives, parameter masks, tuning expenditure, and evaluators. Start with
tuned scalarization, shared-then-private, persistent private-then-shared,
one strongest conventional negotiator, and raw-direction line search using
the same validation budget. Treat advantage/reward coupling and loss
reduction as separate ablation axes, not as part of a negotiator.

## Authorization boundary

Planned until P1/P2 results establish learning headroom and response
opportunity. The local side freezes practical margins, compute and search
budgets, evaluator, and all comparison masks before opening.
