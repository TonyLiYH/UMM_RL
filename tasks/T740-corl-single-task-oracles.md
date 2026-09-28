---
id: T740
title: CoRL understanding-only and generation-only GRPO references
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T740-corl-single-task-oracles
depends_on: [T711, T755]
blocks: [T756, T760]
allowed_paths: ["tasks/T740-corl-single-task-oracles.md", "configs/corl/oracles-v1/", "runs/corl-oracles-v1/", "reports/T740/", "src/comppareto/adapters/corl/", "tests/adapters/corl/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T740: CoRL single-task GRPO references

## Objective

Run compute-matched understanding-only and generation-only GRPO from the same
checkpoint and data universe to define negative transfer relative to
single-task reference trajectories.

## Required design

Run both:

1. matched per-task exposure references, for interference attribution;
2. matched total accelerator-budget references, for practical comparison.

The task must not call these runs mathematical oracles or assume that they are
global optima. Report held-out capability metrics, task exposure, total compute,
rollouts, reward calls, effective tokens, backwards, parameter masks, and
checkpoint schedule.

## Authorization boundary

Planned until T711/T755 are accepted and the local side freezes exposure,
total-compute, rollout, token, reward-call, optimizer-step, seed, evaluator,
and practical-effect budgets.
