---
id: T782
title: Matched ownership-guided pruning test
parent: T700
status: planned
priority: P1
owner: unassigned
reviewer: local-research-agent
branch: agent/T782-ownership-pruning-test
depends_on: [T781, T725]
blocks: [T784]
allowed_paths: ["tasks/T782-ownership-pruning-test.md", "configs/ownership/pruning-v1/", "runs/ownership-pruning-v1/", "reports/T782/", "src/comppareto/ownership/", "tests/ownership/"]
source_revision: "06154436e528fe91d67c0920505b5e65b14a9895"
created_at: 2026-10-09
updated_at: 2026-10-09
---

# T782: Matched pruning test

## Objective

Test whether calibration-only ownership measurements predict held-out U/G
damage better than random groups and conventional importance ranks at matched
removed parameter count and layer/type mix. Measure actual latency and memory.
Separate immediate damage from recovery after equal training budgets.

## Entry and decision gate

T781 must pass its stability gate and T725 must supply a reproducible joint
reference. Before `ready`, freeze selected groups, held-out sets, baselines,
recovery budget, joint success metric, uncertainty rule, and resource ceiling.
If the ownership ranking has no held-out predictive advantage, report that
outcome and do not reselect groups on the held-out results.
