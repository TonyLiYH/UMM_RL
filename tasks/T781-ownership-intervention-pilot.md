---
id: T781
title: Janus-Pro shared-group reversible intervention pilot
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T781-ownership-intervention-pilot
depends_on: [T711, T780]
blocks: [T782, T783]
allowed_paths: ["tasks/T781-ownership-intervention-pilot.md", "configs/ownership/pilot-v1/", "runs/ownership-pilot-v1/", "reports/T781/", "src/comppareto/adapters/corl/", "src/comppareto/ownership/", "tests/ownership/"]
source_revision: "06154436e528fe91d67c0920505b5e65b14a9895"
created_at: 2026-10-09
updated_at: 2026-10-09
---

# T781: Reversible ownership pilot

## Objective

Implement the T780 structured-group registry and temporary zero gates on the
common Janus-Pro checkpoint. Prove exact restoration, deterministic parameter
membership, and measured U/G effects on two disjoint calibration splits.
Record shared groups and task-private controls separately.

## Entry gate

T780's protocol must be accepted and T711 must establish semantically valid
U/G evaluation paths. Before `ready`, local planning freezes the group/sample
budget, evaluator revisions, resource ceiling, output schema, and first-report
contract. A small implementation smoke may be planned separately before that
gate; this task does not authorize GPU execution yet.

## Acceptance and stop gate

Report continuous paired U/G effects, uncertainty, layer/type strata,
random/magnitude controls, group interactions, evaluator failures, inference
cost, and classification stability. Preserve unresolved groups. Advance to
T782/T783 only if variation is nontrivial and category rules remain stable
across splits at the frozen practical-effect thresholds. A failed gate is a
valid result and closes category-guided training until a new decision.
