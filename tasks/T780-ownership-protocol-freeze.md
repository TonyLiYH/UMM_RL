---
id: T780
title: Janus-Pro shared-group and U/G evaluation protocol freeze
parent: T700
status: ready
priority: P0
owner: local-research-agent
reviewer: user
branch: main
depends_on: []
blocks: [T781, T782, T783]
allowed_paths: ["tasks/T780-ownership-protocol-freeze.md", "docs/plans/", "configs/ownership/protocol-v1/", "reports/T780/"]
source_revision: "06154436e528fe91d67c0920505b5e65b14a9895"
created_at: 2026-10-09
updated_at: 2026-10-09
---

# T780: Freeze the ownership measurement protocol

## Objective

Make the current parameter-ownership hypothesis executable without a GPU run.
Specify complete attention-head and MLP-channel intervention groups in the
Janus-Pro shared backbone, their projections, layer/type strata, parameter
counts, and task-private controls. Specify frozen understanding and generation
evaluation sets, matched capability pairs, decoding/sampling seeds, metrics,
noise checks, calibration/held-out split IDs, and contamination checks.

## Required decisions

Define the zero-gate intervention and restoration test; practical-effect and
equivalence thresholds from evaluator noise and random-group controls; paired
uncertainty estimates; unresolved classification; group interaction probes;
and the maximum pilot groups, examples, GPU-hours, and storage. Record the
exact common pre-RL checkpoint and source revisions or a named blocker.

## Pass/fail gate

The protocol is ready for user review when every analysis unit, metric, split,
threshold rule, control, resource ceiling, and stop condition is explicit and
can be implemented without using held-out results to choose groups. This task
performs planning and source inspection only. It authorizes no GPU run and no
claim that an ownership map is stable.

## Deliverables

A frozen protocol and machine-readable group/evaluation manifest under
`configs/ownership/protocol-v1/`, a source-backed ambiguity ledger, and a
reviewable decision summary under `reports/T780/`. Record any unavailable
checkpoint or evaluator as a blocker rather than replacing it silently.
