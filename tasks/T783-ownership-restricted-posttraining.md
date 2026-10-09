---
id: T783
title: Matched restricted U/G post-training test
parent: T700
status: planned
priority: P1
owner: unassigned
reviewer: local-research-agent
branch: agent/T783-ownership-restricted-posttraining
depends_on: [T781, T725, T740]
blocks: [T784]
allowed_paths: ["tasks/T783-ownership-restricted-posttraining.md", "configs/ownership/restricted-training-v1/", "runs/ownership-restricted-training-v1/", "reports/T783/", "src/comppareto/adapters/corl/", "src/comppareto/ownership/", "tests/ownership/"]
source_revision: "06154436e528fe91d67c0920505b5e65b14a9895"
created_at: 2026-10-09
updated_at: 2026-10-09
---

# T783: Restricted post-training test

## Objective

Test whether opening prespecified shared groups changes U-only, G-only, and
joint CoRL outcomes in the direction predicted by the frozen ownership map.
Compare against random-group, full-shared, and private-only controls from the
same checkpoint.

## Entry and decision gate

T781 must pass, T725 must establish the joint reference, and T740 must establish
single-task headroom. Before `ready`, freeze group selection, trainable
parameter counts, task exposure, rollouts, reward calls, optimizer rule,
search allowance, GPU time, held-out metrics, practical effect, and confidence
rule. Record realized shared and private parameter changes. Advance a method
claim only if a prespecified joint criterion improves at matched budget;
otherwise retain the negative or mixed result.
