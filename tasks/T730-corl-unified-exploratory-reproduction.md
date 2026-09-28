---
id: T730
title: Unified GRPO joint and private-enabled headroom pilots
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T730-corl-unified-exploratory-reproduction
depends_on: [T711, T755]
blocks: [T756, T760]
allowed_paths: ["tasks/T730-corl-unified-exploratory-reproduction.md", "configs/corl/unified-v1/", "runs/corl-unified-v1/", "reports/T730/", "src/comppareto/adapters/corl/", "tests/adapters/corl/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T730: Unified GRPO joint and private-enabled headroom pilots

## Objective

Run bounded one-seed joint training pilots from the same Janus-Pro-1B
checkpoint under two explicitly different trainability policies:

1. faithful frozen-private CoRL-derived Unified GRPO;
2. an audited small-private-module configuration applied equally to all
   comparison methods.

Both task capabilities must be evaluated on fixed, held-out evaluators.
The purpose is to establish joint headroom and whether trainable-private
capacity creates a response opportunity; it is not a paper reproduction claim.

## Required design

- fixed semantically corrected GRPO protocol from T711;
- real instrumentation from T755;
- at least early, middle, and final training snapshots;
- fixed behavior/data/evaluator policy within each method;
- reported task exposure, total GPU time, rollouts, reward calls, effective
  tokens, backwards, parameter masks, and rejected/invalid groups;
- same private mask across the two joint configurations and future comparators.

## Advance gate

Advance to T756 only if at least one joint configuration has measurable
learning headroom on both development metrics or leaves a clear trade-off
relative to the single-task references. Absence of headroom triggers one
preregistered horizon extension, then stops this model/setup rather than
opening broader method sweeps.

## Authorization boundary

Planned only. The local reviewer must freeze data exposure, horizon, evaluator,
practical margin, corrected protocol, private mask, and GPU budget after T711
and T755. No executor may start from this placeholder.
