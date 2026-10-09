---
id: T725
title: CoRL Unified-only reference reproduction
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T725-corl-unified-reference
depends_on: [T711, T712]
blocks: [T760]
allowed_paths: ["tasks/T725-corl-unified-reference.md", "configs/corl/unified-reference-v1/", "runs/corl-unified-reference-v1/", "reports/T725/", "src/comppareto/adapters/corl/", "tests/adapters/corl/"]
source_revision: "e001548976ede8c8dd9805e5602fc2ab9ac6274b"
created_at: 2026-10-09
updated_at: 2026-10-09
---

# T725: CoRL Unified-only reference reproduction

## Objective

Run an auditable Unified-only reference from the same base checkpoint as T712.
Measure understanding and generation on the same trained checkpoint, then
compare to the base and to published CoRL figures where their data and
evaluation protocols are genuinely comparable.

## Entry and protocol freeze

T711 and T712 must be locally accepted. Before execution, freeze the
paper-described recipe, public executable recipe, corrected-protocol delta,
training horizon, batch exposure, seed, train/dev/test IDs, reward/evaluator
versions, checkpoint selection rule, metric definitions, paper-comparison
eligibility, and pass tolerances. Derive the GPU and storage ceilings from
T712's measured throughput and checkpoint bytes; the local review side must
write them into this task before marking it `ready`.

## Required execution and evidence

- Run one declared seed through the full frozen Unified-only horizon; retain
  base, intermediate, and final checkpoint hashes and training curves.
- Evaluate both capabilities on the same checkpoint using frozen suites and
  report uncertainty or sample counts with every metric.
- Reconcile examples, tokens, optimizer steps, rollouts, reward calls,
  backwards, GPU-hours, wall time, memory, and checkpoint storage against the
  recipe. Explain any divergence from published settings.
- Compare base, trained, and eligible paper-reported Unified-only numbers at
  preregistered tolerances. Report non-comparable figures as context only.
  Do not tune on held-out test results to hit a paper number.

## Acceptance distinction

`Framework reproduced` requires a complete, auditable run of the frozen
corrected protocol and valid U/G metrics. `Paper result reproduced` additionally
requires comparable published metrics to fall within preregistered
tolerances. A mismatch is a result to diagnose and does not license a hidden
recipe change. If the projected full run exceeds the frozen budget, return a
measured blocker and revised estimate rather than run a shorter horizon under
the same claim.

## First report and deliverables

The first report must contain the exact recipe/protocol delta, evaluator and
split audit, T712-based resource projection, comparison tolerances, and first
run status. Deliver the manifest, logs, complete curves and metrics, checkpoint
hashes, cost ledger, paper comparison table, discrepancy analysis, and claim
check under `reports/T725/` and `runs/corl-unified-reference-v1/`.
