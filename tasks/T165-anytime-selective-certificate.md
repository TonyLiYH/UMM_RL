---
id: T165
title: Anytime-valid selective finite-response controller
parent: T100
status: planned
priority: P2
owner: unassigned
reviewer: local-research-agent
branch: agent/T165-anytime-selective-certificate
depends_on: [T160, T170]
blocks: [T300]
allowed_paths: ["docs/math/", "docs/theory/", "src/comppareto/", "tests/", "configs/t1c/", "runs/t1c-selective-*/", "reports/T165/", "tasks/T165-anytime-selective-certificate.md"]
source_revision: "c481c0ee25245ebace0d6d50d881145a972046cf"
created_at: 2026-10-05
updated_at: 2026-10-09
---

# T165: Anytime-valid selective finite-response controller

## Research claim

A finite-response Pareto method can make a statistically valid commit,
reject, or continue decision under adaptive certification and optional
continuation, rather than treating uncertainty as a one-shot error bar.

## Objective

Construct a joint support-aware confidence sequence for finite-response
hypergradients and use it to control a sequential direction/step controller.
The controller may purchase task samples, edge/block probes, additional native
response steps, or tangent refinements until a robust margin is positive or a
budget is exhausted.

## Dependencies and inputs

Accepted T160 posterior trajectory bounds, accepted T170 joint robust
certificate primitives, and the exact finite-response oracle T155.

## Frozen protocol

The target is the rerun-response value (F_i^{K,\mathrm{rerun}}). Selection
and certification data must be separated or covered by a time-uniform event.
Optional continuation is part of the protocol. A fixed-sample confidence
interval reused after adaptive stopping is not sufficient.

## Execution stages

1. Implement a fixed-sample blockwise baseline and an anytime-valid scalar
   directional baseline.
2. Implement a joint covariance-aware confidence sequence on sparse overlap
   families; retain the union-bound construction as a reference.
3. Implement commit/reject/continue logic and deterministic tangent-radius
   accounting.
4. Implement uniform, continuous-KKT, oracle, and value-of-information
   allocation baselines.
5. Evaluate disjoint, chain, star, random-sparse, and full-overlap families.

## Pass/fail gate

The controller must satisfy the preregistered simultaneous coverage and false-
safe thresholds, detect deliberately unstable trajectories, and improve
certified-step recall or total cost over fixed uniform certification on at
least one sparse family. Any advantage must disappear under full overlap.

## First report

Publish the filtration/query model, confidence-sequence construction,
stopping-time proof, joint covariance assumptions, query costs, controller
state machine, allocation baselines, thresholds, seeds, and CPU estimate
before execution.

## Required deliverables

Proof note, confidence-sequence implementation, controller tests, frozen
configurations, manifests, coverage/false-safe/recall/cost tables, allocation
regret plots, unstable-trajectory cases, and a claim-check report.

## Artifact and provenance requirements

Every result maps to graph, covariance model, query schedule, stopping time,
confidence level, seed, response horizon, tangent approximation, source
revision, configuration hash, and exact-oracle outcome.

## Failure and retry rules

Do not treat optional stopping as harmless. Do not replace joint coverage with
per-task marginal coverage. Do not infer stochastic independence from sparse
parameter overlap. Any change to the filtration, confidence construction, or
query cost model after results are visible requires local review.

## Successor opening

Accepted T165 contributes to T100 and is required before T300 can be promoted
to ready.

## Review history

- 2026-10-05 — Planned as the sequential extension of T160/T170.

