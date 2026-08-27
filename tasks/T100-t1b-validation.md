---
id: T100
title: T1b independent solver and approximation validation
parent: T000
status: running
priority: P0
owner: local-research-agent
reviewer: user
branch: feature/t1b-validation
depends_on: []
blocks: [T160, T300]
allowed_paths: ["src/comppareto/", "tests/", "configs/", "runs/t1b-*/", "reports/T100/", "tasks/T1*.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T100: T1b validation

## Research claim

The compensation-aware quadratic and approximation machinery must match independent numerical references before it is applied to model gradients.

## Objective

Complete random overlap, KKT reference, indefinite-curvature, approximation-error, negotiation-audit, finite-response posterior-error, and graph-localized robust-certificate evidence.

## Dependencies and inputs

T1a code and manifest are the baseline. T110–T170 provide independent evidence.

## Allowed changes

Synthetic solvers, tests, configurations, run metadata, and T1b reports only.

## Frozen protocol

Reference implementations must be independent of the formula under test; failures and counterexamples remain recorded.

## Execution stages

Execute T110, T120, and T130; use their outputs to open T140, T150, and T160; use accepted T150 and T160 to open T170; integrate a T1b decision report.

## Pass/fail gate

All seven children are accepted, independent-solver and finite-response posterior errors satisfy the preregistered thresholds, the robust certificate meets its simultaneous-coverage gate, and no unhandled trust-region counterexample remains.

## First report

Publish proposed task-family dimensions, seeds, reference solvers, tolerances, and expected CPU cost before implementation.

## Required deliverables

Tests, configurations, manifests, error curves, KKT residuals, trajectory-residual bounds, certificate-coverage evidence, allocation regret, counterexamples, and a parent-level claim check.

## Artifact and provenance requirements

All generated numeric evidence is committed or referenced with source revision and configuration hash.

## Failure and retry rules

Numerical failures are test cases; changing tolerances requires local review.

## Successor opening

Accepted T100 and T210 jointly unblock T300.

## Review history

- 2026-08-26 — Local planner opened T1b coordination; no terminal review decision.
