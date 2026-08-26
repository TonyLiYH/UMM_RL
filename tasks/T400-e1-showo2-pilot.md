---
id: T400
title: E1 Show-o2 controlled pilot
parent: T000
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T400-e1-showo2-pilot
depends_on: [T300]
blocks: [T500, T600]
allowed_paths: ["src/comppareto/", "tests/", "configs/e1/", "experiments/", "runs/e1-*/", "reports/T400/", "tasks/T4*.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T400: E1 Show-o2 pilot

## Research claim

At equal compute and identical private response, CompPareto should improve worst-task retained gain over strong scalarization, MOBLO, MGDA, Chebyshev, and Nash baselines.

## Objective

Freeze budgets, run baselines and proposed variants, then perform confirmatory evaluation.

## Dependencies and inputs

Accepted D0 evidence and an accepted Show-o2 admission.

## Allowed changes

E1 training/evaluation code, configs, manifests, and reports within the declared paths.

## Frozen protocol

No pretraining; tokenizer/VAE remain frozen unless a local ablation task says otherwise.

## Execution stages

T410, T420, T430, and T440.

## Pass/fail gate

Confirmatory worst-task gain and hypervolume satisfy the preregistered intervals and no capability guardrail fails.

## First report

Return measured memory, throughput, estimator overhead, budget table, and smoke success before the full wave.

## Required deliverables

Resolved budgets, baseline and method runs, per-seed metrics, capability slices, costs, failures, and claim audit.

## Artifact and provenance requirements

Formal manifests record checkpoints, hashes, task updates, tokens/samples, FLOPs, GPU-hours, wall-clock, and metric revisions.

## Failure and retry rules

Only preregistered infrastructure retries are allowed; numerical divergence remains a completed failed trial.

## Successor opening

Accepted T400 opens T500 and permits local consideration of T600.

## Review history

- 2026-08-26 — Planned; D0 is not accepted.

