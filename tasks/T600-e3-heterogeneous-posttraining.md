---
id: T600
title: E3 heterogeneous preference and RL post-training
parent: T000
status: planned
priority: P1
owner: unassigned
reviewer: local-research-agent
branch: agent/T600-e3-heterogeneous-posttraining
depends_on: [T400]
blocks: []
allowed_paths: ["src/comppareto/", "tests/", "configs/e3/", "runs/e3-*/", "reports/T600/"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T600: E3 heterogeneous post-training

## Research claim

Finite native private responses should remain useful when understanding and generation use different preference or reward estimators.

## Objective

Combine understanding DPO/OPD with generation Flow-DPO/GRPO only after the supervised E1 claim passes.

## Dependencies and inputs

Accepted T400, frozen reward services, and explicit online-rollout budget accounting.

## Allowed changes

E3 adapters, reward interfaces, configs, tests, manifests, and reports.

## Frozen protocol

Exact best response is not claimed for online rollouts; the operational object is finite optimizer-state-aware adaptation.

## Execution stages

Reward smoke, variance audit, equal-budget baseline wave, proposed method wave, and guardrail evaluation.

## Pass/fail gate

Positive worst-task gain with acceptable KL drift, reward overoptimization, stability, and overhead.

## First report

Publish reward revisions, rollout schema, estimator variance, service capacity, and expected GPU cost.

## Required deliverables

Reward health checks, run manifests, task metrics, KL/variance/acceptance traces, failures, and claim audit.

## Artifact and provenance requirements

Rollout and reward-service revisions are mandatory in every formal manifest.

## Failure and retry rules

Reward collapse or unbounded drift stops the task; do not tune on final test metrics.

## Successor opening

No automatic successor; local review decides broader modalities.

## Review history

- 2026-08-26 — Planned; E1 is not accepted.

