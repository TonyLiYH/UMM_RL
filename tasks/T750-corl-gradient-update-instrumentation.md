---
id: T750
title: Per-task GRPO gradient and optimizer-update instrumentation
parent: T700
status: ready
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T750-corl-gradient-update-instrumentation
depends_on: []
blocks: [T760, T770]
allowed_paths: ["tasks/T750-corl-gradient-update-instrumentation.md", "src/comppareto/instrumentation/", "tests/instrumentation/", "configs/instrumentation/corl/", "reports/T750/", "runs/instrumentation-corl-v1/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T750: Per-task GRPO gradient and optimizer-update instrumentation

## Objective

Implement a framework-neutral instrumentation layer, with a CoRL adapter, that
records per-task shared gradients before clipping, combined gradients after
negotiation/clipping, and realized AdamW parameter updates by block.

## Required measurements

- per-task norm, cosine, norm ratio, and finite status;
- block-local statistics and overlap IDs;
- PCGrad-ready vectors and MGDA convex-hull inputs;
- pre/post-clipping norms and clipping coefficient;
- AdamW moment summaries and realized \(\Delta\theta\);
- direction norm, near-zero rate, and directional derivatives;
- rollout, token, reward-call, backward, and wall-time counts.

## Execution

1. Deterministic toy tests with shared/private modules.
2. Sequential task batches at one shared-state hash.
3. Prove instrumentation does not alter gradients or optimizer results.
4. Integrate a mock CoRL-compatible model.
5. Optionally use T710 assets for one real diagnostic batch if available.

## Pass/fail gate

Instrumented and uninstrumented updates must match within dtype tolerance;
ownership must be complete; logs must reconstruct the combined gradient and
realized update; no silent device fallback is allowed.

## Resource envelope

- CPU by default;
- optional real-model smoke at most 1 H20 GPU-hour;
- no persistent research training.

## Automated submission gate

```bash
bash scripts/validate_task_submission.sh T750
```

