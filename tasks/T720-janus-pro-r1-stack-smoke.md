---
id: T720
title: Janus-Pro-R1 reusable SFT and GRPO stack smoke
parent: T700
status: ready
priority: P1
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T720-janus-pro-r1-stack-smoke
depends_on: []
blocks: []
allowed_paths: ["tasks/T720-janus-pro-r1-stack-smoke.md", "configs/janus-pro-r1/admission/", "runs/janus-pro-r1-stack-v1/", "reports/T720/", "src/comppareto/adapters/janus_pro_r1/", "tests/adapters/janus_pro_r1/", "vendor/janus-pro-r1/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T720: Janus-Pro-R1 reusable SFT and GRPO stack smoke

## Objective

Pin and download the public Janus-Pro-R1 code, Janus-Pro checkpoint, released
bounded training data, and smallest usable reward setup; run one SFT optimizer
smoke and one GRPO rollout/backward/optimizer smoke; identify reusable
dataloader, checkpoint, optimizer, reward, and rollout modules.

## Frozen protocol

- Do not claim Janus-Pro-R1 is a CoRL reproduction.
- Use released bounded examples or deterministic subsets.
- Keep SFT and RL environments separately pinned when required.
- Execute from verified local SSD.
- At most 4 SFT and 4 GRPO optimizer steps.
- Reward models use a pinned official offline configuration or a fully
  documented service endpoint.

## Required checks

- source/model/data/reward licenses, revisions, hashes, and sizes;
- SFT loss/backward/optimizer/scheduler/save/resume;
- GRPO rollout/reward/loss/backward/optimizer/save/resume;
- trainable/frozen parameter inventory;
- memory and GPU-time accounting;
- reusable-component comparison against CoRL needs.

## First report

Commit and push the asset inventory, environment split, data format,
reward-model plan, commands, GPU topology, and expected cost before execution.

## Pass/fail gate

Both smokes must produce finite losses, authorized parameter changes, and
reloadable checkpoints. Missing required code, model, license, or reward path
is blocking.

## Resource envelope

- at most 8 H20 GPUs;
- at most 12 H20-equivalent GPU-hours;
- no benchmark-scale training or evaluation.

## Automated submission gate

```bash
bash scripts/validate_task_submission.sh T720
```
