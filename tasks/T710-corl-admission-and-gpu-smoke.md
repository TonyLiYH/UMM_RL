---
id: T710
title: CoRL assets, implementation audit, and GPU optimizer smoke
parent: T700
status: running
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T710-corl-admission-gpu-smoke
depends_on: []
blocks: [T730, T740]
allowed_paths: ["tasks/T710-corl-admission-and-gpu-smoke.md", "configs/corl/admission/", "runs/corl-admission-v1/", "reports/T710/", "src/comppareto/adapters/corl/", "tests/adapters/corl/", "vendor/corl/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16T13:15Z
---

# T710: CoRL assets, implementation audit, and GPU optimizer smoke

## Research claim

The public CoRL/ULM-R1 stack can be converted into a pinned, auditable
Janus-Pro-1B Unified-GRPO training entry point before method comparisons.

## Objective

Download and pin the official Janus-Pro-1B checkpoint and `x2x_rft_22k`
dataset, audit public CoRL against the paper, repair only locally demonstrated
correctness defects behind explicit switches, and run a bounded GPU
rollout/reward/forward/backward/optimizer smoke.

## Frozen protocol

- Pin exact Git, model, dataset, tokenizer, VQ, and reward-model revisions.
- Execute assets and caches from verified local SSD.
- Preserve `upstream_exact` and `corrected_candidate` paths separately.
- Test image-token causal shift, masks, reference separation, reward dispatch,
  reward variance, token reduction, and rollout/teacher-forcing consistency.
- At most 8 optimizer steps on at most 32 unique source records.
- The produced checkpoint is disposable engineering evidence.

## Mandatory implementation checks

1. Enumerate `requires_grad` parameters and optimizer membership.
2. Verify frozen heads still transmit gradients to shared parameters.
3. Test image-token next-token alignment.
4. Test batched MC/OE reward dispatch.
5. Freeze the paper-versus-public-code discrepancy table.
6. Measure reward ranges, zero-variance groups, effective tokens, and clipping.
7. Verify reference immutability when enabled.
8. Keep upstream-exact and corrected-candidate outputs separate.

## GPU smoke

```text
load pinned Janus-Pro-1B
→ load x2x_rft_22k micro-split
→ generate understanding and image candidates
→ compute rewards and separate U/G GRPO losses
→ backward and optimizer step
→ repeat ≤8 steps
→ save/reload disposable checkpoint
→ evaluate fixed smoke examples before/after
```

## First report

Before large download or GPU execution, commit and push revisions, licenses,
asset sizes, local-SSD paths, discrepancy table, correctness tests, commands,
GPU topology, and expected cost.

## Pass/fail gate

Assets must be hash-pinned; mandatory checks must have explicit outcomes;
rollouts, rewards, losses, and gradients must be finite; at least one optimizer
step must change only authorized parameters; checkpoint reload and resource
accounting must pass. A confirmed upstream defect is a valid result but blocks
T730 until a corrected protocol is locally frozen.

## Resource envelope

- at most 8 H20 GPUs;
- at most 16 H20-equivalent GPU-hours;
- at most 32 unique records and 8 optimizer steps;
- no benchmark-scale evaluation.

## Automated submission gate

```bash
bash scripts/validate_task_submission.sh T710
```

## Review history

- 2026-09-16 — Remote executor entered pre-created worktree/branch
  `agent/T710-corl-admission-gpu-smoke` (HEAD `448c517`, already even with
  `origin/main`, no merge needed) and set status to `running`. Starting the
  CPU-side audit: identified the official upstream as GitHub `mm-vl/ULM-R1`
  (paper "Co-Reinforcement Learning for Unified Multimodal Understanding and
  Generation", arXiv:2505.17534; CoRL = Co-Reinforcement Learning), pinned
  `HEAD` `0c92629f9b307a32bb286ae3562809e941d1bb0b` on `main`, Apache-2.0
  licensed. Confirmed target assets resolve: `deepseek-ai/Janus-Pro-1B` (MIT,
  HF sha `960ab33191f61342a4c60ae74d8dc356a39fafcb`) and
  `mm-vl/x2x_rft_22k` (HF dataset sha `f52833ce01b5657294bed87f23f27d04b92838b9`,
  22,479 rows, ~11GB parquet, ungated). Read the full `corl/open_r1/` training
  stack (`grpo_janus_unify.py`, `trainer/grpo_trainer_unified.py`,
  `rewards/r_base.py`, `rewards/r_t2i.py`) and `janus/models/modeling_vlm.py`
  read-only from a shallow clone at `/tmp/corl_audit` (not committed).
  Publishing the first report next, before any download/GPU execution, per
  the task's explicit gate.
