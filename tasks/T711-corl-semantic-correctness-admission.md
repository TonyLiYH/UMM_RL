---
id: T711
title: CoRL semantic correctness and fixed-anchor admission
parent: T700
status: ready
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T711-corl-semantic-correctness-admission
depends_on: []
blocks: [T730, T740, T755]
allowed_paths: ["tasks/T710-corl-admission-and-gpu-smoke.md", "tasks/T711-corl-semantic-correctness-admission.md", "configs/corl/admission/", "runs/corl-admission-v1/", "reports/T710/", "reports/T711/", "src/comppareto/adapters/corl/", "tests/adapters/corl/", "vendor/corl/"]
source_revision: "1fcb9e9964c678a330f32514f211828d551dc48c"
created_at: 2026-09-28
updated_at: 2026-09-28
---

# T711: CoRL semantic correctness and fixed-anchor admission

## Research claim

A Unified-GRPO baseline is suitable for optimization research only after its
image policy scoring, behavior-policy relation, reward dispatch, reduction,
parameter ownership, and counterfactual anchors have semantic—not merely
finite—validation.

## Objective

Repair and validate the T710 blockers using a pinned CoRL fork plus explicit
protocol switches. Freeze a correctness-repaired research protocol that later
tasks can use without silently claiming a paper reproduction.

## Required protocol identities

Every report/config must distinguish:

1. **paper-described**: published prose with unresolved details named;
2. **upstream-executable**: released code after path/dependency resolution;
3. **corrected research protocol**: explicit semantic fixes used by this
   project;
4. **reduced smoke**: small resource-limited engineering run.

## Mandatory semantic tests

### A. Image autoregressive scoring

Using two real prompts of unequal length and retained image-token trajectories:

- compare cached generation, prefix-only scoring, and full teacher forcing at
  every token position;
- test current-position and preceding-position target alignment;
- perturb a target/future image token while holding the queried prefix fixed;
- verify a valid next-token distribution is invariant to the target and future
  token perturbations;
- run an FP32 reference where practical and report BF16 deviation separately.

### B. Sampling/scoring identity

- establish a CFG=1, no-truncation correctness baseline;
- record all logits processors, temperature, top-k/top-p, masks, and behavior
  log probabilities;
- if CFG sampling remains supported, label it as a distinct guided-policy
  surrogate and do not call conditional-only scores its behavior probability.

### C. QA reward dispatch

- per-completion dispatch for MC and OE;
- singleton, all-MC, all-OE, and mixed batches with \(G=2\);
- verbose MC answer, bare choice, exact OE, partial OE, and invalid/missing
  type fixtures;
- permutation equivariance and explicit rejection of length mismatch.

### D. Rewards, reduction, and masks

- evaluate each fixed example alone and in a left-padded unequal-length batch;
- demonstrate padding-invariant TIM spans using semantic masks;
- record per-group reward vectors, standard deviations, advantages, EOS masks,
  effective U/G token counts, and declared reduction;
- verify a zero-variance group is handled explicitly.

### E. Update and state integrity

- actual optimizer parameter object IDs and duplicate checks;
- separate U-only and G-only gradients through frozen heads;
- tensor-level authorized parameter and reference-model equality;
- save/resume equality of U/G log probabilities, optimizer moments, scheduler,
  RNG, and resolved configurations.

### F. Fixed-anchor finite-response surrogate

For one diagnostic window, freeze trajectories, behavior log probabilities,
masks, reward/evaluator snapshots, advantages, private optimizer state, RNG,
and a disjoint audit batch. Candidate perturbations must evaluate the same
anchored surrogate. This task does not conduct full response experiments; it
only establishes a valid counterfactual object.

## Pass/fail gate

Pass requires all semantic tests above to pass at preregistered tolerance.
Any unresolved causal scoring, behavior/scoring mismatch, reward dispatch, or
state-restoration defect blocks T730/T740/T755. A finding that the upstream
path differs from the corrected research protocol is expected and must remain
visible in the deliverables.

## Resource envelope

- at most 2 H20 GPUs;
- at most 16 H20-equivalent GPU-hours;
- at most 64 unique source records;
- at most 12 optimizer steps;
- no benchmark-scale training.

## First report

Commit and push test cases, semantic invariants, tolerances, exact protocol
delta, model/data/reward revisions, storage paths, commands, and cost estimate
before GPU execution.

## Automated submission gate

```bash
bash scripts/validate_task_submission.sh T711
```

