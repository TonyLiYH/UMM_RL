# Show-o2 first-attempt plan

## 1. Purpose

The first Show-o2 attempt is paired with an exact-oracle track. It is an
admission and diagnostic-feasibility programme, not a joint-training result. It
should answer three real-model questions in order:

1. Can the official checkpoint and both task paths be reproduced?
2. Can shared/private blocks and independent optimizer state be identified,
   snapshotted, restored, and differentiated?
3. Can a small, reversible finite-response experiment measure raw,
   commit-response stop-gradient, and rerun-response finite-unroll signals
   without exceeding the declared memory and compute budget?

The attempt does not test whether CompPareto improves final model quality. That
claim remains behind T1b, T170, and the formal D0 Gate.

One executor may perform T155 and T210 sequentially. The tasks still use
separate branches and evidence packages so that oracle failures and Show-o2
admission failures remain independently reviewable.

## 2. Scientific claims served

| Claim | Evidence in this attempt | Excluded conclusion |
|---|---|---|
| Show-o2 is an executable public pilot | Pinned official code/checkpoint, license, dependencies, understanding and generation smoke | No claim of reproduced headline benchmark quality unless the official evaluation is run under a frozen tolerance |
| Its parameterization supports compensation-aware diagnostics | Explicit block registry draft and optimizer-state ownership map | No claim that the chosen partition is uniquely correct |
| Finite-response differentiation is technically feasible | Reversible \(K=1\), then optional \(K=3\), subspace unroll and finite-difference checks | No claim that the resulting estimator is accurate at full-model scale |
| State-aware signals differ measurably from raw or stop-gradient signals | Logged directional derivatives on a small frozen diagnostic batch | No claim of better final training performance |

## 3. Execution DAG

```text
T155 exact finite-response oracle ───────────────┐
                                                 │
T210 Show-o2 admission
  ├─ official revision, license, checkpoint and dependency audit
  ├─ understanding smoke
  ├─ generation smoke
  └─ shared/private block-map draft
          ↓ local acceptance
T215 Show-o2 finite-response diagnostic feasibility
  ├─ deterministic block registry
  ├─ parameter and optimizer-state snapshot/restore
  ├─ raw and commit-response gradients
  ├─ rerun-response exact unroll on a selected subspace
  ├─ finite-difference and rollback checks
  └─ memory, FLOPs, wall-clock and artifact report
          ↓ local acceptance plus T100/T170 acceptance
T300 formal held-out D0 diagnostics
```

T155 and T210 may be executed sequentially by one remote Agent, or in parallel
with the currently authorized T110, T120, and T130 tasks. T215 remains planned
until T210 is accepted; T160 later consumes accepted T155 plus T110–T130.

## 4. T210 admission protocol

### Stage 1 — first report, no expensive GPU work

The executor records:

- official Show-o repository URL and exact revision;
- checkpoint identifier, source, measured size, and expected hash procedure;
- model and code licenses plus redistribution constraints;
- tokenizer, VAE, text encoder, scheduler, and evaluation dependencies;
- understanding and generation entry points;
- expected VRAM, storage, wall-clock, and GPU count;
- missing assets, credentials, incompatible versions, or unofficial patches;
- proposed parameter-block boundaries and ambiguous modules.

The first report is committed to the task branch before expensive execution.

### Stage 2 — environment and checkpoint smoke

- build a pinned environment;
- load the official checkpoint without modifying its weights;
- record library and CUDA versions;
- record peak host and GPU memory;
- hash checkpoint files and resolved configuration;
- fail explicitly if CUDA is required but unavailable.

### Stage 3 — task-path smoke

- run one official or minimally adapted understanding example;
- run one official or minimally adapted generation example;
- save sanitized commands, prompts/inputs, sampler settings, outputs, and logs;
- distinguish execution success from metric reproduction.

### Stage 4 — block-map draft

For every relevant module, record:

- fully qualified name;
- shape and parameter count;
- understanding activity;
- generation activity;
- proposed shared/private/frozen role;
- optimizer ownership;
- ambiguity or dynamic-routing caveat.

No arbitrary ownership assignment is allowed.

## 5. T215 finite-response feasibility protocol

### Frozen operational semantics

The first diagnostic implements both:

- **commit response:** run the private response at the current shared state and
  hold the resulting private state fixed while differentiating a candidate
  shared change;
- **rerun response:** rerun the same private response at the candidate shared
  state and differentiate through the complete selected state transition.

These quantities are logged separately.

### Minimal trainable subspace

The executor proposes, before running:

- one small shared block active in both tasks;
- one understanding-private block;
- one generation-private block;
- frozen tokenizer, VAE, and unrelated backbone blocks;
- \(K=1\) as the mandatory response horizon;
- \(K=3\) only if \(K=1\) passes memory and correctness checks.

The purpose is diagnostic identifiability, not model-quality improvement.

### Required comparisons

For each task and selected block:

1. raw shared gradient;
2. post-adaptation commit-response stop-gradient;
3. rerun-response exact finite unroll;
4. directional finite-difference reference;
5. parameter-only state differentiation versus complete optimizer-state
   differentiation when the optimizer has memory.

### Correctness checks

- snapshot and restore parameters, gradients, optimizer tensors, counters, RNG
  state, and data-order state;
- verify that a rolled-back diagnostic leaves the persistent model state
  unchanged;
- compare automatic differentiation with central finite differences on a
  fixed low-dimensional direction;
- record unsupported or nondifferentiable operations rather than silently
  detaching them;
- use disjoint adaptation and meta batches in the main smoke, with same-batch
  evaluation only as a named diagnostic.

### Initial numerical gates

- relative directional-derivative error at most \(10^{-3}\) when the reference
  magnitude exceeds \(10^{-8}\);
- absolute error at most \(10^{-6}\) near zero;
- restored floating tensors match their snapshots within the dtype-appropriate
  exact or declared numerical tolerance;
- optimizer counters, RNG state, and data-order identifiers match exactly;
- no persistent parameter update occurs;
- peak memory and wall-clock are recorded for every estimator;
- any OOM, nondifferentiable transition, missing optimizer state, or unstable
  finite difference remains a reported failure.

These are feasibility gates, not D0 scientific thresholds.

## 6. Data and evaluation controls

The attempt uses a small, immutable diagnostic manifest containing:

- one understanding batch group;
- one generation batch group;
- adaptation/meta split identifiers;
- prompts or question-answer inputs;
- preprocessing and sampler parameters;
- source, license, hashes, and decontamination status where applicable.

The first attempt does not tune on these examples and does not use their output
as a final benchmark result.

## 7. Resource envelope

### T210

- first report: CPU and metadata work only;
- smoke target: one GPU where supported;
- executor reports the measured requirement before expanding resources;
- no joint training or hyperparameter search.

### T215

- default cap: two GPUs and eight H20-equivalent GPU-hours;
- \(K=1\) must pass before \(K=3\);
- no full-backbone unroll;
- no multi-seed quality comparison;
- stop after the first reproducible feasibility matrix or at the resource cap.

Changing this envelope requires a local task revision.

## 8. Success, failure, and next decision

### T210 passes when

- official sources and licenses are pinned;
- both task paths execute;
- external components are fully recorded;
- trainable shared/private blocks are auditable;
- no hidden dependency prevents local modification and evaluation.

### T215 passes when

- state snapshot/restore is reproducible;
- raw, commit-response, and rerun-response quantities are separately
  measurable;
- at least one selected shared/private subspace passes the finite-difference
  gate;
- complete optimizer-state differentiation can be implemented or its exact
  blocker is demonstrated;
- resource measurements support a credible D0 budget.

### Stop or revise when

- the public training interface cannot expose the required shared/private
  transition;
- optimizer state cannot be restored reproducibly;
- differentiation relies on undeclared detachments;
- the selected response exceeds the resource envelope;
- the two task paths depend on incompatible or unavailable assets.

Passing T210/T215 establishes technical feasibility only. Formal predictive
validity remains a T300 result.
