# T215 — First report

Published before any GPU execution, per `tasks/T215-showo2-finite-response-feasibility.md`'s
"First report" requirement. Covers the proposed diagnostic subspace, state inventory, batch/seed
manifest, finite-difference design, expected memory/runtime, required assets, and exact commands.

## 0. Source and scope

- Accepted T210 evidence is the sole basis for the model/environment facts below; no new download
  or checkpoint substitution is made. Official Show-o2 repo pinned at commit
  `45a5a2de01d1ebd10cd5864d29310a76476cdf23` (unchanged from T210), local checkout at
  `/apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/Show-o/show-o2/` (outside this
  repo, per `dev-env-paths.md`).
- Model/tokenizer/VAE/HF-cache assets already migrated to local SSD by T210 at
  `/dockerdata/t210-showo2/` inside the H20-FoldUMM container; reused read-only, no re-migration.
- Accepted environment reused as-is: dedicated venv `/root/venvs/showo2`
  (`configs/admission/showo2/environment-lock.md`), `torch==2.5.1+cu124`,
  `transformers==4.47.0`, `diffusers==0.31.0`. This task does not modify that lock file (outside
  `allowed_paths`); it is referenced, not owned, by T215.
- **Data-integrity note**: this task file's `source_revision` was corrected from a malformed value
  (`217d183473a14ad48852205ea3f2746301915729`, not a git object in this repository) to the real
  commit `217d183b30995db4ac82158259f45800e57e2eb1` — see the task file's review history for the
  full record. This correction is unrelated to the diagnostic design below.

## 1. Proposed minimal trainable subspace

Per `docs/plans/showo2-first-attempt.md`'s minimal-subspace requirement (one small shared block,
one understanding-private block, one generation-private block, frozen tokenizer/VAE/unrelated
backbone), directly measured on the loaded checkpoint (CPU-only introspection, no GPU touched, same
venv/cache as the accepted admission):

| Role | Module path | Parameter count | Notes |
|---|---|---|---|
| Shared (active in both paths) | `model.fusion_proj` | 6,493,824 | Matches `configs/admission/showo2/parameter-block-registry.yaml` exactly. Confirmed via source read (`models/modeling_showo2_qwen2_5.py`): called unconditionally inside `forward()` whenever `image_latents is not None`; `t2i_generate()` calls `self(...)` → `forward()` with `image_latents` set, so both `inference_mmu.py`'s and `inference_t2i.py`'s call paths execute this module. Only skipped on a pure-text-only branch neither task path uses. |
| Understanding-private | `model.und_trans.layers[0]` | 15,239,504 | One transformer-encoder block from the SigLIP-derived understanding tower. |
| Generation-private | `model.diffusion_head_a[0]` | 85,999,744 | One block from the generation diffusion head. |

Everything else (tokenizer, VAE, remaining 25 `und_trans` layers, remaining 9 `diffusion_head_a`
blocks, the Qwen2.5 LLM backbone, all embeddings/heads not listed above) is frozen
(`requires_grad=False`) for the entire diagnostic — no full-backbone unroll, consistent with the
resource envelope.

**Correction candidate discovered during this measurement** (recorded here, not acted on outside
this task's scope): `model.und_trans.layers` is a live `ModuleList` of **26** entries, not 8 as
narratively stated in T210's accepted `reports/T210/parameter-block-registry.md` ("8-layer
transformer encoder ... matches config's `num_und_trans_layers=8`"). `26 × 15,239,504 ≈
396,227,104`, which reconciles with that same report's machine-readable
`understanding_private` total of `397,141,792` in
`configs/admission/showo2/parameter-block-registry.yaml` (SigLIP-so400m natively ships 27 encoder
layers; the model code executes `del self.und_trans.layers[-1]`, leaving 26). The
machine-readable YAML total was and remains numerically correct; only the prose "8-layer"
description in T210's report appears to be a narrative error. `reports/T210/parameter-block-registry.md`
is outside this task's `allowed_paths` (T210 is `accepted`/closed), so this task does not edit it —
this note is the complete record of the discrepancy from T215's side. T215 refers to the
understanding-private block as index 0 of the 26-layer live list, not an 8-layer list.

## 2. State inventory (snapshot/restore targets)

For each of the 3 subspace parameter tensors (`fusion_proj`, `und_trans.layers[0]`,
`diffusion_head_a[0]`):

- **Parameters**: raw `.data` tensors (fp32, as loaded).
- **Gradients**: `.grad` tensors (zeroed/`None`-checked before and after every diagnostic pass).
- **Optimizer state** (AdamW, `torch==2.5.1` semantics): per-parameter `state[p] = {'step':
  tensor, 'exp_avg': tensor_like(p), 'exp_avg_sq': tensor_like(p)}`. A fresh `AdamW` instance is
  constructed over exactly the 3 subspace tensors (all other parameters excluded from the
  optimizer entirely, matching "no persistent parameter update" outside the declared subspace).
- **Counters**: the optimizer's per-parameter `step` tensor (torch 2.5.1 stores `step` as a
  0-dim tensor per parameter, not a shared Python int).
- **RNG state**: `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()` (one entry per visible
  GPU), Python `random.getstate()`.
- **Data-order state**: a fixed, explicit index list per batch (see §3) — no shuffling sampler is
  used, so "restore" here means re-asserting the same fixed index list, not a stateful sampler
  cursor.

Snapshot/restore is implemented as deep-copy-on-CPU (`.clone().cpu()` for tensors,
`copy.deepcopy` for RNG state objects) taken immediately before every diagnostic transition and
reasserted immediately after, per parameter/optimizer-state/RNG/data-order target above. The
correctness check compares every restored tensor against its pre-transition snapshot
(bitwise/dtype-tolerance per §5) before any subsequent transition is attempted.

## 3. Batch and seed manifest (disjoint adaptation/meta split)

Per the protocol's requirement to use disjoint adaptation/meta batches in the main smoke (same-batch
eval only as a named diagnostic), both task paths reuse official Show-o2 demo assets already present
in the pinned repo clone — no new asset acquisition required:

| Path | Role | Asset | Source |
|---|---|---|---|
| Understanding | Meta (held-out eval) | `docs/mmu/pexels-jane-pham-727419-1571673.jpg` + question `"Describe the image in detail."` | Same asset T210 used in its accepted MMU smoke — reused here for continuity/cross-check against T210's recorded output, not for adaptation. |
| Understanding | Adaptation | `docs/mmu/pexels-mccutcheon-1148998.jpg` + question `"Describe the image in detail."` | Disjoint official demo asset, not used by T210. |
| Generation | Meta (held-out eval) | Prompt `"A red bicycle leaning against a brick wall, photorealistic, natural daylight."` | Same prompt T210 used in its accepted T2I smoke. |
| Generation | Adaptation | First line of `prompts/t2i_prompts.txt` (official candidate prompt, portrait subject) | Disjoint official prompt, not used by T210. |

**Named same-batch diagnostic** (explicitly labeled as such, not the main smoke): one additional
run per path reuses the meta batch as its own adaptation batch, to characterize how much of the
commit/rerun signal is attributable to genuine held-out generalization vs. same-batch overfit —
reported separately, never substituted for the disjoint-batch result.

**Seeds** (fixed, recorded in the run manifest):

| Purpose | Seed |
|---|---|
| Adaptation-step stochasticity (dropout/sampling noise during inner adaptation, if any) | 1215 |
| Meta-batch evaluation stochasticity | 2215 |
| Finite-difference Rademacher direction sampling | 42, 43, 44 (3 random directions) |

Generation-path diagnostic passes use `num_inference_steps=1` (distinct from T210's admission-smoke
value of 10) — a speed-only reduction to keep per-pass cost minimal for the diagnostic; this is not
a claim about generation quality and is recorded as such in `notes.md`.

## 4. Rerun/commit-response pseudocode

Notation: `theta_s` = `fusion_proj` params (shared, differentiated in the outer loss); `theta_p` =
`und_trans.layers[0]` or `diffusion_head_a[0]` params (private, adapted). All other parameters are
frozen for the entire diagnostic.

```python
def private_forward_and_loss(model, theta_s, theta_p, batch, task):
    # unmodified official forward()/t2i_generate() call path; only theta_s/theta_p require grad
    out = model(batch, ...)  # task in {mmu, t2i}
    return official_loss_fn(out, batch.targets)

def inner_adamw_step(theta_p, grad_p, opt_state, lr, differentiable):
    # exact AdamW update rule; differentiable=True keeps the moment-buffer recurrence in the graph
    ...
    return theta_p_new, opt_state_new

# RAW: baseline sensitivity, no private adaptation
L_meta_raw = private_forward_and_loss(model, theta_s, theta_p_0, meta_batch, task)
raw_grad = torch.autograd.grad(L_meta_raw, theta_s)

# COMMIT-RESPONSE (stop-gradient): adapt theta_p at the current theta_s, hold the result fixed
L_adapt = private_forward_and_loss(model, theta_s, theta_p_0, adapt_batch, task)
grad_p = torch.autograd.grad(L_adapt, theta_p_0, create_graph=False)
theta_p_1, opt_state_1 = inner_adamw_step(theta_p_0, grad_p, opt_state_0, lr, differentiable=False)
L_meta_commit = private_forward_and_loss(model, theta_s, theta_p_1.detach(), meta_batch, task)
commit_grad = torch.autograd.grad(L_meta_commit, theta_s)

# RERUN-RESPONSE (exact finite unroll): rerun the same private response at the candidate theta_s,
# differentiate through the complete selected state transition (no detach)
L_adapt_diff = private_forward_and_loss(model, theta_s, theta_p_0, adapt_batch, task)
grad_p_diff = torch.autograd.grad(L_adapt_diff, theta_p_0, create_graph=True)
theta_p_1_diff, opt_state_1_diff = inner_adamw_step(theta_p_0, grad_p_diff, opt_state_0, lr, differentiable=True)
L_meta_rerun = private_forward_and_loss(model, theta_s, theta_p_1_diff, meta_batch, task)
rerun_grad = torch.autograd.grad(L_meta_rerun, theta_s)  # backprops through the K=1 inner step

# PARAMETER-ONLY vs COMPLETE optimizer-state differentiation (stage 6):
#   parameter-only: opt_state_1_diff's exp_avg/exp_avg_sq buffers are detached before use in the
#     next step's update rule (moments treated as constants);
#   complete: the moment-buffer recurrence itself stays in the create_graph=True chain, so
#     rerun_grad also flows through how theta_s affected exp_avg/exp_avg_sq.
```

`K=3` repeats the adaptation/differentiation step 3 times (same `adapt_batch`, same seed) before
computing the meta loss — attempted only after the `K=1` gate in §5/§7 below passes.

## 5. Finite-difference design and tolerances

- **Directions**: 4 fixed unit vectors in `theta_s` (`fusion_proj`) space — 3 Rademacher-random
  (seeds 42/43/44) and 1 "natural" direction (`raw_grad` from §4, normalized), giving both random
  coverage and a gradient-aligned sanity direction.
- **Step size**: `eps = max(1e-3 * (||theta_s|| / sqrt(numel(theta_s))), 1e-6)` (scale-adaptive,
  standard per-element-magnitude convention, with a fixed floor).
- **Estimator**: central difference, `(L(theta_s + eps*v) - L(theta_s - eps*v)) / (2*eps)` for
  each direction `v`, using the *rerun-response* protocol (full inner adaptation rerun at each
  perturbed `theta_s`, non-differentiable/`create_graph=False` since only the scalar loss is
  needed) — compared against `v · rerun_grad` from §4, and separately against `v · commit_grad`
  for a labeled (expected-to-diverge) reference.
- **Gates** (verbatim from the frozen protocol): relative error ≤ 1e-3 when the FD reference
  magnitude exceeds 1e-8; absolute error ≤ 1e-6 near zero; restored tensors match pre-transition
  snapshots within dtype tolerance; optimizer counters/RNG/data-order match exactly after every
  rollback; OOM, a nondifferentiable transition, missing optimizer state, or an unstable FD
  estimate (e.g. large variance across the 2 sides of a direction) is recorded as a failure for
  that configuration, not silently retried with a different subspace/eps.

## 6. Expected memory and runtime

Baseline (T210 accepted, forward-only, `docs/plans` unrelated): MMU peak ~13.87GiB alloc / ~38.48GiB
reserved; T2I peak ~12.36GiB alloc / ~12.69GiB reserved (`reports/T210/r2-r5-ssd-rerun.md`).

- **Added optimizer/gradient memory**: gradient buffers for the 3 subspace tensors (~107.7M params,
  fp32) ≈ 431MB; AdamW moment buffers (`exp_avg`+`exp_avg_sq`) ≈ 862MB. Small relative to the
  multi-GiB baseline.
- **Added activation memory for backward**: the `create_graph=True` rerun-response chain retains
  the inner-step computation graph through the outer loss; expect peak reserved memory to increase
  by roughly 1.5-2x the forward-only figure for `K=1`, i.e. up to ~60-75GiB for the MMU path (fits
  a single 96GB H20) and up to ~25GiB for the T2I path (`num_inference_steps=1`, per §3).
- **GPU count**: default execution on GPU 0 alone (currently free on H20-FoldUMM). The second
  allowed GPU is reserved only as a contingency if `K=3`'s deeper unroll graph exceeds single-device
  memory — documented as a stop condition, not a default plan, per the "do not expand ... GPU count
  ... after observing diagnostic values" retry rule.
- **Runtime estimate**: per task instance (MMU-diagnostic, T2I-diagnostic), `K=1` requires roughly
  raw(1 fwd+bwd) + commit(2 passes) + rerun(2 passes, `create_graph=True`) + optimizer-state variant
  (roughly doubles commit/rerun compute) + FD (4 directions × 2 sides = 8 forward-only rerun-response
  evaluations) ≈ 20-30 forward-equivalent passes. At T210's measured ~10-30s/forward-equivalent
  (MMU heavier than T2I), this is ~5-15 minutes per task instance, ~10-30 minutes for both instances
  combined ≈ **0.2-0.5 GPU-hours** for `K=1`. `K=3` (conditional) adds roughly another
  10-20 minutes ≈ 0.2-0.3 GPU-hours. **Total estimate ≈ 0.4-0.8 GPU-hours**, comfortably inside the
  8-GPU-hour cap with large margin, and expected to use only 1 of the 2 allowed GPUs.

## 7. Required assets

No new downloads. Reuses in place:

- T210's local-SSD checkpoint/tokenizer/SigLIP/VAE/HF-cache at `/dockerdata/t210-showo2/`.
- The pinned official Show-o2 repo clone (commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23`).
- The two additional official demo assets identified in §3 (`docs/mmu/pexels-mccutcheon-1148998.jpg`,
  `prompts/t2i_prompts.txt` line 1), both already present in the pinned clone.

## 8. Exact commands

New adapter code lives under this task's own `allowed_paths`
(`src/comppareto/adapters/showo2/`, `configs/feasibility/showo2/`), importing the official Show-o2
model classes rather than modifying them:

```bash
cd /apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/Show-o/show-o2 && \
HF_HOME=/dockerdata/t210-showo2/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/apdcephfs_cq9/share_1447896/yihangli/workspace/UMM_RL/src:$PYTHONPATH \
/root/venvs/showo2/bin/python -m comppareto.adapters.showo2.run_feasibility \
  --config /apdcephfs_cq9/share_1447896/yihangli/workspace/UMM_RL/configs/feasibility/showo2/diagnostic-v1.yaml \
  --output-dir /apdcephfs_cq9/share_1447896/yihangli/workspace/UMM_RL/runs/feasibility-showo2-v1/
```

Storage preflight (run before the command above, per §9):

```bash
HF_HOME=/dockerdata/t210-showo2/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/model_storage_preflight.py \
  --path /dockerdata/t210-showo2 \
  --minimum-free-bytes 10000000000 \
  --output configs/feasibility/showo2/storage-preflight.json
```

## 9. Resource budget and stop conditions

- Cap: 2 H20 GPUs, 8 H20-equivalent GPU-hours (envelope in the task file). Estimated use (§6):
  ≈0.4-0.8 GPU-hours on 1 GPU — large margin retained for reruns/debugging.
- `K=1` runs and passes its full gate (§5) before `K=3` is attempted, per the frozen protocol.
- Stop conditions: any OOM, nondifferentiable transition, state-restoration mismatch, or unstable
  FD result is recorded as a failure for that configuration and the trainable subspace/response
  horizon/GPU count/budget is not expanded in response, per the failure-and-retry rules.
- This report's subspace and resource estimate remain inside the frozen protocol and the resource
  envelope; per the task file's "First report" section, execution may proceed once the storage
  preflight (§8) passes, without a separate approval round.
