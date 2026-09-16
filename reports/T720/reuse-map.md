# T720 reuse-map — dataloader / checkpoint / optimizer / reward / rollout modules vs. CoRL needs

This maps the concrete, working modules T720 built and exercised against what
T730 (official CoRL Unified-RL exploratory reproduction) and T740 (CoRL
single-task GRPO oracles) will need once T710 unblocks them. Both T730 and
T740 are still `status: planned` at the time this report is written (they
depend on T710, not on T720) — nothing here is imported by CoRL code yet;
this is a forward-looking audit, and every claim below is scoped to "this
module could be lifted/adapted", never "this module has been proven to work
against the actual CoRL/Janus-Pro-1B code path", which T720 never touched.

## 1. Directly reusable, model-architecture-independent modules

| Module | What it does | Reuse fit for CoRL (T730/T740) |
|---|---|---|
| `src/comppareto/adapters/janus_pro_r1/param_inventory.py` | Pure, torch-free `classify_parameters()`: splits any `(name, parameter)` iterable into trainable/frozen buckets by dotted-name prefix, plus a `requires_grad_mismatches` cross-check. Operates on a `Protocol` (`.requires_grad`, `.numel()`), not a concrete class. | Directly reusable as-is. CoRL/Janus-Pro-1B's GRPO trainer will have its own trainable-prefix set (likely different names than Janus-Pro-R1's `language_model`/`gen_embed`/`gen_head`/`gen_aligner`/`aligner`), but the classification *function* takes `trainable_prefixes` as a parameter — CoRL only needs to supply its own prefix tuple, not reimplement the split-and-count-and-cross-check logic. This directly answers T700's "trainable parameter ownership" concern (see T700 objective) with a shared, unit-testable primitive. |
| `src/comppareto/adapters/janus_pro_r1/grpo_math.py` | Pure-numpy `group_relative_advantage()` (per-group standardized advantage, `eps`-guarded against a zero-std degenerate group) and `policy_gradient_loss()` (`-mean(log_probs * advantages)`), independent of torch. | Directly reusable for T740's single-task GRPO oracles and T730's Unified-RL reproduction *if and only if* CoRL's official Unified-RL trainer uses the same group-relative-advantage GRPO formulation (this needs to be confirmed against the actual CoRL/Unified-RL source once T710 lands it — not yet checked, since T710 was not read as part of T720's scope). If confirmed compatible, this removes one whole reimplementation-and-cross-check burden from T730/T740. |
| `src/comppareto/adapters/janus_pro_r1/env.py`'s pattern (not the module itself — its paths are Janus-Pro-R1-specific) | Env-var-driven path resolution with local-SSD-under-`/dockerdata` defaults, a `require_cuda()` guard that fails loudly rather than silently falling back to CPU, and a documented `run_output_root()` / git-tracked-evidence split (large checkpoints stay off git, only small hashed summaries are copied into the repo). | The *pattern*, not the code, is what transfers: T730/T740 should each write their own `env.py` following this exact template (one function per resolved path, every default overridable, one `require_cuda()` guard reused verbatim or copied). |

## 2. Reusable with adaptation (architecture-coupled but same interface shape)

| Module | What it does | Reuse fit for CoRL |
|---|---|---|
| `src/comppareto/adapters/janus_pro_r1/sft_smoke.py`'s probe-parameter-selection fix | Both smokes originally picked `next(iter(model.named_parameters()))` to snapshot a before/after value for the "authorized parameter change" check — for `MultiModalityCausalLM` this lands on `vision_model.vision_tower.pos_embed`, which is on the *understanding* branch and is structurally unreachable from the T2I-only forward path this smoke exercises, producing a false-negative ("param unchanged") despite real learning happening elsewhere. Fixed by restricting probe selection to a parameter on the actually-exercised forward path (`gen_head`/`gen_embed`/`gen_aligner`/`language_model` prefixes). | This exact bug class (probe-parameter selection landing on an architecturally-unreachable tensor for the branch under test) is a strong prior risk for any CoRL smoke on a similarly multi-branch (understanding+generation) architecture like Janus-Pro-1B. T730/T740 should budget review time specifically for this, not assume `next(iter(named_parameters()))` is a safe probe choice. |
| `src/comppareto/adapters/janus_pro_r1/grpo_smoke.py`'s `_per_token_logps_visual_cfg` | Standalone re-implementation of `GRPOTrainer._get_per_token_logps(..., addcfg=True, visual=True)` (from `grpo_trainer_t2iv1.py` lines 358-398), extracted so it can be called without instantiating the full `trl.GRPOTrainer` subclass (which needs a full `accelerate` process group + dataloaders + reference model). | If CoRL's Unified-RL trainer is *also* a `trl.GRPOTrainer` subclass with the same multi-process-launch overhead, this "extract the per-token-logprob math into a standalone callable" technique is directly transferable as an approach, even though the concrete CFG-doubling/image-token math is Janus-Pro-R1-specific and would need re-deriving against whatever CoRL's own visual log-prob computation looks like. |
| Checkpoint save/reload pattern (`torch.save(model.state_dict(), ...)` + `load_state_dict(..., strict=True)` + before/after value equality check on the same probe parameter) used identically in both `sft_smoke.py` and `grpo_smoke.py` | Produces the `reload_ok` / `n_missing` / `n_unexpected` / `reload_matches_trained_values` fields both smokes report. | Directly reusable as a checklist/pattern for T730/T740's own checkpoint-reload smoke test, regardless of model architecture — `strict=True` + a specific-tensor equality check is architecture-agnostic. |

## 3. NOT reusable (Janus-Pro-R1-specific, no CoRL analog claimed)

- The vendored rollout call itself (`JanusLLamaModel.generate_with_refine`, `task_list=[1]`, CFG-guided VQ-token autoregressive sampling) is Janus-Pro-R1's own generation architecture (LlamaGen VQ-16 tokenizer + `gen_head`/`gen_embed`/`gen_aligner`) and has no reason to resemble whatever rollout Janus-Pro-1B/CoRL's Unified-RL path uses — no reuse claimed.
- `T720InternVLReward` (the reward-model wrapper) is a two-line subclass of upstream's `InternVLReward` that only parameterizes the checkpoint path; CoRL's own reward setup is unknown at time of writing (T710 not read as part of T720) and no compatibility claim is made.
- The dataloader path exercised by SFT's smoke (`vendor/janus-pro-r1/janus-sft/data/t2i_examples/labels/*.jsonl`, one JSON object per line with `{promptid, prompt, data:[...]}`) is Janus-Pro-R1's own released format; CoRL's data format was not inspected as part of this task and no reuse claim is made without that comparison.

## 4. Scope boundary on this comparison

This reuse-map is necessarily one-sided: it lists what *T720 built* and
assesses plausible fit against T730/T740's *stated objectives* (T700's child
sequence and the two tasks' own one-paragraph objective statements — the only
CoRL-side material this task's `allowed_paths` and time budget permitted
reading). It does not read T710's own deliverables, CoRL's actual trainer
source, or Janus-Pro-1B's actual architecture, since none of those are inside
T720's `allowed_paths` (`vendor/janus-pro-r1/`,
`src/comppareto/adapters/janus_pro_r1/`, etc.) and doing so would have
exceeded this task's scope. Any of the "reusable with adaptation" or "pure,
architecture-independent" claims above should be re-verified by whoever
implements T730/T740 against the actual CoRL source before being relied upon.
