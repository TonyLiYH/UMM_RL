# T720 claim-check — open items, deviations, and scope reductions

Every item below is a fact about what this task actually did or found, kept
separate from `reports/T720/result-summary.md` so a reviewer can audit each
claim independently. Nothing here is resolved by simply writing it down —
several are genuinely open (licensing) and are reported as such rather than
assumed away.

## 1. Open licensing items (upstream facts, not fixable by T720)

- **Janus-Pro-R1 code (`wendell0218/Janus-Pro-R1` @ `0e40b3aa291cb15770f69affc956602c217490af`) has no LICENSE file.** Checked `LICENSE`/`LICENSE.md`/`LICENSE.txt` at repo root and under `janus-sft/`, `janus-rl/` — all return 404. Under GitHub's terms of service, the default is all-rights-reserved (no license grant beyond viewing the source). This is vendored into `vendor/janus-pro-r1/` for the purpose of this admission/smoke task only.
- **`midbee/Janus-Pro-R1-Data` (the released HF dataset) has no license tag** on its HF dataset card. One shard (`train-0000-of-0524.parquet`, 875,432,539 bytes) was downloaded and hash-verified for the admission data-loader check only.

These are recorded, not resolved — they are facts about the upstream
project's own licensing hygiene, outside this task's ability to fix.

## 2. Documented scope reductions (not silent omissions)

- **GRPO smoke has no frozen-reference-model KL term and no PPO-style ratio clip.** `grpo_math.py`'s module docstring explains why: with exactly one optimizer step per rollout batch (matching upstream's own per-step generate-then-train loop), the probability ratio `pi_theta/pi_theta_old` is exactly 1 by construction on the step it is computed on the same policy — there is nothing for a clip to clip. Omitting the KL term trades off the ~14.84GB a second co-resident bf16 7B reference-model copy would cost against the smoke's memory budget (see section 4 for actual observed peak memory).
- **Rollout scope reduced to `task_list=[1]`** (only the initial generation stage of upstream's 3-stage `generate_with_refine`, skipping stages 2/3 — self-check reflection and conditional regeneration). Documented in `grpo_smoke.py`'s module docstring.
- **Image resolution reduced**: `image_token_num_per_image=64`/`img_size=128` instead of upstream's `576`/`384` — 9x fewer autoregressive decode steps per rollout image, same real sampling code path.
- **Full-parameter GRPO fine-tune** (all of `model.parameters()` get `requires_grad=True`, no freeze), matching upstream's own `grpo_trainer_t2iv1.py`, confirmed by grep: upstream only ever sets `requires_grad=False` on the separate frozen `ref_model` copy, never on the policy model itself. T720's smoke has no reference model, so nothing is frozen — this is not a T720 simplification, it is upstream's own design faithfully reproduced with the KL/ref-model piece cut (see above).
- **One real 1/524 data shard downloaded (875MB) but the GPU SFT smoke trains on the smaller in-repo `t2i_examples` (1 prompt, 8 images).** `promptid 15` (the in-repo example) is confirmed present in the downloaded full-dataset shard, i.e. the in-repo bounded examples are a genuine subset of the same released dataset, not a separate synthetic fixture — verified by direct lookup in the downloaded parquet shard.
- **GRPO smoke used only 1 unique bounded prompt** (of the 2 vendored prompts referenced in the module docstring) across all 4 steps — both smoke's prompt source files were re-checked and only 1 unique `prompt` field existed in the label files that fed the loader in this particular run (see `_load_bounded_prompts`'s dedup-by-value logic in `grpo_smoke.py`); no new prompts were invented, this is a fact about the vendored input data's actual duplication, not a smoke-script bug.

## 3. Environment corrections discovered only after execution (first-report.md's plan was wrong on these four points, corrected post-hoc)

- `deepspeed==0.15.4` turned out to be a real SFT install requirement (vendored `trainer/utils/__init__.py` imports it unconditionally at package-init time), not skipped as originally planned.
- `flash_attn` turned out to be a real RL install requirement (the policy model's vendored `modeling_vlm.py` hardcodes `flash_attention_2` unconditionally), not skipped as originally planned (the InternVL guarded-fallback claim was correct in isolation but wrongly generalized to the whole RL venv).
- `peft==0.10.0` is a real RL install requirement, not mentioned in the original plan (`internvl_img.py` -> `modeling_internvl_chat.py` unconditionally imports it, though `InternVLReward` never applies it).
- `scipy==1.15.3` is required in both venvs for a reason unrelated to Janus-Pro-R1: importing `comppareto.adapters.janus_pro_r1.*` first imports `comppareto/__init__.py` (outside this task's `allowed_paths`), which unconditionally imports `.quadratic`, which imports `scipy.optimize.minimize`.
- `numpy` is pinned to `2.2.6` in the RL venv, not upstream's `2.3.1`: no `numpy>=2.3.0` wheel exists for this cp310 venv (`numpy>=2.3.0` requires Python>=3.11); `2.2.6` is what `torch`/`torchvision` already resolve to transitively.

Full discovery narrative (exact error text, root cause, fix, and verification
for each) is in `reports/T720/failure-ledger.md` and
`configs/janus-pro-r1/admission/environment-{sft,rl}-lock.md`.

## 4. Bugs found and fixed in vendored/adapter code (real evidence, not assumed)

- **Vendored `generate_with_refine` crash**: `vendor/janus-pro-r1/janus-rl/src/open_r1/llama.py`'s `JanusLLamaModel.generate_with_refine` unconditionally calls `selfcheck.squeeze()` in its return statement, but `selfcheck` is only ever reassigned from its initial `[]` inside the `task_list[-1] >= 2` branch — calling with `task_list=[1]` (this smoke's documented, bounded stage-1-only rollout) crashes with `AttributeError: 'list' object has no attribute 'squeeze'`. Fixed with a one-line, clearly-commented guard: `selfcheck.squeeze() if torch.is_tensor(selfcheck) else selfcheck`.
- **SFT and GRPO smoke probe-parameter selection bug** (same bug class, found and fixed independently in each smoke): `next(iter(model.named_parameters()))` picks `vision_model.vision_tower.pos_embed` — the *first* parameter in module-registration order for `MultiModalityCausalLM`, which is on the understanding branch and structurally unreachable from the T2I-only generation forward path both smokes exercise. This produced a false-negative "parameter unchanged" result despite real learning happening on the exercised path. Fixed in both `sft_smoke.py` and `grpo_smoke.py` by restricting probe selection to a parameter whose dotted name starts with one of the forward-path prefixes actually exercised (SFT: `language_model`/`gen_embed`/`gen_head`/`gen_aligner`/`aligner`; GRPO: `gen_head`/`gen_embed`/`gen_aligner`/`language_model`).
- **`jsonschema` missing from the SFT venv**: `scripts/model_storage_preflight.py` imports `comppareto`, which transitively imports `comppareto.repo_state.tasks`, which does `from jsonschema import Draft202012Validator` — `jsonschema` is not a Janus-Pro-R1 dependency at all, just a transitive import cost of running any `comppareto`-importing script from inside the pinned venv. Fixed with `pip install jsonschema` (resolved `jsonschema==4.26.0`).
- **InternVL2_5-8B `trust_remote_code=True` custom `.py` files gap**: the initial `curl`-based direct-file download of the 4 safetensors shards + tokenizer did not include the model's custom modeling/configuration `.py` files that `trust_remote_code=True` requires at load time; fixed by fetching the missing files (see `environment-rl-lock.md` for the exact file list and verification).

## 5. Architectural note: `/dockerdata` is unreachable from the manifest-verification shell

All GPU execution runs inside the Taiji container against
`/dockerdata/t720-janus-pro-r1/...` (the verified local-SSD execution copy,
satisfying the frozen protocol's "execute from verified local SSD"
requirement). But `/dockerdata` is a container-exclusive mount, not reachable
from the plain CQ9 shell that runs
`scripts/validate_task_submission.sh`/`scripts/verify_manifest_artifacts.py`.
Since `verify_manifest_artifacts()` re-hashes every `manifest.json` artifact's
`canonical_uri` from wherever it is invoked, `runs/janus-pro-r1-stack-v1/manifest.json`'s
`artifacts` list only references CQ7-canonical or in-repo paths (the base
checkpoint and reward checkpoint both already had independent CQ7-canonical
copies, independently re-hashed this task and confirmed identical to the
container-execution copies; the data shard and both smoke `summary.json`
evidence files were copied out to CQ7/in-repo paths with hash verification).
The two large (14.06GB, 14.84GB) disposable smoke checkpoint `.pt` files live
only on `/dockerdata` and are intentionally *not* listed as manifest
artifacts — their sha256/bytes/reload-check facts are still real and are
reported in `runs/janus-pro-r1-stack-v1/metrics.json` (produced by the smoke
scripts themselves, inside the container, immediately after save/reload),
just not independently re-hashable from outside the container. This is a
documented scope decision, not a missing check.

## 6. `attrdict`/Python-3.10 shim

Both `janus-sft` and `janus-rl` vendored code do `from attrdict import
AttrDict`, which fails on Python>=3.10 because `attrdict` itself does `from
collections import Mapping` (moved to `collections.abc` in 3.10). Both
`sft_smoke.py` and `grpo_smoke.py` apply an identical, documented shim
(`_apply_attrdict_collections_shim()`) that back-fills the missing
`collections.*` aliases from `collections.abc` before importing any vendored
module that needs `attrdict`. This is a compatibility shim for running
upstream's own unmodified code on a newer Python, not a change to upstream's
logic.
