# T720 failure ledger

Every failure encountered during T720 execution, in chronological order, with
the exact error, root cause, fix, and verification. Entries are append-only
evidence, not a narrative summary — see `reports/T720/claim-check.md` for the
consolidated open-items/deviations view and `reports/T720/result-summary.md`
for the final pass/fail outcome.

## [FIXED] Vendored `generate_with_refine` crashes with `task_list=[1]`

- **Where**: `vendor/janus-pro-r1/janus-rl/src/open_r1/llama.py`, `JanusLLamaModel.generate_with_refine`.
- **Error**: `AttributeError: 'list' object has no attribute 'squeeze'`.
- **Root cause**: the function's return statement unconditionally calls `selfcheck.squeeze()`, but `selfcheck` is only ever reassigned from its initial `[]` inside the `task_list[-1] >= 2` branch (stages 2/3, self-check/regeneration). This smoke's documented, bounded rollout uses `task_list=[1]` (stage 1 only), so `selfcheck` is always still the initial `[]` at the return statement.
- **Fix**: one-line guard, `selfcheck.squeeze() if torch.is_tensor(selfcheck) else selfcheck`.
- **Verification**: GRPO smoke ran 4 steps end-to-end after the fix, all 4 losses finite (see `metrics.json` `grpo.steps`), no further crash.

## [FIXED] SFT smoke probe-parameter selection false-negative

- **Where**: `src/comppareto/adapters/janus_pro_r1/sft_smoke.py`.
- **Symptom**: `next(iter(model.named_parameters()))` selects `vision_model.vision_tower.pos_embed` — structurally unreachable from the T2I-only SFT forward path (`language_model`/`gen_embed`/`gen_head`/`gen_aligner`/`aligner` trainable prefixes per upstream's own `train_setup()`), so the before/after value check always reports "unchanged" regardless of whether real learning happened.
- **Root cause**: probe-parameter selection did not account for `MultiModalityCausalLM`'s multi-branch (understanding + generation) architecture — the literal first parameter in module-registration order is on the wrong branch.
- **Fix**: restrict probe selection to a parameter whose dotted name starts with one of the trainable prefixes actually exercised by this smoke's forward path.
- **Verification**: after the fix, `authorized_param_change.trainable_changed == true` with `trainable_probe_grad_norms_per_step = [0.0756, 0.0743, 0.0716, 0.0871]` (all nonzero, all 4 steps) — real evidence, not assumed. `frozen_param_name = "vision_model.vision_tower.pos_embed"` (the original, wrong probe choice) is separately confirmed `frozen_unchanged: true`, i.e. correctly frozen, not a masked bug.

## [FIXED] GRPO smoke probe-parameter selection false-negative (same bug class)

- **Where**: `src/comppareto/adapters/janus_pro_r1/grpo_smoke.py`.
- **Symptom/root cause**: identical to the SFT case above.
- **Fix**: identical technique, restricted to `gen_head`/`gen_embed`/`gen_aligner`/`language_model` prefixes (GRPO's full-parameter fine-tune has a slightly different exercised-prefix set than SFT's, since GRPO has no separate frozen reference model).
- **Verification**: `authorized_param_change.changed == true` with `probe_grad_norms_per_step = [10.27, 11.13, 11.39, 7.92]` (all nonzero, all 4 steps).

## [FIXED] `jsonschema` missing from the SFT venv

- **Where**: running `scripts/model_storage_preflight.py` inside the container via the pinned SFT venv.
- **Error**: `ModuleNotFoundError: No module named 'jsonschema'`.
- **Root cause**: `model_storage_preflight.py` imports `comppareto`, which transitively imports `comppareto.repo_state.__init__` -> `.tasks` -> `from jsonschema import Draft202012Validator`. `jsonschema` is not a Janus-Pro-R1 dependency at all — purely a transitive cost of importing the `comppareto` package from inside a minimally-pinned venv.
- **Fix**: `pip install jsonschema` into `/dockerdata/t720-janus-pro-r1/venvs/sft` (resolved `jsonschema==4.26.0` plus `jsonschema-specifications`/`referencing`/`rpds-py`; `attrs`/`typing-extensions` already present).
- **Verification**: re-ran `model_storage_preflight.py` -> clean `status: "pass"` (`configs/janus-pro-r1/admission/storage-preflight.json`).

## [FIXED] `deepspeed` missing from the original SFT environment plan

- **Where**: SFT venv build, first attempt at running `sft_smoke.py`.
- **Root cause**: `vendor/janus-pro-r1/janus-sft/trainer/utils/__init__.py` unconditionally imports `deepspeed` at package-init time (only used for an unused CLI-arg extension on the path this smoke exercises, but still required to be *importable*). The original plan (`reports/T720/first-report.md` section 2) assumed it was skippable.
- **Fix**: pinned `deepspeed==0.15.4` (upstream's own RL-side pin) — the latest deepspeed release at fetch time was incompatible with `torch==2.6.0`.
- **Verification**: SFT venv import of `trainer.utils` succeeded; SFT smoke ran end-to-end.

## [FIXED] `flash_attn` missing from the original RL environment plan

- **Where**: RL venv build, first attempt at loading the policy model.
- **Root cause**: `vendor/janus-pro-r1/janus-rl/src/open_r1/models/modeling_vlm.py` hardcodes `flash_attention_2` unconditionally for the policy model. The original plan incorrectly generalized InternVL's own guarded flash_attn fallback (which is real and correct in isolation, for InternVL only) to the whole RL venv.
- **Fix**: installed a prebuilt local `flash_attn` wheel matching `torch==2.6.0`'s ABI.
- **Verification**: policy model load succeeded (`grpo-smoke` summary `policy_load_seconds: 5.725`), rollout ran.

## [FIXED] `peft` missing from the original RL environment plan

- **Where**: RL venv build, first attempt at loading the InternVL reward model.
- **Root cause**: `internvl_img.py`'s import of `InternVLChatModel` transitively hits `modeling_internvl_chat.py`'s unconditional `from peft import ...` (never actually applied by `InternVLReward.evaluate()`, but required to be importable).
- **Fix**: pinned `peft==0.10.0`.
- **Verification**: reward model load succeeded (`grpo-smoke` summary `reward_load_seconds: 2.984`).

## [FIXED] InternVL2_5-8B `trust_remote_code=True` custom `.py` files missing after initial download

- **Where**: `/dockerdata/t720-janus-pro-r1/models/InternVL2_5-8B/`, first load attempt.
- **Root cause**: the initial direct `curl`-based download fetched only the 4 safetensors shards + tokenizer files, missing the custom modeling/configuration `.py` files that `AutoModel`/`AutoTokenizer`'s `trust_remote_code=True` path requires to be present alongside the weights.
- **Fix**: fetched the missing `.py` files from the same pinned HF revision; independently confirmed the pre-existing CQ7-canonical copy of InternVL2_5-8B already had these files present (unrelated provenance — likely downloaded via a tool that includes them by default).
- **Verification**: `T720InternVLReward.__init__` (subclass of upstream's `InternVLReward`, only the checkpoint path parameterized) loaded successfully with `trust_remote_code=True`; reward scores produced for all 4 GRPO steps (`metrics.json` `grpo.steps[*].rewards`).

## [FIXED] `numpy==2.3.1` (upstream's RL pin) has no wheel for this venv

- **Where**: RL venv build.
- **Error**: pip could not find a `numpy==2.3.1` wheel for the venv's Python 3.10 / manylinux target.
- **Root cause**: `numpy>=2.3.0` requires Python>=3.11; this venv is Python 3.10.
- **Fix**: pinned `numpy==2.2.6`, the version `torch`/`torchvision` themselves already resolve to transitively in this venv.
- **Verification**: RL venv installs cleanly; GRPO smoke's own numpy-vs-torch loss cross-check (`numpy_vs_torch_loss_abs_diff`, all 4 steps `< 1.1e-05`) confirms numpy-side math is numerically consistent with the torch-side computation despite the version deviation from upstream's pin.

## [FIXED] `scipy` missing from both venvs (unrelated to Janus-Pro-R1 itself)

- **Where**: first `python -m comppareto.adapters.janus_pro_r1.{sft_smoke,grpo_smoke}` invocation in each venv.
- **Root cause**: invoking any `comppareto.adapters.janus_pro_r1.*` entry point first imports `comppareto/__init__.py` (outside this task's `allowed_paths`), which unconditionally imports `.quadratic`, which imports `scipy.optimize.minimize`.
- **Fix**: pinned `scipy==1.15.3` in both venvs.
- **Verification**: both smoke entry points import and run successfully.

## No unresolved [FAIL] entries

Every failure encountered during T720 execution was diagnosed and fixed; none
were worked around by silently reducing scope without documentation (see
`reports/T720/claim-check.md` section 2 for the scope reductions that *were*
deliberate, pre-documented design choices rather than failure workarounds).
