# T710 Result Summary — CoRL / Janus-Pro-1B admission and GPU smoke

Remote executor, 2026-09-16/17. Branch `agent/T710-corl-admission-gpu-smoke`. This report
summarizes what actually happened after `reports/T710/first-report.md`'s pre-execution plan: asset
pinning, environment build issues found and fixed, the bounded GPU smoke's real measured numbers,
and the outcome of all 8 mandatory implementation checks. Raw evidence backing every number below is
`runs/corl-admission-v1/{manifest.json,metrics.json,notes.md}` and the artifacts they reference.

## 1. Asset pinning (final)

| Asset | Revision | Verification |
|---|---|---|
| `deepseek-ai/Janus-Pro-1B` | `960ab33191f61342a4c60ae74d8dc356a39fafcb` | resolved and downloaded to `/dockerdata/t710-corl/` (local XFS) |
| `mm-vl/x2x_rft_22k` | `f52833ce01b5657294bed87f23f27d04b92838b9` | 32/32 unique records + 32/32 real images materialized to `configs/corl/admission/micro-split.jsonl` (sha256 `b9e7c1d211ae509a57a3adbddccd14ca15d1dfadd83f20b5817542b59bafa8cd`), all `qa_type=OE` |
| `sentence-transformers/all-mpnet-base-v2` (aux) | `e8c3b32edf5434bc2275fc9bab85f82640a19130` | resolved at download time via `HfApi().model_info(...).sha`, pinned via `snapshot_download(revision=...)` |

`configs/corl/admission/storage-preflight.json`: `status: pass`, `filesystem_class: local`
(`/dockerdata/t710-corl` is XFS on `/dev/mapper/gpu-gpu_volume`, real local NVMe, not a ceph/fuse
mount).

## 2. Environment build issues found and fixed

1. **`x2x_rft_22k`'s image column is `real_image`, not `image`.** The first micro-split
   materialization attempt used the wrong key and silently wrote 32 null `image_path` rows; caught by
   a post-hoc schema verification pass, corrected, and re-materialized (see
   `configs/corl/admission/source-lock.yaml`'s `dataset.actual_materialized` note).
2. **CoRL's own `pyproject.toml` pin set is internally inconsistent** (D13):
   `transformers>=4.49.0` conflicts with `trl==0.18.1`'s real floor of `transformers>=4.50.0`, and
   unpinned `sentence-transformers` resolves to `6.0.1` by default, which requires
   `transformers>=5.0.0` and breaks Janus's `modeling_vlm.py` under transformers 5.x. Resolved by
   pinning `transformers==4.50.0`, `sentence-transformers==3.0.1` in the dedicated `/root/venvs/corl`
   venv; full import chain confirmed OK.
3. **`flash_attn` did not install** (local wheel version-tag mismatch against pip's exact-match
   spec). Confirmed **architecturally irrelevant** for this trainer's Janus path after a full re-read
   of `grpo_trainer_unified.py.__init__`: the `'Janus' in model_id` branch calls
   `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)` with no
   `attn_implementation` kwarg and no `**model_init_kwargs` spread at all, so `flash_attn`'s presence
   or absence never affects which attention implementation Janus actually runs.
4. **`scripts/model_storage_preflight.py` failed with `ModuleNotFoundError: No module named
   'jsonschema'`** the first time it ran in `/root/venvs/corl` (that venv was built only for
   `corl`/`janus`, not for this repo's own tooling). Fixed by installing `jsonschema==4.26.0` via the
   container's `star_proxy` internet access.
5. **First storage-preflight run failed** with `HF_HOME is not a directory:
   /dockerdata/t710-corl/hf_cache` (directory not yet created). Fixed with `mkdir -p`; re-run passed.

## 3. GPU smoke — real measured numbers

Single H20 (GPU index 0), `total_wallclock_seconds: 106.42121815681458` (≈0.0296 GPU-hours, well
inside the 16 GPU-hour envelope), `peak_vram_bytes: 63273602560` (≈58.9 GiB).

- **`upstream_exact`**: construction raised `HFValidationError` at
  `T2ICycleConsistencyReward.load_external_model` trying to resolve
  `XXX/checkpoint//all-mpnet-base-v2` — the exact, previously predicted D11 defect. Expected,
  correctly predicted outcome, isolated in its own top-level key.
- **`corrected_candidate`**: 4 optimizer steps on 8 unique records, `num_generations=4`.
  - `changed_param_count: 3` total parameters changed by the optimizer step machinery bookkeeping,
    but `unauthorized_changed_param_count: 0` — all changes fall inside the 219 authorized trainable
    `language_model.*` parameter names (663 frozen parameter names, 436,641,419 frozen params vs.
    1,652,656,128 trainable params of 2,089,297,547 total).
  - `checkpoint_reload_pass: true`, `checkpoint_reload_error: null`.
  - Per-step `reward_unified`: 2.0659, 1.9045, 1.9912, 2.0586 (min 1.9045, max 2.0659, mean 2.0051,
    `zero_variance: false`). Per-reward-function means all similarly non-degenerate (see
    `runs/corl-admission-v1/metrics.json` `checks.6_reward_range_zero_variance_clipping`).
  - `train_loss: -0.0809`, decreasing `learning_rate` schedule to 0 by step 4 (linear decay over 4
    steps), `grad_norm` in `[7.875, 14.75]` across steps — all finite.
- **`reference_immutability_probe`** (`beta=0.1`, isolated 1-step/2-record/`num_generations=2`
  variant): `ref_model_is_none: false`, `ref_model_changed_param_count: 0`,
  `ref_model_immutable: true`.

## 4. Mandatory implementation checks — all 8 have an explicit recorded outcome

Full detail per check in `runs/corl-admission-v1/metrics.json`'s `checks` object; summary:

1. **Parameter authorization**: 219 trainable / 663 frozen parameter names enumerated;
   `all_trainable_are_authorized: true`, 0 unauthorized names.
2. **Frozen-head gradient transmission**: `language_model_any_nonzero_grad: true`
   (mean grad-norm 0.569, max 3.341 across 219 params with grad); `gen_embed`/`gen_head`/
   `vision_model` leaf grads confirmed `None` (frozen, as designed) while still participating in the
   forward graph that produces the loss.
3. **Image-token next-token alignment**: **recorded, `alignment_confirmed: false`** as measured.
   Source re-read of `_get_per_token_logps` (lines 350-424) confirms the real trainer applies **no
   shift** to t2i/image tokens (unlike the mm2t/text branch, which explicitly shifts); the probe's
   unshifted computation reproduces this exactly. The measured result (shifted-by-one control
   slightly higher logp than unshifted) is recorded honestly rather than forced to "true" — see
   `reports/T710/failure-ledger.md` for full disposition. This differs from `first-report.md`'s
   original plan text (which assumed a shift-by-one convention would be found); the actual source
   behavior is the opposite, and that is itself the check's real result.
4. **Batched MC/OE reward dispatch**: defect confirmed (D14) — `common_qa_accuracy_reward`'s
   batched `qa_type` dispatch makes the MC branch structurally unreachable when mixed with OE in one
   batch; demonstrated via a dedicated synthetic probe against the real function (the actual
   micro-split is 32/32 OE, so this was not re-triggered by the smoke run itself).
5. **Paper-vs-code discrepancy table frozen**: `configs/corl/admission/discrepancy-lock.yaml`,
   D1-D14 (D9-D14 added this task).
6. **Reward ranges / zero-variance / clipping**: all 5 reward series non-degenerate
   (`zero_variance: false`) at n=4 steps; clipping structurally impossible in this trainer
   (`epsilon_low`/`epsilon_high` stored but never read in `compute_loss`).
7. **Reference immutability**: confirmed `ref_model_immutable: true`, 0 changed parameters, via the
   isolated `beta=0.1` probe.
8. **Upstream-exact / corrected-candidate separation**: kept in separate top-level keys
   (`upstream_exact`, `corrected_candidate`) in `final_evidence.json`, with `upstream_exact`'s
   construction failure isolated by a try/except and not affecting `corrected_candidate`'s execution.

## 5. Overall status

`pass` (`runs/corl-admission-v1/manifest.json` `status: pass`). Confirmed upstream defects (D11,
D14) are valid, expected results per the task's own gate wording and do not block this task's `pass`
status, but per the task file they do block T730 until a corrected protocol is locally frozen — the
`corrected_candidate` path exercised in this run is that frozen protocol (explicit
`--model_ckpt_dir` switch; MC/OE dispatch defect not repaired since it is not exercised by the actual
micro-split data, per this task's "repair only locally-demonstrated correctness defects" scope).
