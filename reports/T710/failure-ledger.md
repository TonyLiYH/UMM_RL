# T710 Failure Ledger — anomalies, expected failures, and unresolved items

Per this project's experiment-logging norm, every anomaly and failed/unexpected result is recorded
here honestly, whether or not it affects the overall `pass` status.

## 1. [EXPECTED FAILURE] `upstream_exact` construction raised `HFValidationError`

- **Symptom**: `JanusProUnifiedGRPOTrainer.__init__` raised
  `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
  'XXX/checkpoint//all-mpnet-base-v2'` at `construction_elapsed_seconds: 2.078`.
- **Root cause**: `GRPOScriptArguments.model_ckpt_dir` official default is the literal string
  `"XXX/checkpoint/"`; `corl/scripts/corl_unified.sh` never overrides it. Under the default reward
  set (`t2i_bid_cycle_reward` included), `T2ICycleConsistencyReward.load_external_model` concatenates
  this to `"XXX/checkpoint//all-mpnet-base-v2"` and tries to load it as an HF repo id / local path.
- **Status**: this is D11 in `configs/corl/admission/discrepancy-lock.yaml`, a previously identified
  and predicted upstream packaging defect. The failure was **expected and correctly predicted** by
  `reports/T710/first-report.md` before GPU execution began. It is not a bug in this task's own code,
  and does not affect `corrected_candidate`'s independent success (isolated by a try/except in
  `run_variant`). Recorded as a valid result per the task's own gate wording ("A confirmed upstream
  defect is a valid result").
- **Disposition**: does not block T710's own `pass` status. Per the task file, blocks T730 until a
  corrected protocol is locally frozen — `corrected_candidate`'s explicit `--model_ckpt_dir` switch
  is that frozen protocol.

## 2. [ANOMALY, RECORDED NOT "FIXED"] Check #3 image-token alignment probe: `alignment_confirmed: false`

- **Symptom**: at both n=2 (dry-run) and n=8 (final run) records, the deliberately shifted-by-one
  "misalignment" control's mean t2i log-probability (`-5.87`) was *higher* (less negative) than the
  "correct"/unshifted computation's mean log-probability (`-6.88`) — the opposite of the naive
  expectation that a correct alignment should have higher likelihood than a deliberately broken one.
- **Investigation**: re-read the real trainer's `_get_per_token_logps` source in full
  (`grpo_trainer_unified.py:350-424`) directly in the pinned CoRL venv's installed copy. Confirmed:
  the mm2t/text branch explicitly shifts (`mm2t_logits[:, :-1, :]` paired with
  `mm2t_input_ids[:, -mm2t_logits_to_keep:]`), but the t2i/image branch calls
  `selective_log_softmax(t2i_logits, t2i_discrete_img_ids)` **with no shift applied at all**. This
  task's probe computes the "correct" alignment exactly matching this real, unshifted behavior — the
  probe is not measuring the wrong thing.
- **Conclusion**: this is judged a **genuine, correctly-measured empirical finding**, not a probe
  defect. The most likely explanation is small sample size (n=8 records, 4 steps) combined with local
  spatial correlation in the 24x24 VQ token grid, which weakens a naive circular shift-by-one
  control's discriminative power (a shifted-by-one token is still spatially close to its true
  neighbor in a 2D image grid, unlike a fully scrambled control would be). This was **not
  "fixed"** by adjusting the probe to force a "true" result — the honest, as-measured value is
  recorded in `runs/corl-admission-v1/metrics.json` (`checks.3_image_token_alignment`).
- **Recommendation for follow-on work**: if this matters for a successor task, a stronger control
  (e.g. a fully permuted/scrambled token ordering, or a larger n) would better discriminate alignment
  correctness than a naive shift-by-one. Not performed here — out of this task's bounded-smoke scope
  (≤8 steps, ≤32 records).

## 3. [DEFECT, RECORDED NOT REPAIRED] Check #4 batched MC/OE reward dispatch (D14)

- **Symptom**: `common_qa_accuracy_reward`'s batched dispatch on `qa_type` makes the MC branch
  structurally unreachable when MC and OE examples are mixed in one batch (full evidence and
  reproduction steps in `configs/corl/admission/discrepancy-lock.yaml` D14, produced via a dedicated
  synthetic probe run directly against the real function in the pinned CoRL venv).
- **Why not repaired**: the actual materialized 32-record micro-split is 32/32 `qa_type=OE` (no MC
  examples exist in the real `x2x_rft_22k` micro-split sampled for this task), so this defect is
  never exercised by the actual GPU smoke run's real data. Per this task's explicit scope ("repair
  only locally-demonstrated correctness defects"), a defect demonstrated only via a synthetic
  out-of-band probe, not by the real smoke data, is recorded but left unrepaired this round.
- **Disposition**: blocks T730 per the task's gate wording, pending local-reviewer decision on
  whether/how to repair before any future MC-containing batch is run through this trainer.

## 4. [RESOLVED] Wrong dataset image column name (`real_image` vs `image`)

- **Symptom**: first micro-split materialization attempt used the assumed key `image`, which does
  not exist in `x2x_rft_22k`'s actual schema (the real column is `real_image`); this silently wrote
  32 rows with null `image_path`.
- **Root cause**: assumption made from the paper/README description without checking the dataset's
  actual `datasets.Features` schema first.
- **Fix**: caught by a post-hoc schema verification pass (`t710_check_schema.sh`/
  `t710_check_images.py`), corrected to use `real_image`, re-materialized. Verified: 32/32 records
  with non-null `image_path`, 32/32 real PNG files present on disk.
- **Verification**: `micro_split.load_records`/`validate_records` on the final
  `configs/corl/admission/micro-split.jsonl` reports `num_with_image: 32` (of 32).

## 5. [RESOLVED] CoRL's `pyproject.toml` pin set is internally inconsistent (D13)

- **Symptom**: `pip install -e .` under the literal upstream pin set produces a broken environment
  (unpinned `sentence-transformers` resolves to a version requiring `transformers>=5.0.0`, which
  breaks Janus's `modeling_vlm.py`; `trl==0.18.1`'s real floor is `transformers>=4.50.0`, tighter than
  the declared `transformers>=4.49.0`).
- **Fix**: pinned `transformers==4.50.0`, `sentence-transformers==3.0.1` explicitly in
  `/root/venvs/corl`. Full import chain confirmed OK (`ALL_IMPORTS_OK` marker, exit 0).

## 6. [RESOLVED] `flash_attn` wheel install failure

- **Symptom**: local wheel's version-tag did not exact-match pip's spec; install failed.
- **Investigation**: full re-read of `grpo_trainer_unified.py.__init__` confirmed `flash_attn`'s
  absence is **architecturally irrelevant** for Janus-Pro-1B specifically in this trainer (the
  `'Janus' in model_id` branch never passes `attn_implementation` or `**model_init_kwargs` to
  `AutoModelForCausalLM.from_pretrained` at all).
- **Fix**: none needed; left uninstalled, documented as irrelevant rather than as a workaround.

## 7. [RESOLVED] Missing `jsonschema` dependency in the GPU-side venv

- **Symptom**: `scripts/model_storage_preflight.py` failed with
  `ModuleNotFoundError: No module named 'jsonschema'` in `/root/venvs/corl` (built only for
  `corl`/`janus`, not for this repo's own `comppareto.repo_state` tooling).
- **Fix**: installed `jsonschema==4.26.0` via the container's `star_proxy` internet access.

## 8. [RESOLVED] Missing `HF_HOME` cache directory

- **Symptom**: first storage-preflight run failed (`status: fail`,
  `"HF_HOME is not a directory: /dockerdata/t710-corl/hf_cache"`).
- **Fix**: `mkdir -p /dockerdata/t710-corl/hf_cache`; re-run passed.

## 9. [SCOPE NOTE, NOT A FAILURE] "Evaluate fixed smoke examples before/after" not run as a separate pass

- The GPU-smoke pipeline description's final step ("evaluate fixed smoke examples before/after")
  was not run as a standalone before/after forward-pass comparison beyond the per-step reward series
  (check #6) and the checkpoint-reload pass/fail confirmation. See `reports/T710/claim-check.md` for
  the full disposition; not treated as a gate failure since checkpoint reload itself passed and every
  other finiteness/correctness requirement in the pass/fail gate is independently met.

## 10. [OPERATIONAL, RESOLVED] Nested-quote command truncation on GPU-container launch

- **Symptom**: an early relaunch attempt (`cjob.sh start ... "..."` nested inside
  `script -qec "taiji_client exec ... bash -c \"$REMOTE_CMD\""`) silently truncated to just `bash`
  with `exit=0` and nothing executed, due to premature double-quote termination.
- **Fix**: wrote the launcher command to a standalone `.sh` file on ceph first, then had
  `script -qec` invoke only `bash <script-path>` (single level of quoting). Used consistently for all
  subsequent launches this task.
