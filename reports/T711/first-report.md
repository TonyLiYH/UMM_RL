# T711 First Report — CoRL semantic correctness and fixed-anchor admission

Remote executor, 2026-09-28T13:30Z. Branch
`agent/T711-corl-semantic-correctness-admission`, HEAD at time of writing
`d7d1a1a` (task status: running, review-history entry from worktree entry and
fast-forward merge). This report is the mandatory pre-GPU gate. No GPU
compute beyond Group C (which requires no GPU) has executed. The patch
described below has been applied in the container but only the CPU-only
fixture tests have run so far.

## 1. Task scope

T711 repairs and semantically validates the six blocker categories identified
by T710's local hard review: causal image-token scoring (Group A), behavior-
policy scoring identity for T2I (Group B), batched MC/OE reward dispatch
(Group C), padding-invariant TIM (Group D), full-state resume integrity
(Group E), and fixed-anchor surrogate alignment (Group F). The contract
requires all six `checks.<name>.confirmed == true` in
`runs/corl-admission-v1/metrics.json`, with real measured numbers behind every
flag — no fabrication.

Resource envelope: <=2 H20 GPUs, <=16 GPU-hours, <=64 unique records, <=12
optimizer steps. GPU index 0 on H20-FoldUMM only; GPUs 2/3 are occupied by
T215/T216 and must not be touched; the `train2.py` placeholder on other GPUs
must not be killed.

## 2. Group C (QA reward dispatch) — completed pre-GPU

### D14 defect and fix

`common_qa_accuracy_reward` in the pinned CoRL checkout
(`/dockerdata/t710-corl/ULM-R1/corl/open_r1/rewards/r_base.py`) compared
`kwargs['qa_type']` (a Python list when the trainer passes the full batch's
qa_type field) against the string `'MC'` using `== 'MC'`. A list never equals
a string, so the MC branch was structurally unreachable whenever multiple
examples were batched together — including the real trainer's
`num_generations=2+` call site. This is discrepancy D14 in
`configs/corl/admission/discrepancy-lock.yaml`.

The fix normalizes `qa_type` to a per-completion list at the top of the
function, dispatches by index, and raises `ValueError` on length mismatches.
The fix has been applied in the container at the path above. The original
upstream file was backed up to
`/dockerdata/t710-corl/ULM-R1/corl/open_r1/rewards/r_base.py.t710-upstream-orig`
before modification. A unified diff of the fix is archived at
`vendor/corl/patches/0001-fix-per-example-qa-type-dispatch.patch`.

### Fixture test results (13/13 PASS, no GPU required)

All tests ran against the real, live patched `common_qa_accuracy_reward`
imported from `/dockerdata/t710-corl/ULM-R1/corl` inside the container, using
the `/root/venvs/corl` Python. The discriminating fixture for D14 regression
control uses `verbose_mc = "<think>Let's see, the object is
round.</think><answer>The answer is B</answer>"` with solution `"<answer>B</answer>"`:

| Fixture | Result | Value |
|---|---|---|
| singleton_mc_bare_choice | PASS | [1.0] |
| singleton_mc_verbose | PASS | [1.0] |
| singleton_oe_exact | PASS | [1.0] |
| singleton_oe_partial | PASS | r[0]=0.25 (0 < r < 1) |
| all_mc_batch_dispatch (G=4) | PASS | [1.0, 1.0, 0.0, 1.0] |
| mixed_g2_batch_dispatch | PASS | [1.0, 1.0, 0.0, 0.25] (MC/OE correctly separated) |
| list_singleton_mc | PASS | [1.0] |
| list_singleton_oe | PASS | [1.0] |
| string_qa_type_broadcast_mc | PASS | [1.0, 1.0] |
| string_qa_type_broadcast_oe | PASS | [1.0, 1.0] |
| permutation_equivariance | PASS | r[0]==r[2], r[1]==r[3] |
| length_mismatch_qa_type_rejected | PASS | ValueError raised |
| d14_defect_reproduced_on_unpatched_backup | PASS | upstream[0]=0.25, patched[0]=1.0 |

The D14 regression control confirms the discriminating behavior: the upstream
backup scores the verbose-MC completion as 0.25 (OE soft_jaccard fallback,
"The answer is B" vs "B"), while the patched version scores 1.0
(MCQAnswerExtractor extracts the letter "B" correctly). This is the live
numerical proof of both the defect and its fix.

## 3. Remaining groups — GPU execution plan

Groups A, B, D, E, F require the real `JanusProUnifiedGRPOTrainer` on GPU.

**Group A (image-token causal alignment)** — four sub-checks captured from
the first real `_get_per_token_logps` call during `trainer.train()`, plus
targeted extra forward passes. The driver script is at
`/tmp/t711_run_semantic.py` (written, not yet run). Sub-checks:

- A1: per-position current vs shift-by-1 token-id alignment (fraction of
  positions where correct-token log-prob exceeds shifted-token log-prob)
- A2: future-token perturbation invariance (flip token at `cut+5`, max
  abs logit diff at `cut` must be < 1e-3 — causal mask sanity)
- A3: prefix-only truncated forward vs full-sequence teacher-forcing parity
  at the same cut position (max abs diff < 1e-2)
- A4: FP32 reference vs BF16 deviation (language_model temporarily cast to
  float32 for one forward pass, then reverted)

**Group B (behavior/scoring identity)** — source-confirmed mismatch between
CFG-guided generation (`cfg_weight=5`, `temperature=1 # HACK` at
`grpo_trainer_unified.py:264-282`) and the single unguided conditional forward
pass used for scoring in `_get_per_token_logps` (`:349-390`). When
`num_iterations=1` (the default), `old_per_token_logps` is never computed at
all (`:507-509`), so the only probability ever attached to a T2I completion in
the loss is this unguided score. The B check records this source-confirmed
mismatch plus a CFG=1 baseline generation probe.

**Groups D, E, F** — padding-invariant TIM, full-state resume, and fixed-
anchor surrogate. These require additional targeted forward passes and a
checkpoint-resume cycle. The driver for these will be written after A+B results
are confirmed.

## 4. Execution parameters

- Records: 8 (of the 32-record micro-split at
  `configs/corl/admission/micro-split.jsonl`; within the <=64 envelope)
- Steps: 3 for A+B trainer.train(); 1 additional backward for group-D padding
  probe; up to 4 for E resume; Group F adds 1 extra forward pass
- Total optimizer steps: <=12 (within envelope)
- GPU: CUDA_VISIBLE_DEVICES=0, H20-FoldUMM, GPU index 0

## 5. Container execution method

```
cd /path/to/taiji_gpu_start && script -qec "taiji_client exec \
  -scfg 'configs/config_h20_foldumm.json' \
  'gpu_model_train_split_llm218029E40DDAA49' \
  '8b1d81c89f83d4d4019f93a9170a251d' \
  bash -c '<cmd>'" /dev/null 2>&1 | tr -d '\r'
```

## 6. Next steps after this commit

1. Commit and push this report.
2. Run Groups A+B via `/tmp/t711_run_semantic.py` on GPU 0.
3. Write Groups D, E, F driver based on A+B results.
4. Run Groups D, E, F.
5. Merge all results into `runs/corl-admission-v1/metrics.json` with the
   contract-required `checks.<name>.confirmed` keys.
6. Write `reports/T711/result-summary.md` and set status to `awaiting_review`.
