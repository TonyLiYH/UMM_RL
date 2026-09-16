# Run note — `corl-admission-v1`

Formal admission + GPU optimizer smoke run for T710 (CoRL / Janus-Pro-1B Unified-GRPO admission).
`manifest.json` in this directory is the schema-conformant record; this note explains what it
summarizes and where the full narrative lives.

## What this run covers

Three variants, all driven by `src/comppareto/adapters/corl/run_smoke.py` calling the real,
unmodified `JanusProUnifiedGRPOTrainer` (from the pinned CoRL/ULM-R1 commit
`0c92629f9b307a32bb286ae3562809e941d1bb0b`) directly, executed inside the H20-FoldUMM GPU container
on GPU0 (`CUDA_VISIBLE_DEVICES=0`):

1. **`upstream_exact`** — trainer constructed with the official script defaults, including the
   literal placeholder `--model_ckpt_dir XXX/checkpoint/`. Construction raised `HFValidationError`
   inside `T2ICycleConsistityReward.load_external_model` while resolving
   `XXX/checkpoint//all-mpnet-base-v2` as a tokenizer path — this is the exact, previously documented
   upstream defect (see `configs/corl/admission/source-lock.yaml`'s `auxiliary_models` note and
   `configs/corl/admission/discrepancy-lock.yaml`). This is an **expected, correctly predicted
   failure**, not an unplanned run failure; it is isolated in its own top-level key
   (`upstream_exact`) in `final_evidence.json` and does not affect `corrected_candidate`.
2. **`corrected_candidate`** — identical trainer/config, with a real local
   `--model_ckpt_dir` supplied (pointing at the locally pinned `all-mpnet-base-v2` copy) so
   `t2i_bid_cycle_reward`'s external BERTScore model loads successfully. Ran 4 optimizer steps on 8
   unique micro-split records with `num_generations=4`, then saved and reloaded a disposable
   checkpoint.
3. **`reference_immutability_probe`** — an isolated, second trainer instance with `beta=0.1` (KL
   term enabled, forcing `disable_dropout`/reference-model construction) run for 1 optimizer step on
   2 records with `num_generations=2`, to verify the frozen reference policy's parameters do not
   change during training.

## Where the full evidence and narrative live

- **First report** (pre-GPU-execution audit: upstream repo/license/paper cross-check, asset
  revisions, discrepancy table draft, resource estimate): `reports/T710/first-report.md`.
- **Cumulative result narrative** (GPU-hours accounting, environment build issues found and fixed,
  the 8 mandatory-check outcomes in prose): `reports/T710/result-summary.md`.
- **Claim-by-claim cross-check** against `tasks/T710-corl-admission-and-gpu-smoke.md`'s research
  claim/objective/frozen-protocol/pass-fail-gate/resource-envelope language:
  `reports/T710/claim-check.md`.
- **Failure/anomaly ledger** (the expected `upstream_exact` construction failure; the check #3
  alignment-probe's counter-intuitive `alignment_confirmed: false` result and why it is trusted
  rather than "fixed"; the environment build issues hit and resolved):
  `reports/T710/failure-ledger.md`.
- **Discrepancy table** (paper vs. public code, D1-D14, including this task's D9-D14 additions —
  D11 = the `model_ckpt_dir` placeholder defect exercised by `upstream_exact` above, D14 = the
  batched MC/OE reward-dispatch defect): `configs/corl/admission/discrepancy-lock.yaml`.
- **Storage preflight** (local-SSD confirmation for `/dockerdata/t710-corl`):
  `configs/corl/admission/storage-preflight.json`.
- **Remote artifact reverification**: `configs/corl/admission/artifact-verification.json`.
- **Environment freeze** (exact resolved package versions, the `flash_attn`/Janus-path
  architectural-irrelevance finding, the `jsonschema` gap fixed in-container):
  `configs/corl/admission/environment-lock.md`.
- **Materialized micro-split** (32 real `x2x_rft_22k` records + real image paths, all `qa_type=OE`):
  `configs/corl/admission/micro-split.jsonl` (sha256
  `b9e7c1d211ae509a57a3adbddccd14ca15d1dfadd83f20b5817542b59bafa8cd`), referenced as manifest
  artifact `micro-split-jsonl`.
- **Raw smoke output** (all 8 mandatory-check raw measurements, per-variant timings, per-step reward
  series): durable copy at
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/T710-corl-admission/final_evidence.json` (sha256
  `7df844c120f1939da0f64c062488342b453a8afc95ca435346c07caf88c815ca`), referenced as manifest
  artifact `final-evidence-raw`.
- **Raw 1-step/2-record dry-run output** (used to shake out bugs before the bounded real run; not
  itself the pass/fail evidence): durable copy at
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/T710-corl-admission/dryrun_evidence.json` (sha256
  `f9ba29f39e246cd28b7dbc56a34302f25541a949b1f2d05c0479b652d63e28ba`), referenced as manifest
  artifact `dryrun-evidence-raw`.
- **Raw stdout/stderr log of the final bounded run**: durable copy at
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/cjobs/t710_final.log` (sha256
  `878680e59e5baa64d9ff65dad000b5529c4f07011361244ccfa997f30b57725e`), referenced as manifest
  artifact `final-run-log-raw`.
- **Check outcomes and resource accounting**: `runs/corl-admission-v1/metrics.json`.

## Config identity (`config_sha256`)

`manifest.json`'s `config_sha256` (`198cdf1329fba0c1a4ed8dab28e1bf293edf20304cd9a4a202af44281d0441a3`)
is the sha256 of the following canonical JSON (keys sorted, `:`/`,` separators, no extra whitespace),
capturing the model/dataset/code revisions and smoke hyperparameters that define this run:

```json
{
  "auxiliary_models": {"all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130"},
  "code": {"pinned_commit": "0c92629f9b307a32bb286ae3562809e941d1bb0b", "upstream_repo": "https://github.com/mm-vl/ULM-R1"},
  "dataset": {"micro_split_sha256": "b9e7c1d211ae509a57a3adbddccd14ca15d1dfadd83f20b5817542b59bafa8cd", "name": "mm-vl/x2x_rft_22k", "num_records_used": 8, "pinned_revision": "f52833ce01b5657294bed87f23f27d04b92838b9"},
  "model": {"name": "deepseek-ai/Janus-Pro-1B", "pinned_revision": "960ab33191f61342a4c60ae74d8dc356a39fafcb"},
  "run_id": "corl-admission-v1",
  "smoke": {"max_steps": 4, "num_generations": 4, "reference_immutability_probe_beta": 0.1, "reference_immutability_probe_num_generations": 2, "reward_funcs": ["t2i_bid_cycle_reward", "t2i_ti_sim", "qa_accuracy", "format"], "variants": ["upstream_exact", "corrected_candidate", "reference_immutability_probe"]},
  "task_id": "T710"
}
```

## Known limitations (documented, not silently omitted)

1. **Check #3 (image-token alignment) recorded `alignment_confirmed: false`** at both n=2
   (dry-run) and n=8 (final run) records — the deliberately shifted-by-one control's mean logp was
   slightly *higher* than the correct/unshifted computation's mean logp. Source re-read confirms the
   real trainer applies no shift to t2i tokens (matching the probe's "correct" computation exactly);
   this is judged a genuine small-sample/spatial-correlation effect in the 24x24 VQ grid, not a probe
   defect, and is recorded honestly rather than adjusted to force a "true" outcome. See
   `reports/T710/failure-ledger.md` for full disposition.
2. **The batched MC/OE reward-dispatch defect (check #4 / D14) was not re-triggered by the GPU smoke
   run itself** — the real 32-record micro-split is 32/32 `qa_type=OE`, so no MC examples exist to
   exercise the defect in-situ. The defect was instead demonstrated by a dedicated synthetic MC/OE
   probe run directly against `common_qa_accuracy_reward` in the pinned CoRL venv (recorded in
   `configs/corl/admission/discrepancy-lock.yaml` D14), consistent with this task's
   "repair only locally-demonstrated correctness defects" scope — no repair was made since the actual
   smoke data never hits the broken branch.
3. **`upstream_exact`'s construction failure is treated as a pass-contributing, expected outcome**,
   not a masked failure: it reproduces a previously documented, independently-verified upstream
   defect (D11) exactly as predicted, and `corrected_candidate` (the variant this task's frozen
   protocol is actually gated on) completed all steps, checkpoint reload, and the reference
   immutability probe successfully.
4. **Only one GPU (index 0) of the 8 available H20s was used** — the resource envelope permits up to
   8; a single-GPU bounded smoke was sufficient to produce finite rollouts/rewards/losses/gradients
   and one clean optimizer-step/checkpoint-reload cycle within the 16 GPU-hour budget.

## Status

`pass` — `corrected_candidate` completed construction, 4 optimizer steps, and checkpoint save/reload
with `unauthorized_changed_param_count: 0` (0/219 trainable parameter names changed outside the
authorized set) and finite rollouts/rewards/losses/gradients throughout (`total_wallclock_seconds:
106.42`, ≈0.0296 GPU-hours, well inside the 16 GPU-hour envelope). The reference-immutability probe
confirmed `ref_model_immutable: true` (0 changed reference-model parameters). All 8 mandatory
implementation checks have an explicit recorded outcome in `runs/corl-admission-v1/metrics.json`
(`checks.*`), including the two check outcomes that are themselves defect findings (check #4 MC/OE
dispatch = D14, `upstream_exact` construction failure = D11) and the one counter-intuitive but
honestly recorded result (check #3 alignment probe). Remote reverification of every declared
manifest artifact (`configs/corl/admission/artifact-verification.json`) and storage preflight
(`configs/corl/admission/storage-preflight.json`, `status: pass`, `filesystem_class: local`) both
pass. Per the task's own gate wording, the confirmed upstream defects (D11, D14) are valid results
that block T730 until this task's corrected protocol (the `corrected_candidate` path exercised here)
is locally frozen for reuse — they do not block this task's own `pass` status.
