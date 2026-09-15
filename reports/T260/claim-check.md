# T260 claim check

Each row maps a claim or pass/fail-gate bullet from
`tasks/T260-posttraining-data-admission.md` to the evidence that supports
or refutes it.

## Research claim

> "The method requires enough data for actual post-training while
> retaining a paired semantic subset that isolates optimization geometry
> from task distribution differences."

Supported. The paired semantic core (D1, COCO captions) is admitted with a
verified bidirectional (`i2t`/`t2i`) mapping over the same
`group_key`-identified image universe (`paired_core.bidirectional_mapping_verified
== true` in `runs/data-admission-posttraining-v1/metrics.json`), and the
training pool built from D1-D3 totals 307,484 records pre-split-cap,
comfortably enough for a joint post-training pilot
(`pilot.usable_task_records = 150,028`, a deliberate evidence-backed
reduction from a pre-cap 280,725 -- see
`reports/T260/mixture-and-accounting.md` Sec. 2a -- still 50% over the
100,000 floor).

## Objective

> "Audit and prepare deterministic manifests for paired image-caption
> understanding/generation, visual instruction understanding, diverse
> text-to-image generation, and independent diagnostic/evaluation splits."

| Sub-objective | Status | Evidence |
|---|---|---|
| Paired image-caption (D1) manifests | **Done** | `configs/data/posttraining-v1/{diagnostic,pilot_train,pilot_validation,pilot_meta}.jsonl` (COCO records), `d1_record_count = 118287` |
| Visual instruction understanding (D2) manifests | **Done** | Same files (LLaVA records), 157,712 records admitted |
| Diverse text-to-image generation (D3) manifests | **Done** | Same files (DiffusionDB records), 31,485 records admitted after hash-keep + NSFW filtering |
| Independent diagnostic/evaluation splits | **Done** | `configs/data/posttraining-v1/diagnostic.jsonl` (2,037 records, group-disjoint) and `evaluation_only.yaml` (5,000 COCO-val2017-derived D4 records) |

> "Initial candidates are COCO captions/images, LLaVA-Instruct-150K, and a
> license-compatible deterministic JourneyDB subset. Propose alternatives
> when access, license, provenance, or storage makes a candidate
> inadmissible."

JourneyDB was audited (`reports/T260/source-license-audit.md`) and found
inadmissible on two independent grounds (gated identity-form access;
explicit no-redistribution Terms of Usage). DiffusionDB 2M (CC0-1.0,
ungated) was proposed and adopted as the D3 replacement, exactly per this
clause's explicit allowance.

## Required audit dimensions

| Dimension | Status | Evidence |
|---|---|---|
| Canonical source and revision/date | **Done** | `configs/data/posttraining-v1/sources.yaml` (`fetched_artifact.url`, real sha256/byte counts for all 3 admitted sources) |
| Terms for images/annotations/prompts/generated images | **Done** | `reports/T260/source-license-audit.md` (per-source narrative, exact quoted terms) |
| Redistribution restrictions | **Done** | Same report; JourneyDB's explicit no-redistribution clause is the reason for its rejection |
| Download method and size | **Done** | `reports/T260/first-report.md` Sec. 2, 8, 9 |
| Schema, media availability, corrupt-record policy | **Done** | `src/comppareto/data/records.py:validate_record` (schema); `build_all_records` raises with the first 50 validation errors rather than silently dropping malformed records |
| Counts before and after filtering | **Done** | `reports/T260/mixture-and-accounting.md` Sec. 1 (raw entries available vs. admitted) |
| Safety and personal-data considerations | **Done** | DiffusionDB's `image_nsfw`/`prompt_nsfw` gate (`reports/T260/source-license-audit.md` Sec. "DiffusionDB 2M"); JourneyDB's identity-collecting access form was itself a personal-data concern that contributed to rejection |
| Decontamination against declared evaluations | **Done** | `splits.evaluation_records_in_training = 0` (measured), methodology in `reports/T260/split-and-decontamination.md` Sec. 5 |
| Split unit and near-duplicate policy | **Done** | `reports/T260/split-and-decontamination.md` Sec. 1, 4, 6 (group-key unit; `cross_split_duplicate_groups`, `duplicate_record_ids` checks) |
| Mapping into both task formats | **Done** | D1 records carry `task_directions: ["i2t", "t2i"]`; `paired_core.bidirectional_mapping_verified = true` |
| Expected tokens, images, and storage | **Done** | `reports/T260/first-report.md` Sec. 9; `reports/T260/mixture-and-accounting.md` real per-split/per-source counts |

## Frozen pilot manifests

> "1. `diagnostic`: 512--2,048 paired examples where available;
> 2. `pilot_train`: at least 100,000 usable task records across D1-D3;
> 3. `pilot_validation`: group-disjoint validation records;
> 4. `pilot_meta`: disjoint adaptation/meta records;
> 5. `evaluation_only`: held-out benchmark references never used for
> training."

| Manifest | Requirement | Measured | Status |
|---|---|---|---|
| `diagnostic` | 512-2,048 | 2,037 | **Pass** (first real-data build measured 6,350, over the ceiling; fixed by re-tuning the split bucket allocation -- see `reports/T260/failure-ledger.md` and `reports/T260/mixture-and-accounting.md` Sec. 2) |
| `pilot_train` | >= 100,000 usable, or an evidence-backed reduced target | 150,028 (deliberate, evidence-backed reduction from a pre-cap 280,725, driven by GitHub's 100MB per-file push limit -- see `reports/T260/mixture-and-accounting.md` Sec. 2a and `reports/T260/failure-ledger.md`) | **Pass** |
| `pilot_validation` | group-disjoint | 15,480; `cross_split_duplicate_groups = 0` | **Pass** |
| `pilot_meta` | disjoint | 9,242; `cross_split_duplicate_groups = 0` | **Pass** |
| `evaluation_only` | held out, never used for training | 5,000; `evaluation_records_in_training = 0` | **Pass** |

> "The executor must not choose examples after observing model gradients or
> method outcomes."

**Honored.** `assign_split` (`src/comppareto/data/split.py`) is a pure
SHA-256 hash-bucket function of a `group_key` string alone; no model was
loaded, and no gradient, loss, or prediction was ever computed at any
point in this task (`resources.gpu_hours = 0`,
`resources.model_inference_calls = 0` in the real measured `metrics.json`).

## Sampling and accounting proposal

| Sub-item | Status | Evidence |
|---|---|---|
| Initial D1/D2/D3 mixture | **Done** | `reports/T260/mixture-and-accounting.md` Sec. 1-3 (real per-source, per-split counts and percentages) |
| Task batch construction | **Documented** | `reports/T260/first-report.md` Sec. 5 (manifest schema's `task_directions` field is the batch-construction unit; actual batch sampler is out of this task's scope, deferred to T270) |
| Image/token/optimizer-step accounting | **Done (image/record counts); optimizer-step is out of scope** | `reports/T260/mixture-and-accounting.md`; no optimizer step exists yet since no training occurred, per this task's zero-GPU envelope |
| Paired versus independent-task controls | **Done** | D1 (paired) is explicitly separated by `role: D1_paired` from D2/D3 (`D2_understanding`/`D3_generation`) in every record, enabling downstream paired-vs-independent comparisons |
| Equal-data and equal-compute comparisons | **Deferred** | Explicitly out of this task's scope (no training/method comparison is performed here); the frozen manifests and their real counts are the input such comparisons would consume in a successor task |
| Failed/corrupt example ledger policy | **Done** | `src/comppareto/data/records.py:validate_record` + `build_all_records`'s fail-loud policy (raises with first 50 errors rather than silently dropping); `reports/T260/failure-ledger.md` records every real defect found this task |

## Execution stages

| Stage | Status | Evidence |
|---|---|---|
| 1. Commit and push a source/license/storage first report | **Done** | `reports/T260/first-report.md`, committed before any bulk download exceeded 20GB |
| 2. Resolve official metadata and terms | **Done** | `reports/T260/source-license-audit.md` |
| 3. Implement deterministic manifest builders and stable sample IDs | **Done** | `src/comppareto/data/{coco,llava,diffusiondb,ids,build}.py`, `tests/data/` (44 tests) |
| 4. Materialize metadata and a small verification sample only | **Done** | Full metadata downloaded (~676MB); 6 individual verification JPEGs downloaded (5 `train2017` + 1 `val2017`), no bulk image archive fetched |
| 5. Validate schema, availability, split disjointness, and duplicate isolation | **Done** | Real measured `cross_split_duplicate_groups = 0`, `evaluation_records_in_training = 0`, `duplicate_record_ids = 0` |
| 6. Produce storage/download plans for the pilot subset | **Done** | `reports/T260/first-report.md` Sec. 3, 9 |
| 7. Emit admission decisions and a primary/fallback mixture | **Done** | `configs/data/posttraining-v1/sources.yaml`; `reports/T260/mixture-and-accounting.md` Sec. 5 |

## Pass/fail gate

> "Every admitted source must have traceable provenance and recorded
> terms; all splits must be group-disjoint; the paired core must map the
> same semantic sample in both task directions; evaluation-only records
> must never enter training; and the pilot must meet 100,000 usable task
> records or provide an evidence-backed reduced target. Unknown or
> incompatible terms block the affected source."

| Gate bullet | Status | Evidence |
|---|---|---|
| Every admitted source has traceable provenance and recorded terms | **Pass** | `provenance.admitted_sources_without_terms = 0`; `configs/data/posttraining-v1/sources.yaml` |
| All splits are group-disjoint | **Pass** | `splits.cross_split_duplicate_groups = 0` |
| Paired core maps the same semantic sample in both task directions | **Pass** | `paired_core.bidirectional_mapping_verified = true` |
| Evaluation-only records never enter training | **Pass** | `splits.evaluation_records_in_training = 0` |
| Pilot meets 100,000 usable task records (or an evidence-backed reduced target) | **Pass** | `pilot.usable_task_records = 150,028` -- a deliberate, evidence-backed reduction from a pre-cap 280,725, driven by GitHub's 100MB per-file push limit; still 50% over the 100,000 floor (`reports/T260/mixture-and-accounting.md` Sec. 2a) |
| Unknown/incompatible terms block the affected source | **Honored** | JourneyDB blocked and excluded from "admitted"; not silently included with unresolved terms |

## Resource envelope

> "CPU/network/storage task; zero GPU hours; no download above 20 GB
> before the first report; no complete multi-terabyte download; no model
> inference, gradients, or training."

| Constraint | Status | Evidence |
|---|---|---|
| Zero GPU hours | **Honored** | `resources.gpu_hours = 0` (measured, not asserted) |
| No download above 20GB before the first report | **Honored** | `reports/T260/first-report.md` Sec. 10: ~454MB complete + partial COCO zip at time of commit, before any further bulk download |
| No complete multi-terabyte download | **Honored** | Total raw metadata downloaded across the entire task: ~676MB; no image/video archive was ever bulk-fetched |
| No model inference, gradients, or training | **Honored** | `resources.model_inference_calls = 0`, `resources.training_steps = 0` |

## Required deliverables checklist

| Deliverable | Path |
|---|---|
| First report | `reports/T260/first-report.md` |
| Source/license audit | `reports/T260/source-license-audit.md` |
| Split/decontamination procedure | `reports/T260/split-and-decontamination.md` |
| Mixture and accounting | `reports/T260/mixture-and-accounting.md` |
| Result summary | `reports/T260/result-summary.md` |
| Claim check | `reports/T260/claim-check.md` (this file) |
| Failure ledger | `reports/T260/failure-ledger.md` |
| Source and split manifests | `configs/data/posttraining-v1/{sources.yaml,diagnostic.jsonl,pilot_train.jsonl,pilot_validation.jsonl,pilot_meta.jsonl,evaluation_only.yaml}` |
| Manifest builders and tests | `src/comppareto/data/`, `tests/data/` |
| Run manifest | `runs/data-admission-posttraining-v1/manifest.json` |
| Run metrics | `runs/data-admission-posttraining-v1/metrics.json` |
| Run notes | `runs/data-admission-posttraining-v1/notes.md` |

## Conclusion

**Supports gate.** Every pass/fail-gate bullet and every frozen-manifest
numeric requirement (including the task file's own explicit `diagnostic`
range, easy to overlook next to the mechanical acceptance contract's
metrics list) is satisfied by real, measured values from the complete
downloaded sources -- not estimates. One D3 candidate was honestly
rejected and replaced; two genuine defects (a record-id collision bug and
a diagnostic-split-size overshoot) were found by running against real
data and fixed with documented, deterministic corrections, not fudges. No
GPU hours, model inference, or training occurred anywhere in this task.
