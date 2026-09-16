# T260 claim check

Each row maps a claim or pass/fail-gate bullet from
`tasks/T260-posttraining-data-admission.md` to the evidence that supports
or refutes it. This revision (2026-09-16) also maps local review's 8
numbered revision items to their resolution (Sec. "Local review items,
2026-09-16" below), and updates every real-data number to the re-run
build's measured values.

## Research claim

> "The method requires enough data for actual post-training while
> retaining a paired semantic subset that isolates optimization geometry
> from task distribution differences."

Supported. The paired semantic core (D1, COCO captions) is admitted with a
verified bidirectional (`i2t`/`t2i`) mapping over the same
`group_key`-identified image universe (`paired_core.bidirectional_mapping_verified
== true` in `runs/data-admission-posttraining-v1/metrics.json`), and the
training pool built from D1-D3 totals 295,841 records
(`pilot.usable_task_records = 270,207`, no cap applied this round -- 170%
over the 100,000 floor).

## Objective

> "Audit and prepare deterministic manifests for paired image-caption
> understanding/generation, visual instruction understanding, diverse
> text-to-image generation, and independent diagnostic/evaluation splits."

| Sub-objective | Status | Evidence |
|---|---|---|
| Paired image-caption (D1) manifests | **Done** | `configs/data/posttraining-v1/{diagnostic,pilot_train*,pilot_validation,pilot_meta}.jsonl` (COCO records), `d1_record_count = 118287` |
| Visual instruction understanding (D2) manifests | **Done** | Same files (LLaVA records), 157,712 records admitted, each carrying `training_constraints.restricted_as_training_target = true` |
| Diverse text-to-image generation (D3) manifests | **Done** | Same files (DiffusionDB records), 19,842 records admitted after the two-stage archive-then-row filter |
| Independent diagnostic/evaluation splits | **Done** | `configs/data/posttraining-v1/diagnostic.jsonl` (1,828 records, group-disjoint; 737 of them D1-paired) and `evaluation_only.yaml` (5,000 COCO-val2017-derived D4 records) |

> "Initial candidates are COCO captions/images, LLaVA-Instruct-150K, and a
> license-compatible deterministic JourneyDB subset. Propose alternatives
> when access, license, provenance, or storage makes a candidate
> inadmissible."

JourneyDB was audited (`reports/T260/source-license-audit.md`) and found
inadmissible on two independent grounds (gated identity-form access;
explicit no-redistribution Terms of Usage). DiffusionDB 2M (CC0-1.0,
ungated) was proposed and adopted as the D3 replacement.

## Local review items, 2026-09-16 (revision_needed -> resolution)

| # | Item | Resolution | Evidence |
|---|---|---|---|
| 1 | Distinguish `metadata_admitted` from `media_materialized` explicitly/auditably | Every record's `image` object now carries both booleans explicitly; `metrics.json` reports `media.metadata_admitted_records = 300841` vs. `media.media_materialized_records = 65` | `src/comppareto/data/{coco,llava,diffusiondb}.py` record schema; `runs/data-admission-posttraining-v1/metrics.json` |
| 2 | Media-materialization plan for exact retained groups, without full-corpus downloads | New report: 25 DiffusionDB parts + 2 COCO archives, real per-archive bytes/sha256, no content downloaded | `reports/T260/media-materialization-plan.md` |
| 3 | Reduce DiffusionDB archive fan-out if impractical | Two-stage archive-then-row redesign: 1,999 archives/1.24TB -> 25 archives/15.6GB, 18,339 -> 19,842 rows | `src/comppareto/data/diffusiondb.py`; `reports/T260/mixture-and-accounting.md` Sec. 7 |
| 4 | Preregistered REAL media-availability probes per source/split | `comppareto.data.media_check`: 65 real network probes (5 per of 13 source/split pairs), selection rule fixed in code before any probe ran; 65/65 succeeded | `src/comppareto/data/media_check.py`; `reports/T260/media-materialization-plan.md` Sec. 4 |
| 5 | Replace monolithic `pilot_train.jsonl` with shards/index | 7 byte-bounded shards (max 40MB each) + `pilot_train.shards.json` hash-addressed index | `src/comppareto/data/build.py::_write_pilot_train_shards`; `configs/data/posttraining-v1/pilot_train.shards.json` |
| 6 | Explicit, conservative decision on LLaVA GPT-derived text terms | `training_constraints.restricted_as_training_target = true` on every D2 record; restricts assistant-turn text as a training target absent an explicit override | `src/comppareto/data/llava.py::TRAINING_CONSTRAINTS`; `reports/T260/source-license-audit.md` Sec. "D2 -- LLaVA GPT-terms decision" |
| 7 | Near-duplicate checks beyond exact `group_key` equality | Exact-normalized-text duplicate groups (558) + LSH-bounded shingle-Jaccard near-duplicate pairs (205,558 at threshold 0.8), scoped to diagnostic/pilot_validation/pilot_meta | `src/comppareto/data/near_dup.py`; `runs/data-admission-posttraining-v1/metrics.json:near_duplicates` |
| 8 | Report D1 paired examples within `diagnostic` separately from the total | `paired_core.diagnostic_d1_paired_record_count = 737` reported alongside `paired_core.diagnostic_total_record_count = 1828` | `runs/data-admission-posttraining-v1/metrics.json:paired_core`; `reports/T260/split-and-decontamination.md` Sec. 8 |

## Required audit dimensions

| Dimension | Status | Evidence |
|---|---|---|
| Canonical source and revision/date | **Done** | `configs/data/posttraining-v1/sources.yaml` (real sha256/byte counts for all 3 admitted sources) |
| Terms for images/annotations/prompts/generated images | **Done** | `reports/T260/source-license-audit.md` |
| Redistribution restrictions | **Done** | Same report; JourneyDB's explicit no-redistribution clause is the reason for its rejection |
| Download method and size | **Done** | `reports/T260/first-report.md` Sec. 2, 8, 9; `reports/T260/media-materialization-plan.md` |
| Schema, media availability, corrupt-record policy | **Done** | `src/comppareto/data/records.py:validate_record`; `src/comppareto/data/media_check.py` (real probes) |
| Counts before and after filtering | **Done** | `reports/T260/mixture-and-accounting.md` Sec. 1 |
| Safety and personal-data considerations | **Done** | DiffusionDB's `image_nsfw`/`prompt_nsfw` gate; JourneyDB's identity-collecting access form contributed to rejection |
| Decontamination against declared evaluations | **Done** | `splits.evaluation_records_in_training = 0`; `reports/T260/split-and-decontamination.md` Sec. 5 |
| Split unit and near-duplicate policy | **Done** | `reports/T260/split-and-decontamination.md` Secs. 1, 4, 6, 7 (near-dup, new this round) |
| Mapping into both task formats | **Done** | D1 records carry `task_directions: ["i2t", "t2i"]`; `paired_core.bidirectional_mapping_verified = true` |
| Expected tokens, images, and storage | **Done** | `reports/T260/mixture-and-accounting.md`; `reports/T260/media-materialization-plan.md` |

## Frozen pilot manifests

> "1. `diagnostic`: 512--2,048 paired examples where available;
> 2. `pilot_train`: at least 100,000 usable task records across D1-D3;
> 3. `pilot_validation`: group-disjoint validation records;
> 4. `pilot_meta`: disjoint adaptation/meta records;
> 5. `evaluation_only`: held-out benchmark references never used for
> training."

| Manifest | Requirement | Measured | Status |
|---|---|---|---|
| `diagnostic` | 512-2,048 | 1,828 total (737 D1-paired; local review item 8) | **Pass** |
| `pilot_train` | >= 100,000 usable, or an evidence-backed reduced target | 270,207 (uncapped this round; written as 7 byte-bounded shards, local review item 5) | **Pass** |
| `pilot_validation` | group-disjoint | 14,884; `cross_split_duplicate_groups = 0` | **Pass** |
| `pilot_meta` | disjoint | 8,922; `cross_split_duplicate_groups = 0` | **Pass** |
| `evaluation_only` | held out, never used for training | 5,000; `evaluation_records_in_training = 0` | **Pass** |

> "The executor must not choose examples after observing model gradients or
> method outcomes."

**Honored.** `assign_split` (`src/comppareto/data/split.py`) is a pure
SHA-256 hash-bucket function of a `group_key` string alone;
`comppareto.data.diffusiondb`'s archive-selection stage is likewise a
pure function of the archive's own numeric id; the near-duplicate and
media-probe sample selections are both preregistered (fixed by code
before any result is observed). No model was loaded, and no gradient,
loss, or prediction was ever computed at any point in this task
(`resources.gpu_hours = 0`, `resources.model_inference_calls = 0`).

## Sampling and accounting proposal

| Sub-item | Status | Evidence |
|---|---|---|
| Initial D1/D2/D3 mixture | **Done** | `reports/T260/mixture-and-accounting.md` Secs. 1-3 |
| Task batch construction | **Documented** | `reports/T260/first-report.md` Sec. 5; deferred to T270 |
| Image/token/optimizer-step accounting | **Done (image/record counts); optimizer-step is out of scope** | `reports/T260/mixture-and-accounting.md` |
| Paired versus independent-task controls | **Done** | D1 (paired) is explicitly separated by `role: D1_paired` from D2/D3 in every record |
| Equal-data and equal-compute comparisons | **Deferred** | Explicitly out of this task's scope |
| Failed/corrupt example ledger policy | **Done** | `src/comppareto/data/records.py:validate_record` + fail-loud policy; `reports/T260/failure-ledger.md` |

## Execution stages

| Stage | Status | Evidence |
|---|---|---|
| 1. Commit and push a source/license/storage first report | **Done** | `reports/T260/first-report.md` |
| 2. Resolve official metadata and terms | **Done** | `reports/T260/source-license-audit.md` |
| 3. Implement deterministic manifest builders and stable sample IDs | **Done** | `src/comppareto/data/{coco,llava,diffusiondb,ids,build}.py`, `tests/data/` |
| 4. Materialize metadata and a small verification sample only | **Done** | Full metadata downloaded (~676MB); 65 real, preregistered media probes (local review item 4), no bulk image archive fetched |
| 5. Validate schema, availability, split disjointness, and duplicate isolation | **Done** | Real measured `cross_split_duplicate_groups = 0`, `evaluation_records_in_training = 0`, `duplicate_record_ids = 0`, plus real media probes and near-duplicate checks (new this round) |
| 6. Produce storage/download plans for the pilot subset | **Done** | `reports/T260/first-report.md` Secs. 3, 9; `reports/T260/media-materialization-plan.md` (new, exact retained groups, local review item 2) |
| 7. Emit admission decisions and a primary/fallback mixture | **Done** | `configs/data/posttraining-v1/sources.yaml`; `reports/T260/mixture-and-accounting.md` Sec. 6 |

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
| Pilot meets 100,000 usable task records (or an evidence-backed reduced target) | **Pass** | `pilot.usable_task_records = 270,207` -- no cap applied this round, 170% over the 100,000 floor |
| Unknown/incompatible terms block the affected source | **Honored** | JourneyDB blocked and excluded from "admitted" |

## Resource envelope

> "CPU/network/storage task; zero GPU hours; no download above 20 GB
> before the first report; no complete multi-terabyte download; no model
> inference, gradients, or training."

| Constraint | Status | Evidence |
|---|---|---|
| Zero GPU hours | **Honored** | `resources.gpu_hours = 0` |
| No download above 20GB before the first report | **Honored** | `reports/T260/first-report.md` Sec. 10 |
| No single download step above 20GB (media-materialization plan) | **Honored** | `reports/T260/media-materialization-plan.md`: largest single archive is COCO `train2017.zip` at 19,336,861,798 bytes, ~660MB under 20GB |
| No complete multi-terabyte download | **Honored** | Total raw metadata downloaded across the entire task: ~1.47GB; DiffusionDB archive footprint reduced from a would-be ~1.24TB to a planned ~15.6GB; no image/video archive was ever bulk-fetched |
| No model inference, gradients, or training | **Honored** | `resources.model_inference_calls = 0`, `resources.training_steps = 0` |

## Required deliverables checklist

| Deliverable | Path |
|---|---|
| First report | `reports/T260/first-report.md` |
| Source/license audit | `reports/T260/source-license-audit.md` |
| Split/decontamination procedure | `reports/T260/split-and-decontamination.md` |
| Mixture and accounting | `reports/T260/mixture-and-accounting.md` |
| Media-materialization plan (new, local review item 2) | `reports/T260/media-materialization-plan.md` |
| Result summary | `reports/T260/result-summary.md` |
| Claim check | `reports/T260/claim-check.md` (this file) |
| Failure ledger | `reports/T260/failure-ledger.md` |
| Source and split manifests | `configs/data/posttraining-v1/{sources.yaml,diagnostic.jsonl,pilot_train.jsonl,pilot_train.shard1..6.jsonl,pilot_train.shards.json,pilot_validation.jsonl,pilot_meta.jsonl,evaluation_only.yaml}` |
| Manifest builders and tests | `src/comppareto/data/`, `tests/data/` |
| Run manifest | `runs/data-admission-posttraining-v1/manifest.json` |
| Run metrics | `runs/data-admission-posttraining-v1/metrics.json` |
| Run notes | `runs/data-admission-posttraining-v1/notes.md` |

## Conclusion

**Supports gate.** Every pass/fail-gate bullet and every frozen-manifest
numeric requirement is satisfied by real, measured values from the
complete downloaded sources -- not estimates. All 8 local review items
are addressed with real, auditable evidence, not prose restatements. One
D3 candidate was honestly rejected and replaced; the DiffusionDB
archive-fan-out was redesigned to be practically materializable; the
`pilot_train` monolith was replaced with deterministic shards; real
network probes and near-duplicate checks were added; and an explicit,
conservative decision was recorded for LLaVA's GPT-derived text. No GPU
hours, model inference, or training occurred anywhere in this task.
