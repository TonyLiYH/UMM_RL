# T260 result summary

Task: `tasks/T260-posttraining-data-admission.md`. Branch:
`agent/T260-posttraining-data-admission`. Source revision audited:
`45c54ba403a2b5c95985a49206243449436717c8` (recorded in
`first-report.md`, confirmed as an ancestor of this branch at task start).

This revision (2026-09-16) responds to local review's 8-item
`revision_needed` (see `tasks/T260-posttraining-data-admission.md`,
"Local review requirements -- 2026-09-16"). All numbers below are from
the real, re-run build against the complete downloaded sources -- no
number is carried forward unchanged from the prior round without
re-measurement.

## Conclusion

**Supports gate.** All three admitted sources (COCO captions 2017,
LLaVA-Instruct-150K, DiffusionDB 2M) have traceable provenance and
recorded, verifiable terms; every real, measured split-disjointness and
paired-core check required by the acceptance contract and the task
file's pass/fail gate passes; the pilot training pool clears the 100,000
usable-record floor by a wide margin (270,207, uncapped this round); and
zero GPU hours were spent anywhere in this task. One D3 candidate
(JourneyDB) was rejected on license/access grounds and honestly replaced
with a license-cleared fallback (DiffusionDB 2M). All 8 local review
items are addressed with real, auditable evidence -- see
`reports/T260/claim-check.md` for the per-item mapping.

## Current admission run (authoritative status)

The formal run is `runs/data-admission-posttraining-v1/` -- see
`runs/data-admission-posttraining-v1/manifest.json` (`status: pass`,
hash/byte-addressed artifacts for every raw source file and every
produced manifest/config/shard file) and
`runs/data-admission-posttraining-v1/metrics.json` (the real, measured
values below). This is a zero-GPU data-engineering/audit run: no model
was loaded, no inference or training step ran.

Measured, real figures (from `metrics.json`, produced by
`comppareto.data.build` against the complete, real downloaded sources --
not estimated or asserted):

- **Training-pool records built (no cap applied this round)**: 295,841
  total (`d1`=118,287 COCO captions, `d2`=157,712 LLaVA conversations,
  `d3`=19,842 DiffusionDB rows after the two-stage archive-then-row
  filter -- see "DiffusionDB archive fan-out" below).
- **Split counts (as actually written)**: `diagnostic`=1,828;
  `pilot_validation`=14,884; `pilot_meta`=8,922; `pilot_train`=270,207;
  `evaluation_only`=5,000 (COCO `val2017`, one caption/image).
- **`pilot.usable_task_records`**: 270,207 -- no cap is applied this
  round; the prior round's `PILOT_TRAIN_BUCKET_CEILING` (a record-count
  reduction to dodge GitHub's 100MB blob limit) has been replaced by
  deterministic byte-bounded sharding (local review item 5, see below),
  which accommodates the full, uncapped split. Clears the acceptance
  contract's `>= 100,000` gate by 170%.
- **`diagnostic` size vs. the task file's explicit 512-2,048 range**:
  1,828 -- inside the range, measured directly (no re-tuning of
  `SPLIT_FRACTIONS` was needed this round; see
  `reports/T260/mixture-and-accounting.md` Sec. 2).
- **`diagnostic_d1_paired_record_count`** (local review item 8, reported
  separately from the split total): 737 of 1,828 total `diagnostic`
  records are the D1 (COCO caption) bidirectionally-paired subset; the
  remainder are D2 (980) and D3 (111).
- **`splits.cross_split_duplicate_groups`**: 0.
- **`splits.evaluation_records_in_training`**: 0.
- **`paired_core.bidirectional_mapping_verified`**: true.
- **`provenance.admitted_sources_without_terms`**: 0.
- **`duplicate_record_ids`**: 0.
- **`media.metadata_admitted_records`** (local review item 1): 300,841
  (all D1-D4 records, including `evaluation_only`).
- **`media.media_materialized_records`** (local review item 1): 65 (the
  real, preregistered network-probe-confirmed sample -- see "Media
  availability" below). Every other record is `metadata_admitted = true`
  but `media_materialized = false` -- an explicit, auditable field on
  every record, not just prose.
- **`near_duplicates.*`** (local review item 7): 558 exact-normalized-text
  duplicate groups; 205,558 shingle-Jaccard near-duplicate pairs (Jaccard
  `>= 0.8`), scoped to `diagnostic` + `pilot_validation` + `pilot_meta`
  (25,634 records).
- **`resources.gpu_hours`**: 0. `resources.model_inference_calls`: 0.
  `resources.training_steps`: 0.
- **Download volume**: ~1.47GB raw source files this round (COCO
  annotations zip 252,907,541 bytes; LLaVA JSON 228,941,895 bytes;
  DiffusionDB metadata parquet 194,548,652 bytes; DiffusionDB JSONL
  derived-cache 799,203,140 bytes, produced locally by conversion) plus
  65 individual verification-media probes (order of a few MB total). No
  single download step exceeded 20GB; no full media corpus was
  downloaded anywhere in this task.

## Media-materialization plan (local review item 2)

`reports/T260/media-materialization-plan.md` (new this round) documents,
for the exact retained groups only: the 25 kept DiffusionDB part
archives (real per-archive byte size and sha256, obtained from the
Hugging Face Hub's tree API -- no archive content downloaded, total
15,587,973,934 bytes), and COCO's `train2017.zip`
(19,336,861,798 bytes) / `val2017.zip` (815,585,330 bytes) with real
`Content-Length` values from plain HTTP `HEAD` requests. Grand total
across all 27 archives: ~35.7GB, no single archive over 20GB (COCO
`train2017.zip` is the largest, with ~660MB headroom under the ceiling).
This is a plan only -- no bulk media bytes were downloaded or committed
to this repository.

## Media availability (local review item 4)

`comppareto.data.media_check` ran a real, preregistered network probe --
5 records per `(source, split)` pair actually present in the built
manifests (13 pairs, selection rule fixed in code before any probe
result was observed) -- against the live network: 65 probes total, all
65 succeeded. COCO probes resolved over plain HTTP
(`images.cocodataset.org`); DiffusionDB probes resolved via an HTTP
Range read of the specific member inside its live `part-NNNNNN.zip`
archive. Full detail: `reports/T260/media-materialization-plan.md`
Sec. 4.

## `pilot_train` sharding (local review item 5)

`configs/data/posttraining-v1/pilot_train.jsonl` is no longer a single
monolithic file. It is written as 7 deterministic, byte-bounded shards
(`_write_pilot_train_shards`, `PILOT_TRAIN_SHARD_MAX_BYTES = 40,000,000`)
plus an external hash-addressed index, `pilot_train.shards.json`:

| Shard | File | Records | Bytes |
|---|---|---|---|
| 0 | `pilot_train.jsonl` | 64,048 | 39,999,850 |
| 1 | `pilot_train.shard1.jsonl` | 53,555 | 39,999,516 |
| 2 | `pilot_train.shard2.jsonl` | 30,746 | 39,999,125 |
| 3 | `pilot_train.shard3.jsonl` | 30,747 | 39,999,116 |
| 4 | `pilot_train.shard4.jsonl` | 30,743 | 39,999,373 |
| 5 | `pilot_train.shard5.jsonl` | 30,731 | 39,998,951 |
| 6 | `pilot_train.shard6.jsonl` | 29,637 | 32,058,540 |
| **Total** | 7 files | **270,207** | **272,054,471** |

Every shard is well under GitHub's 50MB warn threshold and 100MB hard
limit. Full detail: `reports/T260/mixture-and-accounting.md` Sec. 4.

## DiffusionDB archive fan-out (local review item 3)

| | Prior design | This round |
|---|---|---|
| Archives touched | 1,999 of 2,000 | 25 of 2,000 |
| Total archive bytes | ~1.24 TB | 15,587,973,934 (~15.6GB) |
| Rows retained | 18,339 | 19,842 |

Full derivation and the 25 kept part IDs:
`reports/T260/mixture-and-accounting.md` Sec. 7,
`reports/T260/media-materialization-plan.md`.

## LLaVA GPT-derived text decision (local review item 6)

Every D2 record now carries a machine-readable
`training_constraints.restricted_as_training_target = true` field
(`src/comppareto/data/llava.py::TRAINING_CONSTRAINTS`): the GPT-4-
generated assistant-turn text must not be used as a raw gradient
training target/label in any CompPareto post-training run without an
explicit, separately recorded local-review sign-off overriding this
default. The human instruction text and referenced COCO image remain
unrestricted. Full narrative:
`reports/T260/source-license-audit.md` Sec. "D2 -- LLaVA GPT-terms
decision".

## What was built/run

- First report (audit, before bulk download): `reports/T260/first-report.md`.
- Source/license audit: `reports/T260/source-license-audit.md`.
- Split/decontamination methodology (incl. near-duplicate detection and
  D1-paired-within-diagnostic reporting): `reports/T260/split-and-decontamination.md`.
- Mixture and accounting (real per-split, per-source counts, DiffusionDB
  fan-out before/after, media/near-dup sections):
  `reports/T260/mixture-and-accounting.md`.
- Media-materialization plan (new): `reports/T260/media-materialization-plan.md`.
- Manifest builders: `src/comppareto/data/{coco,llava,diffusiondb,prep_diffusiondb,records,dedup,near_dup,split,ids,media_check,build,run_artifacts}.py`.
- Real manifests built and frozen from the complete, real downloaded
  sources: `configs/data/posttraining-v1/{diagnostic,pilot_validation,pilot_meta}.jsonl`,
  the 7-shard `pilot_train.{jsonl,shard1..6.jsonl}` + `pilot_train.shards.json`,
  and `configs/data/posttraining-v1/evaluation_only.yaml`.
- No model was loaded, no inference or training step ran, anywhere in
  this task.

## Sources, revisions, and licenses (cumulative)

| Source | Repo / URL | Fetched artifact | sha256 | Bytes | License |
|---|---|---|---|---|---|
| COCO captions 2017 | `images.cocodataset.org/annotations/annotations_trainval2017.zip` | full zip (annotation JSON only) | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` | 252,907,541 | CC BY 4.0 (annotations); mixed per-photo Flickr license (images, referenced by path only, never embedded) |
| LLaVA-Instruct-150K | `huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K` | `llava_instruct_150k.json` | `6b68bc5ca2bfd8a71119af0e8454929668ccda6a334955ccc95d114fc8d082fa` | 228,941,895 | CC BY 4.0 (+ conservative `restricted_as_training_target` decision on GPT-4-derived text; see above) |
| DiffusionDB 2M | `huggingface.co/datasets/poloclub/diffusiondb` | `metadata.parquet` (2m subset) | `eecd341187bc91c07f5994ad0660d40228ea025616fd57a509bef8323677c68f` | 194,548,652 | CC0-1.0, fully unrestricted |
| JourneyDB (rejected) | `huggingface.co/datasets/JourneyDB/JourneyDB` | not fetched | n/a | n/a | Gated identity-form access + no-redistribution Terms of Usage -- **rejected**, replaced by DiffusionDB 2M |

## Defects/changes this round, found and fixed by running against real data

See `reports/T260/failure-ledger.md`'s 2026-09-16 entries for the full
record of: the DiffusionDB archive-fan-out redesign, the pilot_train
sharding replacing the prior record-count cap, the real media-
availability probes, the near-duplicate-detection addition, the LLaVA
training-constraints decision, and the D1-diagnostic-paired separate
reporting. Nothing here was fixed by fudging a metric or fabricating a
count; every change is a deterministic code change re-measured against
the complete real sources.

## Scope limitations (documented, not hidden; carried forward from prior rounds, unchanged)

- `evaluation_only` currently covers D1/D4 (COCO `val2017`) only; D2/D3
  have no analogous pre-existing official held-out partition this task
  adopted. See `reports/T260/split-and-decontamination.md` Sec. 3 and
  `reports/T260/failure-ledger.md`.
- COCO's `train2017.zip`/`val2017.zip` `ETag` values are S3
  multipart-upload hashes, not plain content hashes -- an honest gap
  recorded in `reports/T260/media-materialization-plan.md` Sec. 2 rather
  than a fabricated content hash.

## No model training/inference

No GPU was used, no model was loaded, and no training or inference step
ran at any point in this task -- confirmed by `resources.gpu_hours == 0`,
`resources.model_inference_calls == 0`, and `resources.training_steps ==
0` in the real measured `metrics.json`, consistent with the task file's
resource envelope ("CPU/network/storage task; zero GPU hours ... no model
inference, gradients, or training").
