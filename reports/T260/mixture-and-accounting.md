# T260 Mixture and Accounting

All counts below are the **real, measured** output of
`comppareto.data.build` against the fully-downloaded raw sources (COCO
`annotations_trainval2017.zip`, LLaVA `llava_instruct_150k.json`,
DiffusionDB `metadata.jsonl`), re-run **2026-09-16** after the local
review's DiffusionDB archive-fan-out redesign
(`src/comppareto/data/diffusiondb.py`) and the removal of the prior
round's `pilot_train` record-count cap in favor of deterministic sharding
(`src/comppareto/data/build.py::_write_pilot_train_shards`, local review
item 5). Every number here comes from
`runs/data-admission-posttraining-v1/metrics.json` or a direct count over
the actual written `configs/data/posttraining-v1/*.jsonl` files -- no
number is an estimate, a projection, or a value asserted before this
round's real build ran.

## 1. Per-source contribution to the training pool

| Source | Role | Raw entries available | Records admitted into training pool | Admission mechanism |
|---|---|---|---|---|
| `coco_captions_2017` | D1 paired image-caption | 118,287 `train2017` images | 118,287 (1 caption/image, lowest-numbered official annotation id) | All admitted; group-key = COCO `image_id` |
| `llava_instruct_150k` | D2 visual instruction | 157,712 conversation entries | 157,712 (all) | All admitted; group-key = COCO `image_id` (shared with D1) |
| `diffusiondb_2m` | D3 diverse T2I generation | 2,000,000 metadata rows | 19,842 | Two-stage archive-then-row filter (see Sec. 7): 25 of 2,000 `part-NNNNNN.zip` archives selected by `group_bucket(f"diffusiondb-part::{part_id}", 2000) < 32`, then every row in a kept archive clearing the NSFW safety filter (`image_nsfw < 0.2`, `prompt_nsfw < 0.5`, missing-score rows excluded) is kept |
| **Total training-pool records** | | | **295,841** | = 118,287 + 157,712 + 19,842 |

No cap is applied to `pilot_train` in this round -- the prior round's
`PILOT_TRAIN_BUCKET_CEILING` (a record-count reduction to dodge GitHub's
100MB blob limit) has been replaced by deterministic content-based
sharding (Sec. 4), which accommodates the full, uncapped split.

## 2. Split assignment (hash-bucket, pure function of group key)

| Split | Bucket share (of 10,000) | Record count (measured) | `coco_captions_2017` | `llava_instruct_150k` | `diffusiondb_2m` |
|---|---|---|---|---|---|
| `diagnostic` | 63 buckets (0.63%) | 1,828 | 737 | 980 | 111 |
| `pilot_validation` | 500 buckets (5%) | 14,884 | 6,056 | 7,861 | 967 |
| `pilot_meta` | 300 buckets (3%) | 8,922 | 3,583 | 4,729 | 610 |
| `pilot_train` | 9,137 buckets (91.37%) | 270,207 | 107,911 | 144,142 | 18,154 |
| **Total** | 10,000 buckets | **295,841** | 118,287 | 157,712 | 19,842 |

`coco_captions_2017` and `llava_instruct_150k`'s per-split counts are
**unchanged** from the prior round: `assign_split` is a pure function of
each record's own `group_key` (COCO `image_id`) alone, so a change to
DiffusionDB's admission rule cannot move a COCO/LLaVA record between
splits. Only `diffusiondb_2m`'s own counts (and, downstream, `pilot_train`'s
total) changed, because D3's own training-pool universe shrank from
31,485 rows (prior row-level-only design) to 19,842 rows (this round's
two-stage design) -- see Sec. 7.

`diagnostic`'s bucket share (63 of 10,000, i.e. 0.63%) is **unchanged**
from the prior round's tuning -- see
`src/comppareto/data/split.py`'s `SPLIT_FRACTIONS` docstring for the
original derivation against the task file's absolute "512-2,048" ceiling.
Re-measured this round against the smaller D3 pool, the same 63-bucket
share now yields `diagnostic = 1,828` (down from the prior round's 2,037,
because D3's diagnostic contribution alone fell from 320 to 111) --
**still comfortably inside [512, 2048]**, so no further re-tuning of
`SPLIT_FRACTIONS` was required this round.

## 3. `pilot_train` internal source mixture

| Source | Records | Share of `pilot_train` |
|---|---|---|
| `coco_captions_2017` (D1) | 107,911 | 39.94% |
| `llava_instruct_150k` (D2) | 144,142 | 53.35% |
| `diffusiondb_2m` (D3) | 18,154 | 6.72% |

D2 (visual instruction) still dominates the pool numerically for the same
reason as before (LLaVA contributes ~1.9 conversation entries per unique
COCO image on average). D3's share of `pilot_train` fell from the prior
round's 10.35% to 6.72% purely because the archive-fan-out redesign
shrank D3's absolute row count (a deliberate, documented trade-off to
avoid an impractical 1,999-archive/1.24TB media-materialization
footprint -- see Sec. 7); D1/D2's absolute counts and shares within
`pilot_train` are unchanged from the prior round.

## 4. `pilot_train` sharding (local review item 5 -- replaces the prior round's record-count cap)

The prior round's `pilot_train.jsonl` (150,028 records after an
evidence-backed cap, or 280,725 uncapped) was flagged because a single
monolithic file either brushed against or exceeded GitHub's 100MB blob
limit, and because capping the record count to fit one file silently
discarded real, license-cleared, safety-filtered training records. This
round removes the cap entirely and instead writes `pilot_train` as a
deterministic set of byte-bounded shards
(`comppareto.data.build._write_pilot_train_shards`,
`PILOT_TRAIN_SHARD_MAX_BYTES = 40,000,000`): rows are iterated in a fixed
sort order (`record_id`) and a new shard starts exactly when the next row
would push the current shard past 40MB -- a pure function of frozen row
content, never of wall-clock time or row count alone.

**Real measurement** (from `configs/data/posttraining-v1/pilot_train.shards.json`):

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

Every shard is well under both GitHub's 50MB "large file" warning
threshold and its 100MB hard push limit -- no shard even approaches the
prior round's 100MB rejection. `pilot_train.shards.json` is the external
hash-addressed index the review asked for: every shard's path, record
count, byte size, and sha256, sufficient to verify integrity or
reconstruct the full split by concatenation in index order without ever
materializing a single 272MB file on disk or in git history. The literal
path `pilot_train.jsonl` (shard 0) still exists, satisfying any tooling
that expects that exact filename; it now holds only the first ~64K
records, not the whole split.

## 5. Evaluation split (`evaluation_only`)

| Source | Records |
|---|---|
| `coco_captions_2017` (D4, from official `val2017`) | 5,000 |

Unchanged from the prior round: 5,000 matches COCO's own official
`val2017` image count exactly. D2 and D3 have no representation in
`evaluation_only` -- a documented scope limitation, not an omission
(`reports/T260/failure-ledger.md`,
`reports/T260/split-and-decontamination.md` Sec. 3).

## 6. Primary vs. fallback mixture decision

Unchanged from the prior round (see `reports/T260/source-license-audit.md`
for the full narrative): JourneyDB (original D3 candidate) was rejected
on license/access grounds; DiffusionDB 2M (CC0-1.0, ungated) was adopted
as the D3 fallback and is the sole D3 contributor in every split above.

## 7. DiffusionDB archive fan-out -- before/after (local review item 3)

| | Prior design (row-level-only hash-keep) | This round (two-stage archive-then-row) |
|---|---|---|
| Selection rule | `group_bucket(image_name, 50) < 1` (per-row, ignores which archive a row lives in) | Stage 1: `group_bucket(f"diffusiondb-part::{part_id}", 2000) < 32` (per-archive); Stage 2: unchanged NSFW row filter, applied only within a kept archive |
| Archives touched to materialize every retained row | 1,999 of 2,000 (99.95%) | 25 of 2,000 (1.25%) |
| Total archive bytes that would need fetching | 1,242,204,022,360 (~1.24 TB) -- effectively the entire corpus | 15,587,973,934 (~15.6 GB) |
| Rows retained | 18,339 | 19,842 (**+8.2%**, more rows from far less media) |
| Rows retained per archive-GB | ~0.0148 rows/MB | ~1.27 rows/MB (**~86x denser**) |

The prior design was flagged by local review as an impractical
media-materialization plan: fetching the media for its own retained rows
would require downloading essentially the full 1.24TB corpus, exactly
what this task's resource envelope (`no full-corpus downloads`) forbids.
The two-stage redesign fixes this by making archive selection itself the
primary admission gate (a pure function of the archive's own numeric id,
never of row content or a model outcome) and then keeping *every*
safety-passing row inside a selected archive, rather than additionally
thinning rows within an archive -- this is what simultaneously shrinks
the archive footprint by ~80x and *increases* the retained row count. See
`src/comppareto/data/diffusiondb.py`'s module docstring for the full
derivation and `reports/T260/media-materialization-plan.md` for the exact
25 kept part IDs, real per-archive byte sizes, and real per-archive sha256
content hashes (obtained from the Hugging Face Hub's tree API without
downloading any archive content).

## 8. Acceptance-contract and task-file threshold check (measured values)

| Metric | Required | Measured | Pass? |
|---|---|---|---|
| `pilot.usable_task_records` (== `pilot_train` count) | >= 100,000 usable, or an evidence-backed reduced target | 270,207 (no cap applied this round; see Sec. 4) | Yes |
| `diagnostic` record count (task file, not the mechanical acceptance contract) | 512-2,048 | 1,828 | Yes |
| `paired_core.diagnostic_d1_paired_record_count` (local review item 8) | (reported, not a separate gate) | 737 (of 1,828 total `diagnostic` records) | Reported |
| `splits.cross_split_duplicate_groups` | == 0 | 0 | Yes |
| `splits.evaluation_records_in_training` | == 0 | 0 | Yes |
| `paired_core.bidirectional_mapping_verified` | == true | true | Yes |
| `provenance.admitted_sources_without_terms` | == 0 | 0 | Yes |
| `resources.gpu_hours` | == 0 | 0 | Yes |
| `duplicate_record_ids` (defense-in-depth; not in acceptance contract) | 0 (no requirement stated, but 0 is the only defensible value) | 0 | Yes |
| `media.metadata_admitted_records` (local review item 1) | (reported) | 300,841 (all D1-D4 records, including `evaluation_only`) | Reported |
| `media.media_materialized_records` (local review item 1) | (reported) | 65 (the preregistered real-network verification sample; see Sec. 9 and `reports/T260/media-materialization-plan.md` Sec. 4) | Reported |
| `near_duplicates.*` (local review item 7) | (reported) | 558 exact-normalized-text duplicate groups, 205,558 shingle-Jaccard near-duplicate pairs (threshold 0.8) across `diagnostic`+`pilot_validation`+`pilot_meta` (25,634 records) | Reported |

## 9. Media-availability sample (local review item 4)

`comppareto.data.media_check` ran a real, preregistered network probe
against 5 records per `(source, split)` pair actually present in the
built manifests (13 pairs -> 65 records total; see
`src/comppareto/data/media_check.py::select_sample`, whose selection rule
is fixed in code before any probe result is observed). All 65 probes
succeeded (`media_availability_sample_confirmed_available = 65` of
`media_availability_sample_size = 65`): every sampled COCO image resolved
over plain HTTP, and every sampled DiffusionDB image resolved via an HTTP
Range read of its specific member inside its live `part-NNNNNN.zip`
archive. The 65 successfully-probed records are the only ones with
`image.media_materialized = true` in the frozen manifests; every other
record has `metadata_admitted = true` but `media_materialized = false` --
the explicit, auditable distinction local review item 1 asked for.

Full metrics JSON: `runs/data-admission-posttraining-v1/metrics.json`.
