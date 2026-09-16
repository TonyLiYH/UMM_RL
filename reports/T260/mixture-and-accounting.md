# T260 Mixture and Accounting

All counts below are the **real, measured** output of
`comppareto.data.build` against the fully-downloaded raw sources (COCO
`annotations_trainval2017.zip`, LLaVA `llava_instruct_150k.json`,
DiffusionDB `metadata.jsonl`), written to
`runs/data-admission-posttraining-v1/metrics.json`. No number in this
report is an estimate, a projection, or a value asserted before the real
build ran.

## 1. Per-source contribution to the training pool

| Source | Role | Raw entries available | Records admitted into training pool | Admission mechanism |
|---|---|---|---|---|
| `coco_captions_2017` | D1 paired image-caption | 118,287 `train2017` images | 118,287 (1 caption/image, lowest-numbered official annotation id) | All admitted; group-key = COCO `image_id` |
| `llava_instruct_150k` | D2 visual instruction | 157,712 conversation entries | 157,712 (all) | All admitted; group-key = COCO `image_id` (shared with D1) |
| `diffusiondb_2m` | D3 diverse T2I generation | 2,000,000 metadata rows | 31,485 | Deterministic ~2% hash-keep (`group_bucket(image_name, 50) < 1`) **and** NSFW safety filter (`image_nsfw < 0.2`, `prompt_nsfw < 0.5`, missing-score rows excluded) |
| **Total training-pool records** | | | **307,484** | = 118,287 + 157,712 + 31,485 |

**Note:** 307,484 is the pre-split-cap training-pool universe (every
admitted source record, before any per-split assignment). `pilot_train`
specifically is further capped for a git blob-size constraint after split
assignment -- see Sec. 2a. The post-cap grand total actually written
across all four `JSONL_SPLITS` manifests is 176,787 (Sec. 2).

## 2. Split assignment (hash-bucket, pure function of group key)

`pilot_train`'s row below is **post-cap** (see Sec. 2a) -- these are the
real, final counts written to
`configs/data/posttraining-v1/pilot_train.jsonl` and reflected in
`runs/data-admission-posttraining-v1/metrics.json`. The bucket-share
percentages describe hash-space allocation before the Sec. 2a cap is
applied on top; the cap operates *within* `pilot_train`'s own bucket range,
not by changing any split boundary.

| Split | Bucket share (of 10,000) | Record count (measured, post-cap) | coco_captions_2017 | llava_instruct_150k | diffusiondb_2m |
|---|---|---|---|---|---|
| `diagnostic` | 63 buckets (0.63%) | 2,037 | 737 | 980 | 320 |
| `pilot_validation` | 500 buckets (5%) | 15,480 | 6,056 | 7,861 | 1,563 |
| `pilot_meta` | 300 buckets (3%) | 9,242 | 3,583 | 4,729 | 930 |
| `pilot_train` (post-cap) | 9,137 buckets (91.37%), capped to sub-buckets [863, 5727) | 150,028 | 57,517 | 76,985 | 15,526 |
| **Total (post-cap)** | -- | **176,787** | 67,893 | 90,555 | 18,339 |
| *(memo) `pilot_train` pre-cap* | 9,137 buckets (91.37%) | *280,725* | *107,911* | *144,142* | *28,672* |
| *(memo) Total pre-cap* | 10,000 buckets | *307,484* | *118,287* | *157,712* | *31,485* |

**`diagnostic`'s bucket share was deliberately shrunk from a first-pass 2%
(0.02) down to 0.63% (63 of 10,000 buckets).** The task file requires
`diagnostic` to hold "512--2,048 paired examples where available" -- an
*absolute* count ceiling, not a fraction of the training pool. The first
real-data build (fraction 0.02) measured 6,350 diagnostic records, more
than 3x over the 2,048 ceiling, because the training-pool group-key
universe is large. 63 buckets is the largest bucket count that keeps the
real measured diagnostic record count (2,037) at or under 2,048 (64 buckets
measures 2,068, just over). The buckets freed by shrinking `diagnostic` are
absorbed into `pilot_train`'s share (91.37% instead of a plain 90%), not
dropped -- see `reports/T260/failure-ledger.md` for the full measurement
record and `src/comppareto/data/split.py`'s `SPLIT_FRACTIONS` docstring for
the exact derivation.

## 2a. `pilot_train` record-count cap (git blob-size constraint, evidence-backed)

The full, pre-cap `pilot_train.jsonl` (280,725 records) measured
168,907,424 bytes with default JSON formatting, and still 161,091,696
bytes after switching to compact `separators=(",", ":")` -- both over
GitHub's 100MB single-blob push limit, which applies to every blob ever
reachable from a pushed ref's history, not merely the working-tree file at
HEAD. No formatting-only change could bring this under the limit with any
real margin.

The task file explicitly permits meeting "100,000 usable task records or
provid[ing] an evidence-backed reduced target" -- this cap is that
evidence-backed reduced target, not a silent shrink. It is implemented as
`comppareto.data.build.PILOT_TRAIN_BUCKET_CEILING = 5727`: among records
already assigned to `pilot_train` (hash buckets [863, 10000) of the shared
10,000-bucket `group_bucket` space), only those whose own
`group_bucket(group_key) < 5727` are kept; the rest (buckets [5727, 10000),
130,697 records) are dropped. This reuses the exact same deterministic
`group_bucket` hash already used for split assignment -- a pure function of
each record's own `group_key`, never of any model outcome -- applied to a
sub-range, so no new salt or hash function was introduced and no other
split's boundary moved.

**Real measurement** (not extrapolated): the actual rebuilt
`pilot_train.jsonl` under this cap is 150,028 records / 86,150,521 bytes
(86.15MB) -- 50% over the task file's 100,000-record floor, and ~14MB
(14%) of real headroom under the 100MB push limit.

**Mixture preservation check**: the cap changes the internal source
mixture negligibly (see Sec. 3 for the post-cap numbers vs. the pre-cap
38.44% / 51.35% / 10.21% figures above) -- within 0.1-0.14 percentage
points per source, confirming the cap is a uniform thinning of
`pilot_train`, not a source-biased one.

## 3. `pilot_train` internal source mixture

| Source | Records (post-cap) | Share of `pilot_train` (post-cap) | Share (pre-cap, memo) |
|---|---|---|---|
| `coco_captions_2017` (D1) | 57,517 | 38.34% | 38.44% |
| `llava_instruct_150k` (D2) | 76,985 | 51.31% | 51.35% |
| `diffusiondb_2m` (D3) | 15,526 | 10.35% | 10.21% |

D2 (visual instruction) dominates the pool numerically because
LLaVA-Instruct-150K contributes ~1.9 conversation entries per unique image
on average (157,712 entries across 81,479 unique images), while D1
contributes exactly one caption per image by this task's deliberate
`captions_per_image=1` bound (see Sec. 5). This ordering and rough
proportion is preserved after the Sec. 2a cap.

## 4. Evaluation split (`evaluation_only`)

| Source | Records |
|---|---|
| `coco_captions_2017` (D4, from official `val2017`) | 5,000 |

5,000 matches COCO's own official `val2017` image count exactly (one
caption per image, same `captions_per_image=1` bound as D1). D2 and D3
have no representation in `evaluation_only` -- this is a documented scope
limitation, not an omission (`reports/T260/failure-ledger.md`,
`reports/T260/split-and-decontamination.md` Sec. 3).

## 5. Primary vs. fallback mixture decision

* **D1/D4 (COCO captions)**: primary and only candidate; no fallback
  needed. Admitted as originally proposed.
* **D2 (LLaVA-Instruct-150K)**: primary and only candidate; no fallback
  needed. Admitted as originally proposed.
* **D3 (diverse T2I generation)**: the task's original primary candidate,
  **JourneyDB, was rejected** (gated identity-collecting access form +
  explicit no-redistribution Terms of Usage -- see
  `reports/T260/source-license-audit.md`). **DiffusionDB 2M was adopted as
  the fallback**, license-cleared (CC0-1.0, ungated), and is the sole D3
  contributor in every split above. This is the task's one primary/fallback
  substitution, and it is fully documented rather than silently applied.
* **Bound on D1's per-image contribution**: `captions_per_image=1` is a
  deliberate cap (COCO ships ~5 captions/image on average) to keep the
  frozen manifest an auditable, bounded pilot subset rather than absorbing
  the full ~600k-caption corpus; this cap is why D1's record count equals
  the train2017 image count (118,287) rather than a larger multiple of it.

## 6. Acceptance-contract and task-file threshold check (measured values)

| Metric | Required | Measured | Pass? |
|---|---|---|---|
| `pilot.usable_task_records` (== `pilot_train` count) | >= 100,000 usable, or an evidence-backed reduced target | 150,028 (deliberate, evidence-backed reduction from a pre-cap 280,725 -- see Sec. 2a; still 50% over the 100,000 floor) | Yes |
| `diagnostic` record count (task file, not the mechanical acceptance contract) | 512-2,048 | 2,037 | Yes |
| `splits.cross_split_duplicate_groups` | == 0 | 0 | Yes |
| `splits.evaluation_records_in_training` | == 0 | 0 | Yes |
| `paired_core.bidirectional_mapping_verified` | == true | true | Yes |
| `provenance.admitted_sources_without_terms` | == 0 | 0 | Yes |
| `resources.gpu_hours` | == 0 | 0 | Yes |
| `duplicate_record_ids` (defense-in-depth; not in acceptance contract) | 0 (no requirement stated, but 0 is the only defensible value) | 0 (down from 61,916 on the first real-data run before a code fix; see `reports/T260/failure-ledger.md`) | Yes |

Full metrics JSON: `runs/data-admission-posttraining-v1/metrics.json`.
