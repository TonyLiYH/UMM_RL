# Run note -- `data-admission-posttraining-v1`

Formal T260 admission/data-engineering run. `manifest.json` in this
directory is the schema-conformant record; this note explains what it
summarizes and where the full narrative lives.

## What this run covers

A zero-GPU data-engineering run that audits three joint post-training data
sources, builds deterministic frozen manifests from them, and measures
every split-disjointness/paired-core check the acceptance contract and the
task file require. No model was loaded, no inference or training step ran.

1. **Source audit and admission decisions** for COCO captions 2017
   (admitted, D1/D4), LLaVA-Instruct-150K (admitted, D2), JourneyDB
   (rejected: gated identity-form access + explicit no-redistribution
   Terms of Usage), and DiffusionDB 2M (admitted as the D3 replacement,
   CC0-1.0).
2. **Deterministic manifest build** (`comppareto.data.build`) against the
   complete, real downloaded raw sources -- not a sample, not a synthetic
   fixture -- producing `configs/data/posttraining-v1/{diagnostic,pilot_train,pilot_validation,pilot_meta}.jsonl`
   (176,787 training-pool records total, post-split-cap -- see "Known
   transport limitation" below) and `evaluation_only.yaml` (5,000
   COCO-`val2017`-derived D4 records).
3. **Real, measured pass/fail-gate checks**: `cross_split_duplicate_groups
   = 0`, `evaluation_records_in_training = 0`,
   `bidirectional_mapping_verified = true`, `usable_task_records = 150,028`
   (>= the 100,000 floor; a deliberate, evidence-backed reduction from a
   pre-cap 280,725 -- see below), `admitted_sources_without_terms = 0`,
   `gpu_hours = 0`.

## Where the full evidence and narrative live

- **Source/license audit** (exact quoted terms, HF API gate flags):
  `reports/T260/source-license-audit.md`.
- **First report** (committed before any bulk download exceeded 20GB):
  `reports/T260/first-report.md`.
- **Split/decontamination methodology**: `reports/T260/split-and-decontamination.md`.
- **Real per-split, per-source counts and proportions, and the
  primary/fallback mixture decision**: `reports/T260/mixture-and-accounting.md`.
- **Overall pass/fail summary**: `reports/T260/result-summary.md`.
- **Per-claim evidence mapping**: `reports/T260/claim-check.md`.
- **Every genuine defect/blocker/scope-limitation, recorded honestly**:
  `reports/T260/failure-ledger.md` -- including two real code defects
  found by running against real data and fixed this task (a LLaVA
  `record_id` collision bug affecting 61,916 ids, and a `diagnostic`
  split-size overshoot of the task file's 512-2,048 ceiling), plus the
  JourneyDB rejection, the COCO/DiffusionDB evaluation-coverage scope
  limitation, and a download-stall/parallel-range-resume operational
  saga.
- **Manifest builders, stable-ID scheme, dedup/split logic**:
  `src/comppareto/data/{coco,llava,diffusiondb,prep_diffusiondb,records,dedup,split,ids,build,run_artifacts}.py`.
- **44 tests against synthetic fixtures** (all passing):
  `tests/data/` (`PYTHONPATH=src .venv/bin/python -m pytest -q tests/data/`).
- **Source registry with hashes/terms/admission decisions**:
  `configs/data/posttraining-v1/sources.yaml`.

## Raw source provenance (real, hash-verified)

| Raw source | Bytes | sha256 |
|---|---|---|
| COCO `annotations_trainval2017.zip` | 252,907,541 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| LLaVA `llava_instruct_150k.json` | 228,941,895 | `6b68bc5ca2bfd8a71119af0e8454929668ccda6a334955ccc95d114fc8d082fa` |
| DiffusionDB `metadata.parquet` | 194,548,652 | `eecd341187bc91c07f5994ad0660d40228ea025616fd57a509bef8323677c68f` |
| DiffusionDB `metadata.jsonl` (derived cache) | 799,203,140 | `8723784619bd3caaafe71997c5bdbd4736df60c869554318a7b6df39be120f7b` |

All raw source files live on CQ7
(`/apdcephfs_cq7/share_1447896/yihangli/data/posttraining-v1/`), outside
git, per this project's storage rules; only the small derived
manifests/reports/configs above are committed.

## Known limitations (documented, not silently omitted)

1. **`evaluation_only` covers D1/D4 (COCO) only.** D2 (LLaVA) and D3
   (DiffusionDB) have no pre-existing, officially-disjoint held-out
   partition this task adopted; a self-computed hash-based evaluation
   split for them was deliberately not substituted, since it would be
   weaker evidence than COCO's upstream-guaranteed `val2017` disjointness.
   See `reports/T260/split-and-decontamination.md` Sec. 3 and
   `reports/T260/failure-ledger.md`.
2. **LLaVA-Instruct-150K's GPT-4-derived conversation text also carries an
   OpenAI Terms-of-Use caveat** (beyond the dataset's own CC BY 4.0
   release) for any downstream training use -- documented, not
   adjudicated further by this task.
3. **DiffusionDB image bytes were never fetched** (only the metadata
   table); manifests reference `image_name` only. A range-fetch
   verification of one part archive's local file header was planned in
   `first-report.md` Sec. 4 as a lightweight addressability check, in
   addition to the real train2017/val2017 JPEG samples that were fetched
   for COCO.

## Two real defects found by running against real data, and fixed

1. **LLaVA `record_id` collisions** (61,916 measured on the first
   real-data build) -- root cause: LLaVA's `id` field identifies the
   source image, not a unique conversation, and the real file has
   multiple conversations per image. Fixed in
   `src/comppareto/data/llava.py` by appending a deterministic,
   file-order-based per-id occurrence index before hashing into
   `record_id`. Re-measured: `duplicate_record_ids = 0`.
2. **`diagnostic` split exceeded the task file's 512-2,048 record
   ceiling** (6,350 measured on the first real-data build, using a flat
   2% bucket share). Fixed in `src/comppareto/data/split.py` by shrinking
   `diagnostic`'s bucket share to 0.63% (the largest share keeping the
   real measured count at/under 2,048); freed buckets are absorbed into
   `pilot_train`. Re-measured: `diagnostic = 2,037`.

Full detail on both: `reports/T260/failure-ledger.md`.

## Known transport limitation, resolved (not a data defect)

`configs/data/posttraining-v1/pilot_train.jsonl` originally exceeded
GitHub's 100MB per-file push limit at 161,091,696 bytes (280,725 records)
even after switching `write_manifests` to compact JSON separators. **This
was resolved**, not left open: `src/comppareto/data/build.py` now applies
`PILOT_TRAIN_BUCKET_CEILING = 5727`, a deterministic, evidence-backed
record-count reduction (same `group_bucket` hash already used for split
assignment, restricted to `pilot_train`'s own records) that yields the
real, final `pilot_train.jsonl` at 150,028 records / 86,150,521 bytes
(86.15MB) -- comfortably under the 100MB limit and still 50% over the task
file's 100,000-record floor, per the task file's own explicit allowance
for "an evidence-backed reduced target." The oversized blob had existed
in earlier unpushed commits, so that local history was rewritten (`git
reset --soft` back to the previously-pushed tip, then clean recommits of
the corrected state) before pushing. Full analysis:
`reports/T260/failure-ledger.md`, `reports/T260/mixture-and-accounting.md`
Sec. 2a.

## Status

`pass` -- every acceptance-contract metric and every task-file numeric
requirement is satisfied by real, measured values from the complete
downloaded sources (`runs/data-admission-posttraining-v1/metrics.json`):
`pilot.usable_task_records = 150028` (>= 100,000; a deliberate,
evidence-backed reduction from a pre-cap 280,725),
`splits.cross_split_duplicate_groups = 0`,
`splits.evaluation_records_in_training = 0`,
`paired_core.bidirectional_mapping_verified = true`,
`provenance.admitted_sources_without_terms = 0`,
`resources.gpu_hours = 0`. No model training/inference/method comparison
was performed at any point in this task.
