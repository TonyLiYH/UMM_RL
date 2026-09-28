# Run note -- `data-admission-posttraining-v1`

Formal T260 admission/data-engineering run. `manifest.json` in this
directory is the schema-conformant record; this note explains what it
summarizes and where the full narrative lives.

This run was re-executed in full on 2026-09-16 in response to local
review's `revision_needed` verdict on the prior round (8 numbered
items -- see `tasks/T260-posttraining-data-admission.md`, "Local review
requirements -- 2026-09-16", and `reports/T260/failure-ledger.md`'s
"[REVISION] Local review (2026-09-16)" entry for the full item-by-item
resolution). Every number below is freshly measured against the
complete real downloaded sources, not carried forward from the prior
round.

## What this run covers

A zero-GPU data-engineering run that audits three joint post-training
data sources, builds deterministic frozen manifests from them, and
measures every split-disjointness/paired-core/media/near-duplicate check
the acceptance contract, the task file, and local review require. No
model was loaded, no inference or training step ran.

1. **Source audit and admission decisions** for COCO captions 2017
   (admitted, D1/D4), LLaVA-Instruct-150K (admitted, D2, with an explicit
   conservative `training_constraints` decision on its GPT-derived
   text), JourneyDB (rejected: gated identity-form access + explicit
   no-redistribution Terms of Usage), and DiffusionDB 2M (admitted as
   the D3 replacement, CC0-1.0, via a redesigned two-stage
   archive-then-row admission filter).
2. **Deterministic manifest build** (`comppareto.data.build`) against
   the complete, real downloaded raw sources, producing
   `configs/data/posttraining-v1/{diagnostic,pilot_validation,pilot_meta}.jsonl`,
   a 7-shard `pilot_train.{jsonl,shard1..6.jsonl}` set plus
   `pilot_train.shards.json` (295,841 training-pool records total, no
   cap applied this round), and `evaluation_only.yaml` (5,000
   COCO-`val2017`-derived D4 records).
3. **Real, measured pass/fail-gate checks**: `cross_split_duplicate_groups
   = 0`, `evaluation_records_in_training = 0`,
   `bidirectional_mapping_verified = true`, `usable_task_records = 270,207`
   (>= the 100,000 floor, uncapped), `admitted_sources_without_terms = 0`,
   `gpu_hours = 0`.
4. **New this round**: real preregistered media-availability network
   probes (`media.media_availability_sample_confirmed_available = 65` of
   65), near-duplicate detection beyond exact `group_key` equality
   (`near_duplicates.exact_normalized_text_duplicate_group_count = 558`,
   `near_duplicates.shingle_jaccard_near_duplicate_pair_count = 205558`),
   and an explicit D1-paired-within-diagnostic count
   (`paired_core.diagnostic_d1_paired_record_count = 737` of
   `paired_core.diagnostic_total_record_count = 1828`).

## Where the full evidence and narrative live

- **Source/license audit** (exact quoted terms, HF API gate flags, the
  LLaVA GPT-terms decision): `reports/T260/source-license-audit.md`.
- **First report** (committed before any bulk download exceeded 20GB):
  `reports/T260/first-report.md`.
- **Split/decontamination methodology** (incl. near-duplicate detection
  and D1-paired reporting): `reports/T260/split-and-decontamination.md`.
- **Real per-split, per-source counts, DiffusionDB fan-out before/after,
  pilot_train sharding, media/near-dup sections**:
  `reports/T260/mixture-and-accounting.md`.
- **Media-materialization plan** (exact retained groups, real
  bytes/hashes, no full-corpus download): `reports/T260/media-materialization-plan.md`.
- **Overall pass/fail summary**: `reports/T260/result-summary.md`.
- **Per-claim evidence mapping, incl. all 8 local review items**:
  `reports/T260/claim-check.md`.
- **Every genuine defect/blocker/scope-limitation, recorded honestly**:
  `reports/T260/failure-ledger.md`.
- **Manifest builders, stable-ID scheme, dedup/split/near-dup/media-check
  logic**: `src/comppareto/data/{coco,llava,diffusiondb,prep_diffusiondb,records,dedup,near_dup,split,ids,media_check,build,run_artifacts}.py`.
- **Tests against synthetic fixtures** (all passing): `tests/data/`
  (`PYTHONPATH=src .venv/bin/python -m pytest -q tests/data/`).
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
manifests/reports/configs are committed.

## Real measured split counts (this round)

| Split | Total | COCO (D1) | LLaVA (D2) | DiffusionDB (D3) |
|---|---|---|---|---|
| `diagnostic` | 1,828 | 737 | 980 | 111 |
| `pilot_validation` | 14,884 | 6,056 | 7,861 | 967 |
| `pilot_meta` | 8,922 | 3,583 | 4,729 | 610 |
| `pilot_train` | 270,207 | 107,911 | 144,142 | 18,154 |
| `evaluation_only` | 5,000 | 5,000 (val2017) | -- | -- |
| **Training-pool total** | **295,841** | **118,287** | **157,712** | **19,842** |

## `pilot_train` sharding (replaces the prior round's record-count cap)

`configs/data/posttraining-v1/pilot_train.shards.json` indexes 7
byte-bounded shards (max 40,000,000 bytes each), totaling 270,207
records / 272,054,471 bytes -- every shard well under GitHub's 100MB
hard limit and 50MB warn threshold. See
`reports/T260/mixture-and-accounting.md` Sec. 4 for the per-shard table.

## Known limitations (documented, not silently omitted; carried forward, unchanged unless noted)

1. **`evaluation_only` covers D1/D4 (COCO) only.** D2 (LLaVA) and D3
   (DiffusionDB) have no pre-existing, officially-disjoint held-out
   partition this task adopted. See
   `reports/T260/split-and-decontamination.md` Sec. 3.
2. **LLaVA-Instruct-150K's GPT-4-derived conversation text** now carries
   an explicit, conservative `training_constraints.restricted_as_training_target
   = true` field on every D2 record (upgraded this round from a vague
   prose caveat; see `reports/T260/source-license-audit.md`).
3. **COCO's `train2017.zip`/`val2017.zip` `ETag` values are S3
   multipart-upload hashes**, not plain content hashes -- an honest gap
   recorded in `reports/T260/media-materialization-plan.md` Sec. 2
   rather than a fabricated content hash.

## Changes this round, found and fixed by re-running against real data

See `reports/T260/failure-ledger.md`'s "[REVISION] Local review
(2026-09-16)" entry for the full item-by-item record of: the
DiffusionDB archive-fan-out redesign, the `pilot_train` sharding, the
real media-availability probes, the near-duplicate-detection addition,
the LLaVA training-constraints decision, the D1-diagnostic-paired
separate reporting, and a drive-by correction of a pre-existing stale
"diagnostic 2%" claim in `split-and-decontamination.md`.

## Status

`pass` -- every acceptance-contract metric and every task-file numeric
requirement, plus all 8 local review items, is satisfied by real,
measured values from the complete downloaded sources
(`runs/data-admission-posttraining-v1/metrics.json`):
`pilot.usable_task_records = 270207` (>= 100,000, uncapped),
`splits.cross_split_duplicate_groups = 0`,
`splits.evaluation_records_in_training = 0`,
`paired_core.bidirectional_mapping_verified = true`,
`paired_core.diagnostic_d1_paired_record_count = 737` (of
`diagnostic_total_record_count = 1828`),
`provenance.admitted_sources_without_terms = 0`,
`media.media_availability_sample_confirmed_available = 65` (of 65),
`near_duplicates.exact_normalized_text_duplicate_group_count = 558`,
`near_duplicates.shingle_jaccard_near_duplicate_pair_count = 205558`,
`resources.gpu_hours = 0`. No model training/inference/method
comparison was performed at any point in this task.
