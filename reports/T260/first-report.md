# T260 First Report -- Post-training Data Admission Plan

Status at time of writing: source/license audit complete for all three
admitted candidates plus the rejected D3 candidate; manifest-builder code
and tests are implemented and passing against synthetic fixtures; real
metadata downloads are in progress but not yet complete. **Total data
pulled into this environment so far is well under 20GB** (see "Downloads so
far" below) -- this report is committed before any further bulk download,
per the task's resource envelope.

## 1. Sources

| Dataset | Role | License | Status |
|---|---|---|---|
| COCO captions 2017 (`cocodataset.org`) | D1 paired image-caption, D4 evaluation | CC BY 4.0 (annotations); images per-photo Flickr license, Consortium disclaims ownership | Admitted |
| LLaVA-Instruct-150K (`liuhaotian/LLaVA-Instruct-150K`) | D2 visual instruction | CC BY 4.0 (also subject to OpenAI ToU for GPT-4-derived text) | Admitted |
| JourneyDB (`JourneyDB/JourneyDB`) | D3 diverse T2I generation (original candidate) | Gated access (identity form) + Terms of Usage forbid redistribution/commercial use | **Rejected** |
| DiffusionDB 2M (`poloclub/diffusiondb`) | D3 diverse T2I generation (fallback, adopted) | CC0-1.0, fully unrestricted, ungated | Admitted (replacement for JourneyDB) |

Full narrative evidence (HF API gate flags, exact Terms-of-Use text quoted,
license field values) is in `reports/T260/source-license-audit.md`.

## 2. Declared sizes and what is actually fetched

| Artifact | Declared size | What this task fetches | Why |
|---|---|---|---|
| `annotations_trainval2017.zip` | 252,907,541 bytes | Full zip (annotation/caption JSON only; ~19GB `train2017.zip` / ~1GB `val2017.zip` image archives are **not** fetched) | Captions are the only content this task's manifests need; a small sample of individual JPEGs is fetched separately for verification (see Sec. 4) |
| `llava_instruct_150k.json` | 228,941,895 bytes | Full file (text only, no images) | It is already the smallest complete unit LLaVA distributes; images are the COCO train2017 files above |
| DiffusionDB `metadata.parquet` (2m random subset) | 194,548,652 bytes | Full metadata table (2,000,000 rows); **no image `part-*.zip` bundles are bulk-fetched** | Metadata alone is sufficient to build deterministic prompt/reference manifests; a range-fetch header check on one part verifies real image addressability without downloading a part |

**Running total of bulk metadata downloads: ~676MB**, far under the 20GB
pre-report ceiling and far under any multi-terabyte full-media download
(which this task never performs).

## 3. Pilot storage plan

All raw downloads live on CQ7 (`/apdcephfs_cq7/share_1447896/yihangli/data/posttraining-v1/`)
per this project's storage rules -- large source files never go under CQ9
`workspace/`. Only small, git-trackable derived artifacts are committed to
this repository:

* `configs/data/posttraining-v1/sources.yaml` -- source registry with
  hashes, license terms, and admission decisions (no raw bytes).
* `configs/data/posttraining-v1/{diagnostic,pilot_train,pilot_validation,pilot_meta}.jsonl`
  -- one JSON object per record, referencing media by relative path/URL
  only.
* `configs/data/posttraining-v1/evaluation_only.yaml` -- the held-out
  COCO-val2017-derived evaluation records, in YAML (not JSONL) per the
  acceptance contract's required-file list.

No image, video, or parquet bytes are ever committed to git.

## 4. Verification-sample plan (media, not manifests)

To confirm official file names/paths in the manifests actually resolve to
real media (without bulk-downloading terabytes of images), this task
fetches only:

* A handful (order of 5-10) of individual COCO `train2017`/`val2017` JPEGs
  by direct HTTPS URL (`http://images.cocodataset.org/train2017/<file_name>`),
  chosen from the frozen manifests' own referenced file names.
* A single ranged HTTP GET against one DiffusionDB `images/part-000001.zip`
  sufficient to read the ZIP local file header for one member (confirming
  the archive and the referenced `image_name` truly exist), not a full part
  download.

These samples are stored under CQ7 `data/posttraining-v1/*/sample_images/`
(outside git) purely as manual verification evidence, referenced by
byte-count/hash in `reports/T260/split-and-decontamination.md` and
`reports/T260/result-summary.md`.

## 5. Manifest schema

Every record (see `src/comppareto/data/records.py`) is a flat JSON object:

```json
{
  "record_id": "<namespace>-<sha256[:16]>",
  "source": "coco_captions_2017 | llava_instruct_150k | diffusiondb_2m",
  "role": "D1_paired | D2_understanding | D3_generation | D4_evaluation",
  "split": "diagnostic | pilot_train | pilot_validation | pilot_meta | evaluation_only",
  "group_key": "<source-shared identity, e.g. COCO image_id>",
  "task_directions": ["i2t", "t2i", "vqa", ...],
  "license_tag": "cc-by-4.0 | cc0-1.0",
  "source_native_id": "<original dataset id>",
  "image": {"source_dataset": "...", "source_relative_path": "..."},
  "text": { "...source-specific fields..." }
}
```

`evaluation_only.yaml` wraps the same record shape in a single YAML
document (`records: [...]`) plus an explicit disjointness guarantee string,
per the acceptance contract's required-file list distinguishing `.jsonl`
splits from the single `.yaml` evaluation reference.

## 6. Stable sample-ID scheme

`record_id = f"{namespace}-{sha256(f'{namespace}::{native_key}')[:16]}"`
(`src/comppareto/data/ids.py:stable_id`) -- a pure function of a
source-specific namespace (e.g. `"d1-coco-caption"`) and the dataset's own
native id (e.g. `"397133:37"` for a COCO image/annotation pair). No
timestamp, random seed, or filesystem enumeration order influences it, so
rebuilding from the same frozen source files reproduces byte-identical IDs.

## 7. Split / decontamination procedure (summary; full detail in
`reports/T260/split-and-decontamination.md`)

* Training-pool splits (`diagnostic` 2%, `pilot_validation` 5%,
  `pilot_meta` 3%, `pilot_train` 90%) are assigned by
  `assign_split(group_key)` (`src/comppareto/data/split.py`), a SHA-256
  hash-bucket function of a group key only -- never of any model
  gradient, loss, or training outcome (research-integrity requirement).
* COCO (D1) and LLaVA (D2) share the *same* group key (the normalized COCO
  `image_id`), so an image referenced by both sources always lands in the
  same split -- this is what guarantees zero cross-split duplicate groups
  by construction, verified programmatically in `tests/data/test_build.py`
  and re-verified on the real data before submission.
* `evaluation_only` is **not** hash-assigned at all: it is sourced
  exclusively from COCO's own official `val2017` partition, which is
  disjoint from `train2017` by the original dataset's own construction --
  strictly stronger evidence of leakage-freedom than any self-computed
  split could provide.
* DiffusionDB (D3) additionally applies a deterministic ~2% hash-keep
  subsample plus an NSFW safety filter, both pure functions of the row's
  own recorded metadata.

## 8. Exact commands used / to be used

```bash
# COCO annotations (bulk metadata only)
curl -sk --retry 5 --retry-delay 3 \
  -o /apdcephfs_cq7/.../coco/annotations/annotations_trainval2017.zip \
  https://images.cocodataset.org/annotations/annotations_trainval2017.zip

# LLaVA-Instruct-150K (complete)
curl -sSL -o /apdcephfs_cq7/.../llava_instruct_150k/llava_instruct_150k.json \
  https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json

# DiffusionDB 2m metadata (complete)
curl -sSL -o /apdcephfs_cq7/.../diffusiondb/metadata.parquet \
  https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet

# Parquet -> JSONL cache (system python3, not .venv, since pandas/pyarrow
# are intentionally not a comppareto dependency)
python3 -m comppareto.data.prep_diffusiondb \
  --input /apdcephfs_cq7/.../diffusiondb/metadata.parquet \
  --output /apdcephfs_cq7/.../diffusiondb/metadata.jsonl

# Manifest build (comppareto's own tested code, .venv)
.venv/bin/python -m comppareto.data.build \
  --coco-zip /apdcephfs_cq7/.../coco/annotations/annotations_trainval2017.zip \
  --llava-json /apdcephfs_cq7/.../llava_instruct_150k/llava_instruct_150k.json \
  --diffusiondb-cache /apdcephfs_cq7/.../diffusiondb/metadata.jsonl \
  --output-dir configs/data/posttraining-v1 \
  --metrics-out runs/data-admission-posttraining-v1/metrics.json
```

## 9. Storage / network requirements

* Disk: ~680MB for raw metadata (CQ7), plus a few MB for the verification
  image sample; committed git artifacts (manifests + reports) are expected
  to total a few MB to a few tens of MB.
* Network: three HTTPS downloads (COCO annotations zip, LLaVA JSON,
  DiffusionDB metadata parquet) plus ~10-15 small individual-file GETs for
  the verification sample. No bulk image/video archive is ever fetched.
* Compute: zero GPU hours. All work is CPU-only JSON/YAML/Parquet
  processing and hashing.

## 10. Downloads so far (at time of writing this report)

| File | Bytes so far | Declared total | Complete? |
|---|---|---|---|
| `llava_instruct_150k.json` | 228,941,895 | 228,941,895 | Yes |
| `diffusiondb/metadata.parquet` | 194,548,652 | 194,548,652 | Yes |
| `coco/annotations/annotations_trainval2017.zip` | ~30,670,848 (in progress) | 252,907,541 | No |

Total pulled so far: **~454MB complete + partial COCO zip**, well under the
20GB ceiling that gates this report's commit.
