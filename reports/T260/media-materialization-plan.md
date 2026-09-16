# T260 Media-Materialization Plan

Local review item 2: "Produce a media-materialization plan for the EXACT
retained groups (archive/part IDs, expected bytes, destination paths,
hashes) without downloading full source corpora."

This report is a **plan**, not an execution log: it enumerates exactly
which archives/files would need to be fetched to fully materialize every
group actually retained in the frozen manifests, with real (not
estimated) expected bytes and, where the upstream host publishes one
without requiring a download, a real content hash. Section 4 records the
small number of individual bytes this task *did* actually fetch, as a
live end-to-end test that the addressing scheme below is correct, not as
partial execution of the plan itself. No step below was executed as part
of this task beyond that small verification sample; zero bulk media
bytes were downloaded.

## 1. DiffusionDB (D3) -- archive-level plan, 25 of 2,000 parts

Every D3 record's `image.source_relative_path` has the shape
`images/part-NNNNNN.zip::<image_name>` (`src/comppareto/data/diffusiondb.py`).
Because DiffusionDB only exposes generated images bundled inside per-part
zip archives (never as individually-addressable files), materializing
every retained D3 group requires downloading the **whole archive** for
every part that contains at least one retained row. The two-stage filter
in `diffusiondb.py` (see its module docstring and
`reports/T260/mixture-and-accounting.md` Sec. "DiffusionDB archive
fan-out") selects parts by a pure hash of the part's own id
(`group_bucket(f"diffusiondb-part::{part_id}", num_buckets=2000) < 32`),
independent of row content -- so the set of archives to fetch is fixed
and known *before* materialization, exactly what this plan enumerates.

Real per-archive size and content hash below were obtained from the
Hugging Face Hub's tree API (`GET
/api/datasets/poloclub/diffusiondb/tree/main/images`), which returns each
LFS-backed file's real `size` and `lfs.oid` (a genuine sha256 of the
archive's exact bytes, computed by the Hub at upload time) **without
downloading any of the file's content** -- confirmed live this round (two
paginated calls, 1,000 entries each, 2,000 total, matching
`PART_MODULUS = 2000` exactly).

| # | part_id | Destination path (CQ7) | Expected bytes | sha256 (real, via HF tree API) |
|---|---|---|---|---|
| 1 | 86 | `diffusiondb/images/part-000086.zip` | 696,060,001 | `9cfdbe756abac712633d303747b506687d07a3e5bd29b5eec6a315b081609d94` |
| 2 | 252 | `diffusiondb/images/part-000252.zip` | 696,061,104 | `39f412f7d8e696d20aaa53eb4a82342fbe0db230f92a8cf6d0744766edef5fcb` |
| 3 | 290 | `diffusiondb/images/part-000290.zip` | 666,711,078 | `fc8be3c81464ecbd033fabd4030d2c963766986fb208240acec91038d58967e1` |
| 4 | 293 | `diffusiondb/images/part-000293.zip` | 732,235,404 | `f21104f373ead853401693ba65516ee4d370beb4e72d9fff79ee62d9126b60e9` |
| 5 | 295 | `diffusiondb/images/part-000295.zip` | 701,713,820 | `e9b5ebdab963a4d0ec15a7dd513499f1a049061365fa58d8fe46401bcdd335cb` |
| 6 | 369 | `diffusiondb/images/part-000369.zip` | 672,402,355 | `fb5762cd1b3ed721055b535c54dffba47fe3dbc5071d517e2db91cfd7e1903ae` |
| 7 | 568 | `diffusiondb/images/part-000568.zip` | 616,765,044 | `de2e239bf9ef2d5666cba2d71df2a8dce64c2ea811efdc9602658e2f29effb15` |
| 8 | 599 | `diffusiondb/images/part-000599.zip` | 587,944,731 | `a1f4feb2e5ad3afa381200d5d3ccdfad0558ae90422e8f431b8b32c7ffe6a89f` |
| 9 | 722 | `diffusiondb/images/part-000722.zip` | 692,470,792 | `1b864119c5a82f5e755c75cfe18839f0022bdcca997632296dfdaf9233520c93` |
| 10 | 777 | `diffusiondb/images/part-000777.zip` | 617,724,531 | `97ad9766598d6ccda798f8ec99f885c4c28507bfeea3d301923157127149af7d` |
| 11 | 892 | `diffusiondb/images/part-000892.zip` | 628,146,715 | `6a0f0cdef9aa91c8665313fbdc565d3f2eb4da69c8efb6511fb34daf04e22fd1` |
| 12 | 901 | `diffusiondb/images/part-000901.zip` | 580,821,522 | `5d7014fc507cea576d8b53b3798a2105425140835cc57d66925f4eac69cd593d` |
| 13 | 905 | `diffusiondb/images/part-000905.zip` | 585,993,434 | `100fd698cc3808c9aa7d4806880d3e1b29641156e818f99594fc9d61c94101bc` |
| 14 | 924 | `diffusiondb/images/part-000924.zip` | 584,775,095 | `3e3fbc7cab7761b989bf6392b402891e43ed1f459418aea3b238aa5c5e585644` |
| 15 | 1086 | `diffusiondb/images/part-001086.zip` | 617,435,967 | `f751093b9b39e370282d1d6cd7a58d8449b58dbeb658803913a5abf2ce5a73e7` |
| 16 | 1217 | `diffusiondb/images/part-001217.zip` | 534,629,972 | `2be873e2cee8e1233aa7d618f25c2edaa1b6308fa9e9949c3924ddc100dc7f95` |
| 17 | 1362 | `diffusiondb/images/part-001362.zip` | 646,220,431 | `92efa1fd0d38a976b3f04c74bd6fe4eb61468de00d1dbff3d15f93fb2dd4cf1f` |
| 18 | 1372 | `diffusiondb/images/part-001372.zip` | 580,425,759 | `19553522d2b57f5e4a2cbdad55587232d18858a2cb96eae94d6988331c6ca301` |
| 19 | 1380 | `diffusiondb/images/part-001380.zip` | 589,825,193 | `01a1fc69e3a30018390f7d5bf0ed45f29d14944067dadb4f23438f2f72d6a22f` |
| 20 | 1668 | `diffusiondb/images/part-001668.zip` | 615,074,693 | `c2c7e80bf9e966a05edcefcaf9b242399f1834b5d3f91c955e46b113b74e4d62` |
| 21 | 1700 | `diffusiondb/images/part-001700.zip` | 555,496,340 | `f123b7ff6bb091d32fc5febcb8ec928e92c27293a6a3be0c512d29c4238b6244` |
| 22 | 1750 | `diffusiondb/images/part-001750.zip` | 624,911,012 | `a48cc1bdfa826d2a4a5454f7d2d60fd7341a63c97c6099939d617169debc0d30` |
| 23 | 1778 | `diffusiondb/images/part-001778.zip` | 566,794,078 | `ec5f505479aa881cef8a5e82a41a9906a6907400c8a69de010b0abc3fa42faed` |
| 24 | 1798 | `diffusiondb/images/part-001798.zip` | 537,546,495 | `37ace4cf9f7aeea6de3944b2cbecf662e4ab24ccfaf8e179ad4a6872c63a79a8` |
| 25 | 1907 | `diffusiondb/images/part-001907.zip` | 659,788,368 | `ba1867efca807f5e3405c3b8df83aa0064be874aab2dd628ca5983cbf5593e37` |
| **Total** | 25 parts | `/apdcephfs_cq7/share_1447896/yihangli/data/posttraining-v1/diffusiondb/images/` | **15,587,973,934 (~15.6 GB)** | -- |

Every row's size is real (`lfs.size` from the Hub, matching the
already-recorded per-part sizes used to derive `PART_KEEP_THRESHOLD = 32`
in `diffusiondb.py`), not estimated. All 25 downloads are independently
well under this task's 20GB single-download-step ceiling (largest single
part: 732,235,404 bytes, part-000293); the 15.6GB aggregate is also under
20GB, so this plan's entire D3 media set could in principle be fetched as
25 individually-bounded steps totaling less than one 20GB step's budget,
if a future execution round chose to run it. After fetching each archive,
each retained image inside it would be extracted (or Range-read
individually, as the verification probe below does) and re-hashed
locally against the sha256 the manifest would then record in
`image.media_sha256` -- the archive-level `lfs.oid` above only verifies
the archive's own bytes, not any individual member.

## 2. COCO (D1/D2/D4) -- individually-addressable, no archive fan-out

Unlike DiffusionDB, every COCO image referenced by a D1, D2 (LLaVA), or
D4 record is individually fetchable by its own official path
(`http://images.cocodataset.org/train2017/<file_name>` or
`.../val2017/<file_name>`) -- there is no zip-bundle fan-out problem for
this source at the per-image level. The practical planning unit is
therefore the *official bulk zip* COCO itself publishes per split (the
only way to get every image without ~118K/~5K individual HTTP round
trips), whose declared sizes were confirmed via a real `HEAD` request
this round (no bytes downloaded):

| Group | Destination path (CQ7) | Expected bytes (real `Content-Length`) | Content hash |
|---|---|---|---|
| `train2017` (all COCO D1/D2 retained images) | `coco/images/train2017.zip` | 19,336,861,798 | `ETag "62ff7d7fbcc7e0c0604cbb0f9047ce77-2306"` -- an S3 **multipart-upload** ETag (2,306 parts), not a plain MD5/sha256 of the full file; COCO does not publish a single-hash checksum for this archive. Honest gap, not fabricated -- a real sha256 would have to be computed locally after a full download. |
| `val2017` (all COCO D4 evaluation images) | `coco/images/val2017.zip` | 815,585,330 | `ETag "d366be60d3dc737327160d62453e3973-98"` (same caveat, 98 parts) |

**Flag, not a violation**: `train2017.zip` at 19,336,861,798 bytes
(~19.34 GB) is under this task's 20GB single-download-step ceiling, but
only by ~660MB of headroom -- the tightest margin of any group in this
plan. This task does **not** execute this download (it would materialize
essentially the entire D1/D2 COCO image group, since `captions_per_image
= 1` retains a caption for every `train2017` image); it is recorded here
as the exact, real expected size for a future execution round to budget
against, consistent with "materialize only actually-retained media or
produce a plan" -- every COCO D1/D2 record's group is, in fact, retained
(COCO's images are not subsampled the way DiffusionDB's are), so the
honest plan for "the exact retained groups" for this source is
necessarily the (near-)complete `train2017` archive, not a small subset.
`val2017.zip` (815,585,330 bytes, ~816MB) is comfortably bounded and
covers all 5,000 `evaluation_only` records.

LLaVA-Instruct-150K (D2) references these same `train2017` files by COCO
image id (`src/comppareto/data/llava.py`); it introduces no separate
image-media surface or archive to plan for.

## 3. Aggregate

| Source | Archives/files to fetch | Total expected bytes | Largest single download step |
|---|---|---|---|
| DiffusionDB (D3) | 25 of 2,000 part archives | 15,587,973,934 (~15.6 GB) | 732,235,404 (part-000293, ~0.73 GB) |
| COCO train2017 (D1/D2) | 1 archive | 19,336,861,798 (~19.34 GB) | 19,336,861,798 (~19.34 GB, at the edge of the 20GB ceiling) |
| COCO val2017 (D4) | 1 archive | 815,585,330 (~0.82 GB) | 815,585,330 |
| **Grand total** | 27 archives | **35,740,421,062 (~35.7 GB)** | -- |

No single step in this plan exceeds 20GB; the DiffusionDB portion in
particular is dramatically smaller than the prior round's 1,999-archive/
1.24TB fan-out (see `reports/T260/mixture-and-accounting.md`). The COCO
`train2017` step is real and large because COCO's own images are not
subsampled by this task (only LLaVA/DiffusionDB rows are) -- an honest
consequence of D1/D2's design, not an oversight.

## 4. Real end-to-end verification sample (executed this round, not part of the plan above)

To confirm the addressing scheme in Sections 1-2 is actually correct --
not just plausible on paper -- `src/comppareto/data/media_check.py`'s
real probe functions were run live against both transports:

* **COCO**: `probe_coco_image("train2017/000000000009.jpg")` ->
  `{"available": true, "bytes": 224297, "sha256":
  "35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99",
  "probe_url": "http://images.cocodataset.org/train2017/000000000009.jpg"}`.
* **DiffusionDB**: `probe_diffusiondb_image("images/part-000086.zip::272f29e8-1872-4fa4-bca5-1dfb68ed2108.png")`
  -> `{"available": true, "bytes": 541584, "sha256":
  "8d558e79b8ed45e4c9dbacfe75fd90df7b560851bdcc9ddde404401ff15de523",
  "probe_url":
  "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-000086.zip"}`
  -- confirms the `HTTPRangeFile` + stdlib `zipfile` approach correctly
  reads one member out of a live 696,060,001-byte remote archive via HTTP
  Range requests (through Hugging Face's redirect-to-CDN chain) without
  fetching the other members or the archive as a whole.

Both probes are also part of the frozen manifests' preregistered
media-availability sample (`comppareto.data.build.run_media_availability_sample`,
local review item 4) -- see `runs/data-admission-posttraining-v1/metrics.json`'s
`media` section and the raw probe results this round wrote to
`/tmp/t260work/media_probes_real.json` (not committed; CQ9 tmp/ scope,
reproducible by rerunning `comppareto.data.build` with `--media-check
run`).

## 5. What this plan is not

This plan does not download, extract, or commit any image, video, or
archive byte to this repository or to git. It records exactly what a
future, explicitly-scoped execution round would need to fetch (paths,
byte counts, and, for DiffusionDB, real per-archive content hashes
obtained without downloading), so that round can budget network/storage
without re-deriving the retained-group set from scratch.
