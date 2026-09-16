# T260 Split and Decontamination Procedure

## 1. Group keys: the unit of disjointness

Every record carries a `group_key` identifying the underlying media
identity it was derived from -- **not** the record itself. A single COCO
image can back multiple records (several captions, plus zero or more LLaVA
instruction turns); all of them share one `group_key`. Disjointness is
enforced at the group-key level, not the record level, because splitting
individual captions of the same image across splits would leak the
underlying image between splits even if no single caption string repeats.

* D1 (COCO captions) and D2 (LLaVA instructions): `group_key =
  str(int(coco_image_id))`. Both `src/comppareto/data/coco.py` and
  `src/comppareto/data/llava.py` normalize to the same string form (LLaVA's
  `id` field is a zero-padded version of the same integer COCO image id),
  so an image referenced by both sources always hashes to the same bucket.
* D3 (DiffusionDB): `group_key = image_name` (the generated image's own
  UUID-based filename) -- DiffusionDB has no relationship to COCO image
  ids, so its group-key space is independent.

## 2. Training-pool split assignment (diagnostic / pilot_train /
pilot_validation / pilot_meta)

`src/comppareto/data/split.py:assign_split(group_key)`:

1. Hash the group key: `sha256(f"split-bucket::{group_key}")`, take the
   first 8 hex chars as an integer, reduce mod 10,000 -> a bucket in
   `[0, 10000)` (`src/comppareto/data/ids.py:group_bucket`).
2. Map the bucket to a split via fixed cumulative boundaries: `diagnostic`
   0.63% (buckets 0-62), `pilot_validation` 5% (63-562), `pilot_meta` 3%
   (563-862), `pilot_train` 91.37% (863-9999, absorbing any rounding
   remainder). **Correction (2026-09-16)**: an earlier revision of this
   document stated `diagnostic` as "2% (buckets 0-199)" -- that was
   already stale relative to the code even before this round (the
   `diagnostic` share was re-tuned to 0.63%/63 buckets in a prior round
   after the flat-2% share measured 6,350 real records against the task
   file's 512-2,048 ceiling; see `reports/T260/failure-ledger.md`). The
   boundaries above are the real, current values in
   `src/comppareto/data/split.py:SPLIT_FRACTIONS`, re-verified against the
   source file at the time of this correction.

This is a **pure function of the group key alone** -- it never inspects
any model gradient, loss, prediction, or other training outcome, and it is
computed identically regardless of which source (D1, D2, or D3) is asking,
satisfying the task's research-integrity requirement that examples are
never selected after observing model behavior.

**Guarantee produced**: for any two records with the same `group_key`,
`assign_split` returns the same string both times (it is a pure function),
so no group can straddle two training-pool splits. This is verified
unconditionally by `tests/data/test_split.py::test_assign_split_is_deterministic`
and, across real heterogeneous sources, by
`tests/data/test_build.py::test_llava_and_coco_share_image_lands_in_same_split`.

## 3. Evaluation split: sourced, not hashed

`evaluation_only` is **deliberately not produced by `assign_split` at
all**. It is built exclusively from COCO's own official `val2017` caption
file (`src/comppareto/data/coco.py:iter_evaluation_records`), which has
been disjoint from `train2017` since the dataset's original 2017 release --
i.e. by upstream dataset construction, not by any hash function this
project controls. This is intentionally a *stronger* disjointness argument
than a self-computed partition: it does not depend on trusting this
project's own hash function to never collide with itself.

**Scope limitation** (documented honestly rather than silently omitted):
D2 (LLaVA) and D3 (DiffusionDB) do not have an analogous
pre-existing/official held-out benchmark partition available to this task,
so `evaluation_only` currently contains **D4 COCO-val2017-derived records
only**. D2 and D3 content appears only in the training-pool splits
(`pilot_train` / `pilot_validation` / `pilot_meta` / `diagnostic`). This is
a real scope limitation, not a defect being hidden -- see
`reports/T260/failure-ledger.md`.

## 4. Cross-split duplicate detection

`src/comppareto/data/dedup.py:cross_split_duplicate_groups(records)`
groups all emitted records (across every split, every source) by
`group_key` and returns any group key whose records span more than one
split name. The acceptance contract requires this list's length to be
exactly `0`. Because `assign_split` is a pure deterministic function of the
group key, and every D1/D2 source normalizes to the same group-key space,
this is expected to be `0` by construction; `metrics.json` reports the
actual measured count computed from the real built manifests, not an
assumed value.

## 5. Evaluation-in-training leakage detection

`src/comppareto/data/dedup.py:evaluation_records_in_training(records)`
computes the intersection of `evaluation_only` group keys with the union
of all four training-pool splits' group keys. The acceptance contract
requires this to be exactly `0`. Since `evaluation_only` is sourced only
from `val2017` and every training-pool COCO record is sourced only from
`train2017` (a disjoint id space by COCO's own construction), and LLaVA/
DiffusionDB never reference `val2017` at all, this is expected to be `0`
by construction and is verified against the real built manifests before
submission.

## 6. Record-id collision detection

`src/comppareto/data/dedup.py:duplicate_record_ids(records)` independently
checks that no two records (even across sources) were assigned the same
`record_id` -- a defense against a namespace-collision bug in
`stable_id`, orthogonal to the group-key-based checks above.

## 7. Near-duplicate detection beyond exact `group_key` equality (local review item 7, 2026-09-16)

Group-key equality (Secs. 4-6) only catches records that share the exact
same underlying media identity. It does not catch two records with
*different* `group_key`s whose *text* is near-identical or exactly
identical after normalization -- e.g. two DiffusionDB rows with distinct
generated-image UUIDs but the same or a lightly-edited prompt string.
Local review flagged this as a real gap for the diagnostic/meta/
validation splits specifically, since those are the splits most likely
to be used as small, human-inspected quality checks where duplicate or
near-duplicate content silently wastes the sample.

`src/comppareto/data/near_dup.py` adds two bounded (non-`O(n^2)`) checks,
scoped to exactly the three splits local review named --
`diagnostic`, `pilot_validation`, `pilot_meta` (25,634 real records in
total; `pilot_train` and `evaluation_only` are deliberately excluded,
since they are large enough that an `O(n^2)`-adjacent comparison would
be disproportionate, and their role is training volume/held-out
benchmarking rather than small-sample inspection):

1. **Exact-normalized-text duplicate groups**: each record's
   primary text field is lowercased, whitespace-collapsed, and grouped;
   any group with more than one member is an exact-normalized-text
   duplicate group. Real measured count:
   `exact_normalized_text_duplicate_group_count = 558`.
2. **Shingle-Jaccard near-duplicate pairs**: an LSH-style bounded
   comparison -- records are bucketed by their lexicographically smallest
   word-shingle ("MinHash-lite" signature), and only records sharing a
   bucket are ever compared pairwise (never a full `O(n^2)` scan over all
   25,634 records). A pair is reported if its Jaccard similarity over
   word shingles is `>= 0.8` (`near_duplicates.jaccard_threshold` in
   `metrics.json`). Real measured count:
   `shingle_jaccard_near_duplicate_pair_count = 205558`.

Both counts are **reported, not gating** -- the acceptance contract does
not define a required ceiling for either metric, and this task's role is
to surface the real measured signal (which a downstream review/curation
task can act on), not to silently thin the manifests based on a
threshold this task was never asked to choose. Both are real numbers
from the actual frozen `diagnostic`/`pilot_validation`/`pilot_meta`
manifests, in `runs/data-admission-posttraining-v1/metrics.json`'s
`near_duplicates` object.

## 8. D1-paired records within `diagnostic`, reported separately (local review item 8, 2026-09-16)

The task file describes `diagnostic` as containing "paired examples" --
but `diagnostic` is actually a mixed-source split (D1 COCO + D2 LLaVA +
D3 DiffusionDB records all land in it via the same hash-bucket
mechanism), and only the D1 (COCO caption) records are the
bidirectionally-paired (`i2t`/`t2i`) core the paired-examples language
refers to. Reporting only the split's total record count therefore
overstates how many *paired* examples `diagnostic` actually contains.

`metrics.json`'s `paired_core` object now reports both numbers
explicitly: `diagnostic_total_record_count = 1828` (the whole split, all
three sources) and `diagnostic_d1_paired_record_count = 737` (the D1
COCO-caption subset alone, i.e. the actual paired-example count). Both
are within the task file's "512-2,048 paired examples where available"
range read literally against the paired subset (737) and against the
whole split (1,828) -- so this split-out does not create a new failure,
it only makes explicit which of the two readings the real data satisfies
and by how much, rather than leaving that ambiguity to the reader.

## 9. Verification against real data

All checks above are exercised against synthetic fixtures in
`tests/data/` (all tests passing per `reports/T260/result-summary.md`),
and re-run against the real downloaded COCO/LLaVA/DiffusionDB metadata as
part of building the actual frozen manifests -- the real, measured
values are what is written to
`runs/data-admission-posttraining-v1/metrics.json`, not asserted or
assumed values.
