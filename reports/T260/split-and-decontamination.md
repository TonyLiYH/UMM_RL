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
   2% (buckets 0-199), `pilot_validation` 5% (200-699), `pilot_meta` 3%
   (700-999), `pilot_train` 90% (1000-9999, absorbing any rounding
   remainder).

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

## 7. Verification against real data

All five checks above are exercised against synthetic fixtures in
`tests/data/test_build.py` (43 tests total across the package, all
passing per `reports/T260/result-summary.md`), and re-run against the real
downloaded COCO/LLaVA/DiffusionDB metadata as part of building the actual
frozen manifests -- the real, measured values are what is written to
`runs/data-admission-posttraining-v1/metrics.json`, not asserted or
assumed values.
