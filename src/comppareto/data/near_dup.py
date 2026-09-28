"""Near-duplicate text detection beyond exact ``group_key`` equality.

Local review item 7: ``comppareto.data.dedup`` only ever asks "do two
records share the exact same ``group_key``?" -- a real near-duplicate (for
example two DiffusionDB prompts that differ by one adjective, or two LLaVA
conversations phrased almost identically about visually similar images) can
slip through with two *different* group keys and therefore never gets
flagged by ``dedup.cross_split_duplicate_groups``.

This module adds two independent, purely text-based checks, both scoped by
default to ``diagnostic`` / ``pilot_validation`` / ``pilot_meta`` -- the
three splits the review explicitly names, and the three splits small enough
that even the exact, unbounded pairwise check is cheap. (``pilot_train`` is
excluded by default: at hundreds of thousands of rows, an unscoped O(n^2)
Jaccard comparison would not finish in a reasonable time on this task's
CPU-only resource envelope; nothing prevents calling these functions with a
different ``splits`` argument if a future task needs the same check over a
different subset.)

1. :func:`exact_normalized_text_duplicate_groups` -- groups records whose
   primary text field is byte-identical after lowercasing and
   whitespace-collapsing. Catches near-duplicates that differ only in
   capitalization/whitespace/punctuation-adjacent noise.
2. :func:`near_duplicate_pairs` -- a bounded shingle/Jaccard check: every
   record's text is tokenized into overlapping word k-shingles, records are
   bucketed by their lexicographically smallest shingle (a cheap MinHash-lite
   LSH signature), and only records sharing a bucket are compared pairwise.
   This keeps the check far below O(n^2) in the number of records (bucket
   sizes are small in practice) while still catching near-duplicates that
   exact-text matching misses, without ever exhaustively comparing every
   record against every other record.

Neither function depends on any model gradient, loss, or training outcome --
both are pure functions of each record's own frozen ``text`` field.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Iterable

NEAR_DUP_SPLITS: tuple[str, ...] = ("diagnostic", "pilot_validation", "pilot_meta")

#: Two records whose shingle-Jaccard similarity is at or above this
#: threshold are reported as a near-duplicate pair. 0.8 is a conservative
#: (high-precision, willing to miss borderline cases rather than flood the
#: report with loose matches) choice -- two records must share the large
#: majority of their word-trigrams to be flagged.
JACCARD_THRESHOLD = 0.8

#: Shingle width in words.
SHINGLE_K = 3

_WORD_RE = re.compile(r"[a-z0-9]+")


def _record_text(record: dict[str, Any]) -> str:
    """Return the primary human-readable text field for a record.

    Checked in a fixed priority order covering every admitted source's
    schema: COCO/LLaVA captions and DiffusionDB prompts.
    """
    text = record.get("text") or {}
    for key in ("caption", "first_human_turn", "prompt"):
        value = text.get(key)
        if value:
            return str(value)
    return ""


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _shingles(text: str, *, k: int = SHINGLE_K) -> set[str]:
    tokens = _normalize(text).split()
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def exact_normalized_text_duplicate_groups(
    records: Iterable[dict[str, Any]],
    *,
    splits: tuple[str, ...] = NEAR_DUP_SPLITS,
) -> list[list[str]]:
    """Return groups of ``record_id``s sharing identical normalized text.

    Only records whose ``split`` is in ``splits`` are considered. Records
    with empty/missing text are skipped (nothing to compare).
    """
    by_norm: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.get("split") not in splits:
            continue
        norm = _normalize(_record_text(record))
        if not norm:
            continue
        by_norm[norm].append(record["record_id"])
    groups = [sorted(ids) for ids in by_norm.values() if len(ids) > 1]
    return sorted(groups)


def near_duplicate_pairs(
    records: Iterable[dict[str, Any]],
    *,
    splits: tuple[str, ...] = NEAR_DUP_SPLITS,
    threshold: float = JACCARD_THRESHOLD,
    shingle_k: int = SHINGLE_K,
) -> list[tuple[str, str, float]]:
    """Return ``(record_id_a, record_id_b, jaccard_score)`` near-duplicate pairs.

    Bounded via LSH-style bucketing (see module docstring): only records
    that land in the same "smallest shingle" bucket are ever compared, so
    this never performs a full pairwise scan of the scoped record set.
    """
    scoped = [r for r in records if r.get("split") in splits]
    shingles_by_id: dict[str, set[str]] = {}
    bucket_of: dict[str, list[str]] = defaultdict(list)
    for record in scoped:
        text = _record_text(record)
        shingles = _shingles(text, k=shingle_k)
        if not shingles:
            continue
        shingles_by_id[record["record_id"]] = shingles
        signature = min(shingles)
        bucket_of[signature].append(record["record_id"])

    pairs: list[tuple[str, str, float]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for bucket_ids in bucket_of.values():
        if len(bucket_ids) < 2:
            continue
        for i in range(len(bucket_ids)):
            for j in range(i + 1, len(bucket_ids)):
                pair_key = tuple(sorted((bucket_ids[i], bucket_ids[j])))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                score = jaccard(shingles_by_id[pair_key[0]], shingles_by_id[pair_key[1]])
                if score >= threshold:
                    pairs.append((pair_key[0], pair_key[1], score))
    return sorted(pairs)
