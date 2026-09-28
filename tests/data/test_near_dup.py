from __future__ import annotations

from comppareto.data.near_dup import (
    JACCARD_THRESHOLD,
    exact_normalized_text_duplicate_groups,
    jaccard,
    near_duplicate_pairs,
)


def _record(record_id: str, split: str, *, caption: str | None = None, prompt: str | None = None) -> dict:
    text: dict = {}
    if caption is not None:
        text["caption"] = caption
    if prompt is not None:
        text["prompt"] = prompt
    return {"record_id": record_id, "split": split, "text": text}


def test_jaccard_identical_sets_is_one() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets_is_zero() -> None:
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_is_one() -> None:
    assert jaccard(set(), set()) == 1.0


def test_exact_normalized_text_duplicate_groups_finds_case_and_whitespace_variants() -> None:
    records = [
        _record("r1", "diagnostic", caption="A Dog Running"),
        _record("r2", "diagnostic", caption="a   dog running"),
        _record("r3", "diagnostic", caption="a red car"),
    ]
    groups = exact_normalized_text_duplicate_groups(records)
    assert groups == [["r1", "r2"]]


def test_exact_normalized_text_duplicate_groups_ignores_out_of_scope_splits() -> None:
    records = [
        _record("r1", "pilot_train", caption="a dog running"),
        _record("r2", "pilot_train", caption="a dog running"),
    ]
    assert exact_normalized_text_duplicate_groups(records) == []


def test_exact_normalized_text_duplicate_groups_skips_empty_text() -> None:
    records = [_record("r1", "diagnostic", caption=""), _record("r2", "diagnostic", caption="")]
    assert exact_normalized_text_duplicate_groups(records) == []


def test_near_duplicate_pairs_flags_high_overlap_prompts() -> None:
    records = [
        _record("r1", "pilot_meta", prompt="a photograph of a red sports car in the rain"),
        _record("r2", "pilot_meta", prompt="a photograph of a red sports car in heavy rain"),
        _record("r3", "pilot_meta", prompt="a watercolor painting of a mountain lake at dawn"),
    ]
    pairs = near_duplicate_pairs(records, threshold=0.5)
    pair_ids = {(a, b) for a, b, _ in pairs}
    assert ("r1", "r2") in pair_ids
    assert not any("r3" in pair for pair in pair_ids)
    for _, _, score in pairs:
        assert score >= 0.5


def test_near_duplicate_pairs_respects_threshold() -> None:
    records = [
        _record("r1", "pilot_validation", caption="a small brown dog on a green lawn"),
        _record("r2", "pilot_validation", caption="a small brown dog on a green lawn today"),
    ]
    loose = near_duplicate_pairs(records, threshold=0.5)
    strict = near_duplicate_pairs(records, threshold=0.99)
    assert len(loose) >= len(strict)


def test_near_duplicate_pairs_ignores_out_of_scope_splits() -> None:
    records = [
        _record("r1", "pilot_train", prompt="a red sports car in the rain"),
        _record("r2", "pilot_train", prompt="a red sports car in the rain today"),
    ]
    assert near_duplicate_pairs(records) == []


def test_default_jaccard_threshold_is_conservative() -> None:
    assert JACCARD_THRESHOLD >= 0.5
