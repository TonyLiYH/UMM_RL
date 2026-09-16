from __future__ import annotations

from comppareto.adapters.corl.param_policy import (
    AUTHORIZED_TRAINABLE_PREFIX,
    count_unauthorized_changes,
    is_authorized_trainable,
    summarize_authorization,
)


def test_authorized_prefix_is_language_model():
    assert AUTHORIZED_TRAINABLE_PREFIX == "language_model."


def test_is_authorized_trainable():
    assert is_authorized_trainable("language_model.model.layers.0.self_attn.q_proj.weight")
    assert not is_authorized_trainable("vision_model.blocks.0.attn.qkv.weight")
    assert not is_authorized_trainable("gen_head.output_mlp_projector.weight")
    assert not is_authorized_trainable("gen_embed.weight")
    assert not is_authorized_trainable("aligner.layers.0.weight")


def test_summarize_authorization_all_ok():
    named = [
        ("language_model.model.embed_tokens.weight", True, 100),
        ("language_model.lm_head.weight", True, 50),
        ("vision_model.blocks.0.weight", False, 30),
        ("gen_head.weight", False, 20),
    ]
    summary = summarize_authorization(named)
    assert summary.total_params == 200
    assert summary.trainable_params == 150
    assert summary.frozen_params == 50
    assert summary.trainable_param_names == 2
    assert summary.frozen_param_names == 2
    assert summary.all_trainable_are_authorized is True
    assert summary.unauthorized_trainable_names == []


def test_summarize_authorization_detects_violation():
    named = [
        ("language_model.model.embed_tokens.weight", True, 100),
        ("gen_head.weight", True, 20),  # violates the frozen protocol
    ]
    summary = summarize_authorization(named)
    assert summary.all_trainable_are_authorized is False
    assert summary.unauthorized_trainable_names == ["gen_head.weight"]


def test_count_unauthorized_changes():
    changed = [
        "language_model.model.layers.0.weight",
        "language_model.lm_head.weight",
        "gen_head.weight",
    ]
    count, names = count_unauthorized_changes(changed)
    assert count == 1
    assert names == ["gen_head.weight"]


def test_count_unauthorized_changes_none():
    changed = ["language_model.model.layers.0.weight"]
    count, names = count_unauthorized_changes(changed)
    assert count == 0
    assert names == []
