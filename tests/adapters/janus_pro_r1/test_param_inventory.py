"""Tests for comppareto.adapters.janus_pro_r1.param_inventory (T720).

Uses a tiny fake parameter class (not torch.nn.Parameter) to exercise the
classification logic on a CPU-only, torch-free development machine -- the
module under test is written to accept anything duck-typed like
``ParameterLike`` (``.requires_grad`` + ``.numel()``), which this fake
satisfies exactly.
"""

from __future__ import annotations

from comppareto.adapters.janus_pro_r1.param_inventory import (
    DEFAULT_TRAINABLE_PREFIXES,
    classify_parameters,
)


class _FakeParam:
    def __init__(self, n_elements: int, requires_grad: bool) -> None:
        self._n_elements = n_elements
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._n_elements


def _janus_pro_like_named_parameters() -> list[tuple[str, _FakeParam]]:
    """A miniature stand-in for MultiModalityCausalLM.named_parameters().

    Mirrors upstream's train_setup() split: language_model / gen_embed /
    gen_head / gen_aligner / aligner trainable; vision_model /
    gen_vision_model frozen. requires_grad is set consistently with that
    split (the "no mismatch" case).
    """

    return [
        ("vision_model.blocks.0.weight", _FakeParam(1000, requires_grad=False)),
        ("vision_model.blocks.1.weight", _FakeParam(1000, requires_grad=False)),
        ("gen_vision_model.encoder.weight", _FakeParam(500, requires_grad=False)),
        ("language_model.model.layers.0.weight", _FakeParam(4000, requires_grad=True)),
        ("language_model.model.layers.1.weight", _FakeParam(4000, requires_grad=True)),
        ("gen_embed.weight", _FakeParam(300, requires_grad=True)),
        ("gen_head.weight", _FakeParam(300, requires_grad=True)),
        ("gen_aligner.layers.0.weight", _FakeParam(200, requires_grad=True)),
        ("aligner.layers.0.weight", _FakeParam(200, requires_grad=True)),
    ]


def test_classify_parameters_matches_upstream_train_setup_split() -> None:
    inventory = classify_parameters(_janus_pro_like_named_parameters())

    assert inventory.trainable.n_tensors == 6
    assert inventory.trainable.n_elements == 4000 + 4000 + 300 + 300 + 200 + 200
    assert inventory.frozen.n_tensors == 3
    assert inventory.frozen.n_elements == 1000 + 1000 + 500
    assert inventory.requires_grad_mismatches == []
    assert inventory.total_elements == inventory.trainable.n_elements + inventory.frozen.n_elements
    assert 0.0 < inventory.trainable_fraction < 1.0


def test_classify_parameters_uses_dotted_prefix_not_bare_string_prefix() -> None:
    # "aligner_unrelated" must NOT be classified trainable just because it
    # starts with the literal characters "aligner" -- only a proper dotted
    # path (name == prefix, or name.startswith(prefix + ".")) counts.
    params = [
        ("aligner_unrelated.weight", _FakeParam(10, requires_grad=False)),
        ("aligner.weight", _FakeParam(20, requires_grad=True)),
        ("aligner", _FakeParam(30, requires_grad=True)),
    ]
    inventory = classify_parameters(params)
    assert inventory.trainable.n_elements == 20 + 30
    assert inventory.frozen.n_elements == 10
    assert inventory.requires_grad_mismatches == []


def test_classify_parameters_flags_requires_grad_mismatches() -> None:
    params = [
        # Trainable-prefixed but someone froze it -- should be flagged, not silently trusted.
        ("language_model.embed.weight", _FakeParam(10, requires_grad=False)),
        # Frozen-prefixed but someone (incorrectly) left it trainable -- should also be flagged.
        ("vision_model.blocks.0.weight", _FakeParam(10, requires_grad=True)),
    ]
    inventory = classify_parameters(params)
    assert inventory.requires_grad_mismatches == [
        "language_model.embed.weight",
        "vision_model.blocks.0.weight",
    ]


def test_empty_input_yields_zeroed_inventory_without_division_by_zero() -> None:
    inventory = classify_parameters([])
    assert inventory.total_elements == 0
    assert inventory.trainable_fraction == 0.0


def test_default_trainable_prefixes_matches_upstream_train_setup() -> None:
    assert DEFAULT_TRAINABLE_PREFIXES == (
        "language_model",
        "gen_embed",
        "gen_head",
        "gen_aligner",
        "aligner",
    )
