"""Pure trainable/frozen parameter classification (T720).

Deliberately independent of any specific model class or of torch itself:
operates on any iterable of ``(name, parameter)`` pairs where ``parameter``
exposes ``.requires_grad: bool`` and ``.numel() -> int`` (the shape torch's
``nn.Module.named_parameters()`` already has). This lets the classification
logic itself be exercised by unit tests using a tiny fake parameter class on
a CPU-only, torch-free development machine, while the real GPU smoke runners
(:mod:`comppareto.adapters.janus_pro_r1.sft_smoke`) feed it genuine
``torch.nn.Parameter`` objects from the loaded ``MultiModalityCausalLM``.

Upstream's own ``train_setup()``
(``vendor/janus-pro-r1/janus-sft/trainer/trainer_t2i.py``) selects the
trainable submodule set by name prefix: ``language_model``, ``gen_embed``,
``gen_head``, ``gen_aligner``, ``aligner`` are trainable; ``vision_model``
and ``gen_vision_model`` are frozen. ``DEFAULT_TRAINABLE_PREFIXES`` mirrors
that split so the adapter's own smoke runner asserts the *same* split
upstream's trainer would have produced, rather than inventing a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

#: Upstream's ``train_setup()`` trainable-submodule name prefixes, in the
#: order they appear in ``trainer_t2i.py``. Any parameter whose dotted name
#: starts with one of these is trainable; every other parameter is frozen.
DEFAULT_TRAINABLE_PREFIXES: tuple[str, ...] = (
    "language_model",
    "gen_embed",
    "gen_head",
    "gen_aligner",
    "aligner",
)


class ParameterLike(Protocol):
    """Structural type matching the subset of ``torch.nn.Parameter`` used here."""

    requires_grad: bool

    def numel(self) -> int: ...


@dataclass
class ParameterGroupStats:
    """Aggregate counts for one classification bucket (trainable or frozen)."""

    n_tensors: int = 0
    n_elements: int = 0
    names: list[str] = field(default_factory=list)


@dataclass
class ParameterInventory:
    """Full trainable/frozen split for one model's ``named_parameters()``."""

    trainable: ParameterGroupStats
    frozen: ParameterGroupStats
    #: Names whose ``requires_grad`` disagreed with the prefix-based
    #: classification -- e.g. a trainable-prefixed parameter that someone
    #: froze, or vice versa. Populated so a mismatch is a visible, checkable
    #: fact rather than a silent divergence between "what the code intended"
    #: and "what actually got optimized".
    requires_grad_mismatches: list[str]

    @property
    def total_elements(self) -> int:
        return self.trainable.n_elements + self.frozen.n_elements

    @property
    def trainable_fraction(self) -> float:
        total = self.total_elements
        if total == 0:
            return 0.0
        return self.trainable.n_elements / total

    def to_dict(self) -> dict[str, object]:
        return {
            "trainable": {
                "n_tensors": self.trainable.n_tensors,
                "n_elements": self.trainable.n_elements,
            },
            "frozen": {
                "n_tensors": self.frozen.n_tensors,
                "n_elements": self.frozen.n_elements,
            },
            "total_elements": self.total_elements,
            "trainable_fraction": self.trainable_fraction,
            "requires_grad_mismatches": list(self.requires_grad_mismatches),
        }


def classify_parameters(
    named_parameters: Iterable[tuple[str, ParameterLike]],
    trainable_prefixes: Sequence[str] = DEFAULT_TRAINABLE_PREFIXES,
) -> ParameterInventory:
    """Split ``named_parameters`` into trainable/frozen groups by name prefix.

    A parameter is classified "trainable" iff its dotted name starts with
    one of ``trainable_prefixes`` (matched against the *first* dotted
    component only is not sufficient in general -- upstream nests e.g.
    ``language_model.model.layers.0...`` -- so this checks
    ``name == prefix or name.startswith(prefix + ".")``, i.e. a proper
    dotted-path prefix match, not a bare string prefix match that could
    accidentally match an unrelated parameter like ``aligner_unrelated``).

    Every parameter's actual ``.requires_grad`` is also inspected;
    disagreements with the prefix-based classification are collected in
    ``requires_grad_mismatches`` rather than silently trusted either way.
    """

    trainable = ParameterGroupStats()
    frozen = ParameterGroupStats()
    mismatches: list[str] = []

    for name, param in named_parameters:
        is_trainable_by_prefix = any(
            name == prefix or name.startswith(prefix + ".") for prefix in trainable_prefixes
        )
        if bool(param.requires_grad) != is_trainable_by_prefix:
            mismatches.append(name)

        bucket = trainable if is_trainable_by_prefix else frozen
        bucket.n_tensors += 1
        bucket.n_elements += int(param.numel())
        bucket.names.append(name)

    return ParameterInventory(trainable=trainable, frozen=frozen, requires_grad_mismatches=mismatches)
