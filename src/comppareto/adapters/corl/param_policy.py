"""Pure-python parameter authorization policy for the CoRL/Janus-Pro GRPO smoke.

Derived from ``corl.open_r1.trainer.grpo_trainer_unified.
JanusProUnifiedGRPOTrainer.init_trainable_parameters`` (see
``configs/corl/admission/discrepancy-lock.yaml`` entry D9): only the shared
``language_model`` backbone is left trainable; every other top-level
submodule (vision understanding branch and image-generation branch) is
explicitly frozen via ``requires_grad = False``.

This module has zero third-party dependencies (stdlib only) so it can be
unit-tested without torch/corl/janus being installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# The single prefix the upstream trainer leaves trainable.
AUTHORIZED_TRAINABLE_PREFIX = "language_model."

# Top-level submodules the upstream trainer explicitly freezes
# (`init_trainable_parameters`). Listed for documentation / cross-checking;
# not required for the authorization predicate itself, which is a pure
# allow-list on AUTHORIZED_TRAINABLE_PREFIX.
EXPECTED_FROZEN_PREFIXES = (
    "vision_model.",
    "aligner.",
    "gen_vision_model.",
    "gen_aligner.",
    "gen_head.",
    "gen_embed.",
)


def is_authorized_trainable(param_name: str) -> bool:
    """True iff a parameter is allowed to be trainable under the frozen protocol."""
    return param_name.startswith(AUTHORIZED_TRAINABLE_PREFIX)


@dataclass
class ParameterAuthorizationSummary:
    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_param_names: int
    frozen_param_names: int
    unauthorized_trainable_names: list[str] = field(default_factory=list)
    all_trainable_are_authorized: bool = True

    def to_dict(self) -> dict:
        return {
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "trainable_param_names": self.trainable_param_names,
            "frozen_param_names": self.frozen_param_names,
            "unauthorized_trainable_names": self.unauthorized_trainable_names,
            "all_trainable_are_authorized": self.all_trainable_are_authorized,
        }


def summarize_authorization(
    named_parameters: Iterable[tuple[str, bool, int]],
) -> ParameterAuthorizationSummary:
    """Classify a stream of (name, requires_grad, numel) into an audit summary.

    Kept decoupled from torch: callers pass plain
    ``(name, requires_grad, numel)`` tuples derived from
    ``model.named_parameters()``.
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    trainable_names = 0
    frozen_names = 0
    unauthorized: list[str] = []

    for name, requires_grad, numel in named_parameters:
        total_params += numel
        if requires_grad:
            trainable_params += numel
            trainable_names += 1
            if not is_authorized_trainable(name):
                unauthorized.append(name)
        else:
            frozen_params += numel
            frozen_names += 1

    return ParameterAuthorizationSummary(
        total_params=total_params,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        trainable_param_names=trainable_names,
        frozen_param_names=frozen_names,
        unauthorized_trainable_names=unauthorized,
        all_trainable_are_authorized=len(unauthorized) == 0,
    )


def count_unauthorized_changes(
    changed_param_names: Iterable[str],
) -> tuple[int, list[str]]:
    """Given the names of parameters whose values changed after an optimizer
    step, return (count_unauthorized, names_unauthorized). A changed
    parameter is "authorized" iff it sits under AUTHORIZED_TRAINABLE_PREFIX.
    """
    unauthorized = [
        name for name in changed_param_names if not is_authorized_trainable(name)
    ]
    return len(unauthorized), unauthorized
