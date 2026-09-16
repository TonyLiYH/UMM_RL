"""Rollout, token, reward-call, backward, and wall-time counters.

A plain, mutable accounting object threaded through one combined-update step.
It has no opinion about what a "rollout" or "reward call" means semantically —
that is the adapter's job (see :mod:`comppareto.instrumentation.corl_adapter`)
— it only accumulates counts and timings so every combined-update log can
report them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Iterator


@dataclass
class StepCounters:
    rollout_count: int = 0
    token_count: int = 0
    reward_call_count: int = 0
    backward_count: int = 0
    wall_time_seconds: dict[str, float] = field(default_factory=dict)

    def add_rollout(self, *, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        self.rollout_count += 1
        self.token_count += tokens

    def add_reward_call(self) -> None:
        self.reward_call_count += 1

    def add_backward(self) -> None:
        self.backward_count += 1

    @contextmanager
    def timed(self, phase: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.wall_time_seconds[phase] = self.wall_time_seconds.get(phase, 0.0) + elapsed

    def total_wall_time_seconds(self) -> float:
        return sum(self.wall_time_seconds.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "rollout_count": self.rollout_count,
            "token_count": self.token_count,
            "reward_call_count": self.reward_call_count,
            "backward_count": self.backward_count,
            "wall_time_seconds": dict(self.wall_time_seconds),
            "total_wall_time_seconds": self.total_wall_time_seconds(),
        }
