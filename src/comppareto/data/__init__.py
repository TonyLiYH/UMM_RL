"""Deterministic post-training data admission and manifest builders (T260).

This package audits and freezes manifests for joint post-training data: a
paired image-caption core (D1), a visual-instruction extension (D2), a
diverse text-to-image extension (D3), and diagnostic/evaluation splits (D4).
It never performs model inference, gradient computation, or training; it
only reads source metadata (annotation JSON/parquet files) and emits
deterministic JSON Lines manifests referencing media by canonical path or
URL, never embedding image bytes.

See ``tasks/T260-posttraining-data-admission.md`` and
``configs/data/posttraining-v1/sources.yaml`` for the frozen source list,
license terms, and admission decisions.
"""

from __future__ import annotations
