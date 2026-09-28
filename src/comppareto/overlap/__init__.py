"""T110: seeded PSD quadratic task families across disjoint/partial/full overlap regimes.

Verifies ``comppareto.quadratic.QuadraticTask`` block lifting, Schur-complement
objective changes, and support-locality ("safe-set") relations for randomly
generated multi-task families spanning global dimensions 2-32, per
``tasks/T110-overlap-family.md``'s frozen protocol.
"""

from comppareto.overlap.regimes import (
    OverlapRegime,
    REGIMES,
    RegimeError,
    sample_num_tasks,
    sample_supports,
)
from comppareto.overlap.verify import (
    CaseResult,
    DEFAULT_TOLERANCES,
    OverlapCase,
    TaskCheckResult,
    SafeSetPairResult,
    generate_case,
    verify_case,
)

__all__ = [
    "OverlapRegime",
    "REGIMES",
    "RegimeError",
    "sample_num_tasks",
    "sample_supports",
    "CaseResult",
    "DEFAULT_TOLERANCES",
    "OverlapCase",
    "TaskCheckResult",
    "SafeSetPairResult",
    "generate_case",
    "verify_case",
]
