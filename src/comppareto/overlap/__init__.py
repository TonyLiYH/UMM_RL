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
    CheckOutcome,
    DEFAULT_TOLERANCES,
    OverlapCase,
    TaskCheckResult,
    SafeSetPairResult,
    TaskSpec,
    generate_case,
    verify_case,
)
from comppareto.overlap.manifest import build_run_manifest, case_record, config_hash
from comppareto.overlap.sweep import enumerate_case_keys, run_sweep

__all__ = [
    "OverlapRegime",
    "REGIMES",
    "RegimeError",
    "sample_num_tasks",
    "sample_supports",
    "CaseResult",
    "CheckOutcome",
    "DEFAULT_TOLERANCES",
    "OverlapCase",
    "TaskCheckResult",
    "SafeSetPairResult",
    "TaskSpec",
    "generate_case",
    "verify_case",
    "build_run_manifest",
    "case_record",
    "config_hash",
    "enumerate_case_keys",
    "run_sweep",
]
