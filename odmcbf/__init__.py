"""Optimal-decay matrix control barrier functions.

Reference implementation of the matrix optimal-decay safety filter and of the
planar two-link singularity-avoidance example used to demonstrate it.
"""

from .filters import (
    DECAY_MODES,
    FilterInfeasible,
    FilterResult,
    SafetyFilter,
    min_scalar_decay,
)
from .simulate import (
    BANDS,
    CONTROLLERS,
    NO_RECOVERY,
    RECOVERY_BETAS,
    Trace,
    band_start,
    hold_at_start,
    recovery_rate,
    rollout,
    run_comparison,
    run_recovery,
    summarize,
)
from .tracking import TurningTarget, inverse_kinematics, tracking_controller
from .two_link import DEFAULT_BOUNDARY, TwoLinkArm, TwoLinkKinematics

__all__ = [
    "DECAY_MODES",
    "FilterInfeasible",
    "FilterResult",
    "SafetyFilter",
    "min_scalar_decay",
    "BANDS",
    "CONTROLLERS",
    "NO_RECOVERY",
    "RECOVERY_BETAS",
    "Trace",
    "band_start",
    "hold_at_start",
    "recovery_rate",
    "rollout",
    "run_comparison",
    "run_recovery",
    "summarize",
    "TurningTarget",
    "inverse_kinematics",
    "tracking_controller",
    "DEFAULT_BOUNDARY",
    "TwoLinkArm",
    "TwoLinkKinematics",
]

__version__ = "1.0.0"
