"""The experiments of Section VIII of the paper (singularity avoidance).

Every figure and number in that section comes from one of three closed-loop
experiments on the two-link arm of :mod:`odmcbf.two_link`. Each one runs the standard
MCBF, scalar OD and matrix OD filters of :mod:`odmcbf.filters` with the settings in
:data:`FILTER`, the class-K term ``alpha(H) = H`` (:data:`GAMMA`), the tracking
controller of :mod:`odmcbf.tracking` with gain :data:`GAIN`, and the time step
:data:`DT`.

``hero``
    Fig. 1. The arm :data:`HERO_ARM` tracks a target that turns about joint 1 at
    1 rad/s and is pushed 30% of the way into the outer band, where ``H`` is
    violated, following :data:`TIMELINE`. This run is also the first of the six
    tracking cases.
``tracking``
    Fig. 2 and the numbers in Sec. VIII-C. The same timeline on the arms
    ``l2 = 0.75, 1, 1.25`` (with ``l1 = 1`` and ``sigma_bar = 0.6``), with the target
    pushed into either the outer or the inner band; see :data:`TRACKING_CASES`.
``recovery``
    Fig. 3 and the recovery paragraph of Sec. VIII-C. The hero arm starts at rest
    30% of the way into the outer band, with its target fixed at the start. It runs
    matrix OD with the recovery constraint at ``beta = 1/2`` and ``1/4``, and once
    without it; see :data:`RECOVERY`.

``scripts/reproduce_paper.py`` runs all three and writes the paper's figure files.
``odmcbf/tests/test_paper.py`` checks the numbers quoted in the section against
:func:`run_tracking` and :func:`run_recovery_case`.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from .simulate import BANDS, CONTROLLERS, Trace, rollout, run_recovery
from .tracking import TurningTarget, tracking_controller
from .two_link import TwoLinkArm

#: The hero arm of Fig. 1 and Fig. 3: ``l1 = 1``, ``l2 = 0.75``, ``sigma_bar = 0.6``, so that the
#: safe set is the annulus ``0.918 <= |p| <= 1.511`` and the standard MCBF certifies the rotation
#: of the whole arm about joint 1 only up to 0.823 rad/s (eq. (37) of the paper).
HERO_ARM = dict(l1=1.0, l2=0.75, sigma_bar=0.6)
#: Link 1 and the threshold shared by every arm of the section.
L1, SIGMA_BAR = HERO_ARM["l1"], HERO_ARM["sigma_bar"]
#: Rate of the class-K term ``alpha(H) = gamma H``. The paper uses ``alpha(lambda) = lambda``.
GAMMA = 1.0
#: The OD filters' settings: ``omega_d = 1`` (``Omega_d = I/2``), the matched penalties
#: ``rho_omega = 0.005`` and ``rho_Omega = (4/p) rho_omega = 0.01``, and the recovery constraints
#: ``omega >= omega_d`` and ``sym(Omega) >= beta I`` with ``beta = omega_d / 2``.
FILTER = dict(omega_d=1.0, rho_omega=5e-3, recovery=True)
#: Gain ``kappa`` of the nominal tracking controller, 1/s.
GAIN = 5.0
#: Time step in seconds. Each command is held for ``DT``.
DT = 0.01
#: Initial configuration ``(theta1, theta2)``; the target starts at its end effector.
THETA0 = np.array([0.0, math.pi / 2])
#: The target's timeline, in seconds and rad/s. The target turns about joint 1 at ``rate`` for
#: the whole run. After ``turn`` seconds, its distance from joint 1 moves over ``move`` seconds to
#: ``depth`` of a band's width beyond the boundary of the safe set. It stays there, then returns
#: over another ``move`` seconds, arriving ``return_end`` seconds before the run ends at ``duration``.
TIMELINE = dict(rate=1.0, depth=0.3, turn=1.0, move=0.75, duration=4.5, return_end=0.5)
#: The six tracking cases of Sec. VIII-C: link 2 and the band the target enters. The first is the hero run.
TRACKING_L2 = (0.75, 1.0, 1.25)
TRACKING_CASES: Tuple[Tuple[float, str], ...] = tuple((l2, band) for l2 in TRACKING_L2 for band in BANDS)
#: The recovery experiment of Fig. 3: :func:`odmcbf.simulate.run_recovery` on the hero arm.
RECOVERY = dict(band="outer", depth=0.3, gain=GAIN, dt=DT, horizon=5.0, betas=(0.5, 0.25))
#: Window, in seconds, over which "held" quantities are averaged: the last 2 s before the target
#: starts back, once it has arrived in the band. See :func:`held_window`.
SETTLE = 2.0


def arm(l2: float = HERO_ARM["l2"]) -> TwoLinkArm:
    """The arm of the section with link 2 of length ``l2``: ``l1 = 1``, ``sigma_bar = 0.6``, ``gamma = 1``."""
    return TwoLinkArm(L1, l2, gamma=GAMMA, sigma_bar=SIGMA_BAR)


def band_radius(arm: TwoLinkArm, band: str, depth: float = TIMELINE["depth"]) -> float:
    """Distance from joint 1 ``depth`` of the way into ``band`` from the boundary of the safe set.

    ``depth = 0`` is the boundary circle, ``depth = 1`` the edge of the workspace.
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, got {band!r}")
    r_in, r_out = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    return r_hi + depth * (r_out - r_hi) if band == "outer" else r_lo - depth * (r_lo - r_in)


def target(arm: TwoLinkArm, band: str, theta0=THETA0, **timeline) -> TurningTarget:
    """The target of the section: starts at the end effector of ``theta0`` and follows :data:`TIMELINE`
    into ``band``; keyword arguments override entries of :data:`TIMELINE`."""
    t = {**TIMELINE, **timeline}
    x0, y0 = arm.end_effector(theta0)
    end = t["duration"] - t["return_end"]
    return TurningTarget(math.hypot(x0, y0), band_radius(arm, band, t["depth"]), math.atan2(y0, x0), t["rate"],
                         t_in=(t["turn"], t["turn"] + t["move"]), t_out=(end - t["move"], end))


def held_window(target: TurningTarget, settle: float = SETTLE) -> Tuple[float, float]:
    """``(start, back)``: the last ``settle`` seconds before the target starts back, from its arrival at the latest."""
    back = target.t_out[0]
    return max(back - settle, target.t_in[1]), back


def held_steps(trace: Trace, target: TurningTarget) -> np.ndarray:
    """Mask of the states of ``trace`` in :func:`held_window`."""
    start, back = held_window(target)
    dt = float(trace.t[1] - trace.t[0])
    return (trace.t >= start - dt / 2) & (trace.t < back - dt / 2)


def least_distance(arm: TwoLinkArm, target: TurningTarget, t) -> np.ndarray:
    """Distance from the target to the safe set at times ``t``: what no safe arm can beat."""
    r_lo, r_hi = arm.safe_radius_interval
    radius = np.array([target.radius(tk)[0] for tk in np.atleast_1d(t)])
    return np.maximum(0.0, np.maximum(r_lo - radius, radius - r_hi))


def tracking_error(arm: TwoLinkArm, trace: Trace, target: TurningTarget) -> np.ndarray:
    """Distance from the end effector to the target, ``|p - p*|``, at each state of ``trace``."""
    return np.array([np.linalg.norm(arm.end_effector(theta) - target.position(tk))
                     for tk, theta in zip(trace.t, trace.theta)])


def run_tracking(arm: TwoLinkArm, band: str, **timeline) -> Dict[str, Trace]:
    """The nominal, standard MCBF, scalar OD and matrix OD closed loops of one tracking case.

    Keys and labels are those of :data:`odmcbf.simulate.CONTROLLERS`.
    """
    u_nom = tracking_controller(arm, target(arm, band, **timeline), GAIN)
    duration = {**TIMELINE, **timeline}["duration"]
    return {name: rollout(arm, u_nom, THETA0, dt=DT, horizon=duration, decay=decay, name=name, label=label,
                          **({} if decay is None else FILTER))
            for name, decay, label in CONTROLLERS}


def run_recovery_case(**overrides) -> Dict[str, Trace]:
    """The recovery experiment of Fig. 3, :data:`RECOVERY`, on the hero arm; keyword arguments override it."""
    settings = {**RECOVERY, **overrides}
    return run_recovery(arm(), **settings, omega_d=FILTER["omega_d"], rho_omega=FILTER["rho_omega"])
