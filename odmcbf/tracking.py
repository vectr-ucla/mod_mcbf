"""Tracking a target point with the two-link arm's end effector.

The arm is velocity controlled, ``thetadot = u``. :func:`tracking_controller`
returns the nominal controller

    u_nom(t, theta) = J(theta)^-1 ( pdot*(t) + k (p*(t) - p(theta)) ),

which feeds the target's velocity forward and corrects the end-effector position
error ``p* - p`` along the straight line to the target. Under ``thetadot = u_nom``
the error obeys ``d/dt (p* - p) = -k (p* - p)`` and decays as ``exp(-k t)``.

:class:`TurningTarget` is the target the closed-loop demo uses: a point that turns
about joint 1 at a constant rate while its distance from joint 1 changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .two_link import TwoLinkKinematics


def _smoothstep(t: float, window: Tuple[float, float]) -> Tuple[float, float]:
    """``s`` rising from 0 at ``t = a`` to 1 at ``t = b`` as ``x^2 (3 - 2x)``, ``x = (t - a)/(b - a)``, and ``ds/dt``."""
    a, b = window
    x = (t - a) / (b - a)
    if x <= 0.0:
        return 0.0, 0.0
    if x >= 1.0:
        return 1.0, 0.0
    return x * x * (3.0 - 2.0 * x), 6.0 * x * (1.0 - x) / (b - a)


@dataclass(frozen=True)
class TurningTarget:
    """A target point that turns about joint 1 at a constant rate while its distance changes.

    At time ``t`` the target is at distance ``r(t)`` from joint 1 and at angle
    ``phi0 + rate t`` from the horizontal, with

        r(t) = r0 + (r1 - r0) (s_in(t) - s_out(t)),

    where ``s_in`` rises smoothly from 0 to 1 over the window ``t_in`` and
    ``s_out`` over ``t_out``. The distance moves from ``r0`` to ``r1`` over
    ``t_in``, stays at ``r1``, and returns to ``r0`` over ``t_out`` (never, if
    ``t_out`` is ``None``). The path has a continuous velocity.
    """

    r0: float
    r1: float
    phi0: float = 0.0
    rate: float = 1.0
    t_in: Tuple[float, float] = (0.0, 1.5)
    t_out: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        for name in ("r0", "r1", "phi0", "rate"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.r0 <= 0 or self.r1 <= 0:
            raise ValueError("r0 and r1 must be positive")
        windows = [self.t_in] + ([] if self.t_out is None else [self.t_out])
        for a, b in windows:
            if not (np.isfinite(a) and np.isfinite(b) and a < b):
                raise ValueError("each window (a, b) must have finite a < b")
        if self.t_out is not None and self.t_out[0] < self.t_in[1]:
            raise ValueError("t_out must start after t_in ends")

    def radius(self, t: float) -> Tuple[float, float]:
        """Distance from joint 1 and its rate, ``(r, rdot)``, at time ``t``."""
        s_in, ds_in = _smoothstep(t, self.t_in)
        s_out, ds_out = (0.0, 0.0) if self.t_out is None else _smoothstep(t, self.t_out)
        step = self.r1 - self.r0
        return self.r0 + step * (s_in - s_out), step * (ds_in - ds_out)

    def position(self, t: float) -> np.ndarray:
        """The target point ``p*(t)``."""
        r, _ = self.radius(t)
        phi = self.phi0 + self.rate * t
        return r * np.array([math.cos(phi), math.sin(phi)])

    def velocity(self, t: float) -> np.ndarray:
        """``pdot*(t)``: the rate of the distance along the target's direction, plus its turning."""
        r, rdot = self.radius(t)
        phi = self.phi0 + self.rate * t
        return (rdot * np.array([math.cos(phi), math.sin(phi)])
                + r * self.rate * np.array([-math.sin(phi), math.cos(phi)]))


def inverse_kinematics(arm: TwoLinkKinematics, p) -> np.ndarray:
    """Joint angles ``theta`` with ``p(theta) = p`` and ``0 < theta2 - theta1 < pi``.

    ``p`` must lie strictly inside the workspace annulus ``|l1 - l2| < |p| < l1 + l2``,
    where two configurations reach it; this is the one with ``theta2 - theta1``
    in ``(0, pi)``, the branch on which ``safe_theta_diff_interval`` is stated.
    """
    x, y = float(p[0]), float(p[1])
    r = math.hypot(x, y)
    r_in, r_out = arm.workspace_radii
    if not r_in < r < r_out:
        raise ValueError(f"|p| = {r:g} must lie strictly between {r_in:g} and {r_out:g}")
    l1, l2 = arm.l1, arm.l2
    diff = math.acos(min(1.0, max(-1.0, (r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2))))
    theta1 = math.atan2(y, x) - math.atan2(l2 * math.sin(diff), l1 + l2 * math.cos(diff))
    return np.array([theta1, theta1 + diff])


def tracking_controller(
    arm: TwoLinkKinematics, target, gain: float
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Nominal controller ``u_nom(t, theta) = J(theta)^-1 (pdot*(t) + k (p*(t) - p(theta)))``.

    ``target`` has ``position(t)`` and ``velocity(t)``, such as a :class:`TurningTarget`,
    and ``gain`` is ``k`` in 1/s. The controller needs ``J(theta)`` invertible, which
    holds everywhere except at ``theta2 - theta1 = 0`` or ``pi``, in particular on the
    whole safe set.
    """
    gain = float(gain)
    if not (np.isfinite(gain) and gain >= 0):
        raise ValueError("gain must be finite and nonnegative")

    def u_nom(t: float, theta) -> np.ndarray:
        pdot = target.velocity(t) + gain * (target.position(t) - arm.end_effector(theta))
        return np.linalg.solve(arm.jacobian(theta), pdot)

    return u_nom
