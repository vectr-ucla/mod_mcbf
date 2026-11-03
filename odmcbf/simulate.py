"""Closed-loop rollouts of the safety filters on the two-link arm.

The state is the pair of absolute joint angles ``theta = (theta1, theta2)``
and the arm is velocity controlled, ``thetadot = u``.

Time convention: states carry the initial condition, so ``theta[k]`` is the
state at ``t[k] = k dt`` and ``u[k]`` is the control computed at ``theta[k]``
and held until ``t[k+1]``.

The nominal command is either a constant, as in the notebooks, or a nominal
controller ``u_nom(t, theta)`` evaluated at ``(t[k], theta[k])`` of the run
itself, such as :func:`odmcbf.tracking.tracking_controller`, as in the paper.

:func:`run_comparison` runs the four controllers from inside the safe set.
:func:`run_recovery` starts outside it, in a band where ``H`` is violated, with a
nominal command that holds the arm there. Only the recovery rate of the matrix
filter brings it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from .filters import SafetyFilter, min_scalar_decay
from .tracking import TurningTarget, tracking_controller
from .two_link import TwoLinkArm

#: Controllers compared in the paper, as (key, decay mode, legend label).
CONTROLLERS = (
    ("nominal", None, "nominal"),
    ("standard", "standard", "standard"),
    ("scalar", "scalar", "scalar OD"),
    ("matrix", "matrix", "matrix OD"),
)

#: Recovery margins ``beta`` of :func:`run_recovery`: ``omega_d / 2`` with the
#: default ``omega_d = 1``, the paper's design and the largest value under which
#: the matrix filter stays no more invasive than the standard one, and half of it.
RECOVERY_BETAS = (0.5, 0.25)

#: Key of the run without the recovery constraint in :func:`run_recovery`.
NO_RECOVERY = "no_recovery"

#: The bands of the workspace where ``H`` is violated: next to its outer circle,
#: where the arm is stretched, and next to its inner circle, where it is folded.
BANDS = ("outer", "inner")

#: Window, in seconds, over which settled quantities are averaged.
SETTLE_WINDOW = 2.0

#: A constant command of shape ``(m,)``, or a nominal controller ``u_nom(t, theta)``.
Nominal = Union[np.ndarray, tuple, list, Callable[[float, np.ndarray], np.ndarray]]


@dataclass
class Trace:
    """One closed-loop run."""

    name: str
    label: str
    decay: Optional[str]
    t: np.ndarray                   # (N + 1,) state times
    theta: np.ndarray               # (N + 1, m) absolute joint angles (theta1, theta2)
    u: np.ndarray                   # (N, m) applied joint rates, u[k] holds on [t[k], t[k+1])
    margin: np.ndarray              # (N + 1,) lambda_min(H(theta))
    decay_deviation: np.ndarray     # (N,) omega - omega_d, or ||Omega - Omega_d||_F
    scalar_certificate: np.ndarray  # (N,) smallest scalar decay certifying u[k]
    safety_residual: np.ndarray     # (N,) lambda_min of the applied barrier LMI
    recovery_residual: np.ndarray   # (N,) slack in the recovery constraint
    solve_ms: np.ndarray            # (N,) solver time per step
    Omega: np.ndarray               # (N, p, p) applied decay multiplier
    arm: Optional[TwoLinkArm] = None  # the arm the run was generated with
    u_nom: Optional[np.ndarray] = None  # (N, m) nominal command the control at step k was filtered from

    def control_deviation(self, u_nom=None) -> np.ndarray:
        """``||u - u_nom||`` along the run.

        ``u_nom`` is a constant command, or ``(N, m)`` commands per step; by
        default, the nominal commands recorded by :func:`rollout`.
        """
        if u_nom is None:
            if self.u_nom is None:
                raise ValueError("this trace has no recorded nominal command; pass u_nom")
            u_nom = self.u_nom
        return np.linalg.norm(self.u - np.asarray(u_nom), axis=1)


def rollout(
    arm: TwoLinkArm,
    u_nom: Nominal,
    theta0,
    dt: float = 0.01,
    horizon: float = 8.0,
    decay: Optional[str] = None,
    label: Optional[str] = None,
    name: Optional[str] = None,
    **filter_kwargs,
) -> Trace:
    """Simulate the arm under one controller with an explicit Euler step.

    ``decay=None`` applies the nominal command unfiltered; otherwise the
    corresponding :class:`~odmcbf.filters.SafetyFilter` is applied at every step.
    ``u_nom`` is a constant command of shape ``(m,)`` or a callable
    ``u_nom(t, theta)`` returning one, evaluated at ``(t[k], theta[k])`` of this
    run; the commands used are recorded in ``Trace.u_nom``.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    steps = round(horizon / dt)
    if not np.isclose(steps * dt, horizon):
        raise ValueError("dt must divide the horizon")

    m, p = arm.n_controls, arm.dim
    controller = u_nom if callable(u_nom) else None
    if controller is None:
        u_nom = np.asarray(u_nom, dtype=float)
        if u_nom.shape != (m,):
            raise ValueError(f"u_nom must have shape ({m},)")
    filt = (
        None
        if decay is None
        else SafetyFilter(m, p, decay=decay, **filter_kwargs)
    )

    theta = np.empty((steps + 1, m))
    theta[0] = np.asarray(theta0, dtype=float)
    u = np.empty((steps, m))
    commands = np.empty((steps, m))
    Omega = np.empty((steps, p, p))
    decay_deviation = np.full(steps, np.nan)
    scalar_certificate = np.full(steps, np.nan)
    safety_residual = np.full(steps, np.nan)
    recovery_residual = np.full(steps, np.nan)
    solve_ms = np.full(steps, np.nan)

    omega_d = 1.0 if filt is None else filt.omega_d
    for k in range(steps):
        if controller is not None:
            u_nom = np.asarray(controller(k * dt, theta[k].copy()), dtype=float)
            if u_nom.shape != (m,) or not np.all(np.isfinite(u_nom)):
                raise ValueError(f"u_nom(t, theta) must return a finite array of shape ({m},); "
                                 f"got {u_nom!r} at t = {k * dt:g}")
        commands[k] = u_nom
        if filt is None:
            u[k] = u_nom
            Omega[k] = 0.5 * np.eye(p)
        else:
            res = filt(arm.alpha_H(theta[k]), arm.LgH(theta[k]), u_nom)
            u[k] = res.u
            Omega[k] = res.Omega
            safety_residual[k] = res.safety_residual
            solve_ms[k] = 1e3 * res.solve_time
            if res.recovery_residual is not None:
                recovery_residual[k] = res.recovery_residual
            if decay == "scalar":
                decay_deviation[k] = res.omega - omega_d
            elif decay == "matrix":
                decay_deviation[k] = np.linalg.norm(
                    res.Omega - (omega_d / 2) * np.eye(p)
                )
        alpha_H = arm.alpha_H(theta[k])
        if np.linalg.eigvalsh(alpha_H)[0] > 0:
            scalar_certificate[k] = min_scalar_decay(alpha_H, arm.Hdot(theta[k], u[k]))
        theta[k + 1] = theta[k] + dt * u[k]

    margin = np.array([arm.margin(theta_k) for theta_k in theta])
    return Trace(
        name=name or (decay or "nominal"),
        label=label or (decay or "nominal"),
        decay=decay,
        t=np.arange(steps + 1) * dt,
        theta=theta,
        u=u,
        margin=margin,
        decay_deviation=decay_deviation,
        scalar_certificate=scalar_certificate,
        safety_residual=safety_residual,
        recovery_residual=recovery_residual,
        solve_ms=solve_ms,
        Omega=Omega,
        arm=arm,
        u_nom=commands,
    )


def run_comparison(
    arm: Optional[TwoLinkArm] = None,
    u_nom: Nominal = (1.0, 0.4),
    theta0=(0.0, np.pi / 2),
    dt: float = 0.01,
    horizon: float = 8.0,
    **filter_kwargs,
) -> Dict[str, Trace]:
    """Run the nominal, standard, scalar OD and matrix OD controllers.

    ``arm`` defaults to :class:`~odmcbf.two_link.TwoLinkArm` with unit links.
    The default start ``theta0 = (0, pi/2)`` has ``theta2 - theta1 = pi/2``,
    where the margin is largest. The default command ``u_nom = (1, 0.4)``
    turns joint 1 at 1 rad/s while closing ``theta2 - theta1`` at 0.6 rad/s,
    toward the stretched singular configuration. ``u_nom`` may also be a
    nominal controller ``u_nom(t, theta)``, as in :func:`rollout`.
    """
    arm = TwoLinkArm() if arm is None else arm
    traces = {}
    for name, decay, label in CONTROLLERS:
        traces[name] = rollout(
            arm,
            u_nom,
            theta0,
            dt=dt,
            horizon=horizon,
            decay=decay,
            name=name,
            label=label,
            **filter_kwargs,
        )
    return traces


def band_start(arm: TwoLinkArm, band: str = "outer", depth: float = 0.3) -> np.ndarray:
    """A configuration ``depth`` of the way into a band where ``H`` is violated.

    The end effector is at distance ``r_hi + depth (l1 + l2 - r_hi)`` from joint 1
    in the ``"outer"`` band, where the arm is stretched, or at ``r_lo - depth (r_lo -
    |l1 - l2|)`` in the ``"inner"`` band, where it is folded. So ``depth`` is the
    fraction of the band's width from the boundary of the safe set, as ``--depth``
    in ``scripts/closed_loop_demo.py``, and ``0 < depth < 1`` keeps the start off
    that boundary and off the singular configurations at the edge of the workspace.
    The configuration has ``theta1 = 0`` and ``0 < theta2 - theta1 < pi``.
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, got {band!r}")
    depth = float(depth)
    if not 0.0 < depth < 1.0:
        raise ValueError(f"depth must lie in (0, 1); got {depth:g}")
    r_in, r_out = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    radius = r_hi + depth * (r_out - r_hi) if band == "outer" else r_lo - depth * (r_lo - r_in)
    cos_diff = (radius ** 2 - arm.l1 ** 2 - arm.l2 ** 2) / (2 * arm.l1 * arm.l2)
    return np.array([0.0, float(np.arccos(np.clip(cos_diff, -1.0, 1.0)))])


def hold_at_start(arm: TwoLinkArm, theta0, gain: float = 5.0) -> Callable[[float, np.ndarray], np.ndarray]:
    """The nominal controller of :func:`run_recovery`: it tracks a target that stays at ``p(theta0)``.

    This is :func:`~odmcbf.tracking.tracking_controller` with gain ``gain`` and a
    :class:`~odmcbf.tracking.TurningTarget` that neither turns nor moves, so
    ``u_nom(t, theta) = J(theta)^-1 k (p(theta0) - p(theta))``: zero at
    ``theta0``, and pointing back toward it from anywhere else.
    """
    start = arm.end_effector(theta0)
    radius = float(np.hypot(*start))
    target = TurningTarget(radius, radius, float(np.arctan2(start[1], start[0])), rate=0.0)
    return tracking_controller(arm, target, gain)


def run_recovery(
    arm: Optional[TwoLinkArm] = None,
    band: str = "outer",
    depth: float = 0.3,
    gain: float = 5.0,
    dt: float = 0.01,
    horizon: float = 5.0,
    betas: Sequence[float] = RECOVERY_BETAS,
    **filter_kwargs,
) -> Dict[str, Trace]:
    """Matrix OD from inside a band where ``H`` is violated, with a nominal command that holds the arm there.

    The arm starts at rest at :func:`band_start`, and the nominal command is
    :func:`hold_at_start`: zero at the start and, once the filter has moved the
    arm, pointing back toward it. Unfiltered, the arm stays where it is; filtered,
    it leaves the band only as fast as the filter's decay term demands. With
    ``alpha(H) = gamma H`` and ``lambda_min(H) < 0``, the matrix inequality along
    the unit eigenvector ``v0`` of ``lambda_min(H)`` reads

        d/dt lambda_min(H) = v0' Hdot v0 >= 2 gamma |lambda_min(H)| v0' sym(Omega) v0,

    so under the recovery constraint ``sym(Omega) >= beta I`` the margin rises at
    least at the relative rate :func:`recovery_rate` of ``2 beta gamma``. Without
    the constraint the filter may drop the decay along ``v0`` and leave the arm in
    the band.

    Runs the matrix filter with the recovery constraint at each ``beta`` in
    ``betas``, under the keys ``"beta=<beta>"``, and once without it, under
    :data:`NO_RECOVERY`. ``arm`` defaults to :class:`~odmcbf.two_link.TwoLinkArm`
    with unit links, and ``filter_kwargs`` (``omega_d``, ``rho_omega``, ...) go to
    every filter.
    """
    for key in ("decay", "recovery", "beta"):
        if key in filter_kwargs:
            raise TypeError(f"run_recovery sets {key} for each run; do not pass it")
    betas = [float(beta) for beta in betas]
    if not betas or any(not np.isfinite(beta) or beta <= 0 for beta in betas):
        raise ValueError("betas must hold at least one positive recovery margin")
    if len(set(betas)) != len(betas):
        raise ValueError("betas must be distinct")
    arm = TwoLinkArm() if arm is None else arm
    theta0 = band_start(arm, band, depth)
    u_nom = hold_at_start(arm, theta0, gain)
    traces = {}
    for beta in betas:
        key = f"beta={beta:g}"
        traces[key] = rollout(arm, u_nom, theta0, dt=dt, horizon=horizon, decay="matrix", name=key,
                              label=f"matrix OD, beta = {beta:g}", recovery=True, beta=beta, **filter_kwargs)
    traces[NO_RECOVERY] = rollout(arm, u_nom, theta0, dt=dt, horizon=horizon, decay="matrix", name=NO_RECOVERY,
                                  label="matrix OD, no recovery", recovery=False, **filter_kwargs)
    return traces


def recovery_rate(trace: Trace) -> np.ndarray:
    """``d/dt lambda_min(H) / |lambda_min(H)|`` at each step, under the applied control.

    Entry ``k`` is ``TwoLinkArm.margin_rate(theta[k], u[k]) / |margin[k]|``, the
    first-order rate at which ``u[k]`` raises a negative margin, relative to it;
    NaN where the margin is nonnegative. The margin then evolves as ``margin[k + 1]
    = margin[k] (1 - dt rate[k])`` up to terms of order ``dt**2``. The trace must
    carry its arm, as every :func:`rollout` does.
    """
    if trace.arm is None:
        raise ValueError("this trace has no arm; run it with rollout")
    rate = np.full(len(trace.u), np.nan)
    for k, (theta, u) in enumerate(zip(trace.theta[:-1], trace.u)):
        if trace.margin[k] < 0:
            rate[k] = trace.arm.margin_rate(theta, u) / -trace.margin[k]
    return rate


def summarize(trace: Trace, u_nom=None, omega_d: float = 1.0) -> dict:
    """Numerical summary of one run, including the quantities quoted in the paper.

    ``u_nom`` is as in :meth:`Trace.control_deviation`; by default, the recorded commands.
    """
    dt = float(trace.t[1] - trace.t[0])
    window = max(1, round(SETTLE_WINDOW / dt))

    def _finite_min(values):
        finite = values[np.isfinite(values)]
        return float(finite.min()) if finite.size else None

    def _finite_mean(values):
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else None

    settled = trace.decay_deviation[-window:]
    return {
        "min_margin": float(trace.margin.min()),
        "joint1_rotation": float(trace.theta[-1, 0] - trace.theta[0, 0]),
        "late_joint1_rate": float(trace.u[-window:, 0].mean()),
        "control_deviation_l2": float(
            np.sqrt(dt * np.sum(trace.control_deviation(u_nom) ** 2))
        ),
        "decay_deviation_settled": (
            float(np.mean(settled)) if np.all(np.isfinite(settled)) else None
        ),
        "mean_solve_ms": _finite_mean(trace.solve_ms),
        "median_solve_ms": (
            float(np.median(trace.solve_ms[np.isfinite(trace.solve_ms)]))
            if np.any(np.isfinite(trace.solve_ms))
            else None
        ),
        "min_safety_residual": _finite_min(trace.safety_residual),
        "min_recovery_residual": _finite_min(trace.recovery_residual),
    }
