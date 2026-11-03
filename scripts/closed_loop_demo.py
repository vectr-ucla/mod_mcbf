"""Closed loops that track a target point near either boundary of the safe set.

Every case keeps the plant and the safety filters of the paper's Section VIII, in the
absolute joint angles of ``two_link_manipulator_kinematics.md`` (the note): velocity control
``thetadot = u``, and the standard MCBF, scalar OD and matrix OD filters of
``odmcbf/filters.py`` with the paper's settings (``odmcbf.paper.FILTER``: ``omega_d = 1``,
``rho_omega = 5e-3``, ``rho_Omega = (4/p) rho_omega``, recovery constraints
``omega >= omega_d`` and ``sym(Omega) >= (omega_d/2) I``). The runs are
``odmcbf.simulate.rollout``.

The nominal command is the paper's tracking controller, eq. (32): it tracks a moving
target point ``p*`` with the end effector ``p``, by the controller of
``odmcbf/tracking.py``,

    u_nom(t, theta) = J(theta)^-1 ( pdot*(t) + k (p*(t) - p(theta)) ),

evaluated at each arm's own state: the target's velocity fed forward, plus the
position error along the straight line to the target. Unfiltered, the error decays
as ``exp(-k t)``; each filter then keeps its arm as close to the target as its
certificate allows.

The arm is ``TwoLinkArm(l1, l2, sigma_bar=...)`` with ``H = J J^T - sigma_bar^2 I``
and ``alpha(H) = gamma H``. The safe set is the annulus ``r_lo <= |p| <= r_hi``
about joint 1: stretched configurations outside the outer circle and folded ones
inside the inner circle violate ``H >= 0``.

The target starts at the end effector of ``theta0`` (``(0, pi/2)`` by default, as in
the paper) and turns about joint 1 at ``--rate``. After ``--turn`` seconds of
turning only, its distance from joint 1 moves to ``r1`` over ``--move`` seconds; it
stays there and comes back over the ``--move`` seconds that end 0.5 s before the end
of the run, after which every arm can reach it again.
The cases differ in ``r1`` (``--case``), and the hero case also in its defaults:

``hero`` (default)
    Fig. 1 of the paper, the experiment defined in ``odmcbf.paper``. This is the ``outer``
    case on the arm ``l1 = 1``, ``l2 = 0.75``, ``sigma_bar = 0.6``, with ``--depth 0.3``,
    ``--turn 1``, ``--move 0.75`` and ``--duration 4.5``. The target turns at 1 rad/s,
    4.5 rad in all, well short of a full turn, and goes 0.072 into the band. Anywhere in
    this arm's safe set, the standard MCBF certifies rotation of the whole arm about
    joint 1 only up to 0.82 rad/s (Section 7 of the note), so it falls behind while the
    target is still in the safe set. In the band, scalar OD stays off the boundary and
    matrix OD slides along it next to the target.
    Besides ``NAME_paths.png`` and ``NAME_closed_loop.png``, it writes the paper's figure
    as two untitled subfigures: ``NAME_a`` (the closed loops in one workspace) and
    ``NAME_b`` (the joint rates each filter certifies at the boundary). Each is a ``.pdf``
    and a ``.png`` at its printed size, 0.49 of the paper's column width. It also writes
    ``NAME_tracking`` (the distance to the target, not in the paper) and ``NAME.tex``, a
    LaTeX float with the two subfigures. Any flag overrides its default.
``inner``
    ``r1`` in the folded band, ``--depth`` of the way from ``r_lo`` to ``|l1 - l2|``.
``outer``
    ``r1`` in the stretched band, ``--depth`` of the way from ``r_hi`` to ``l1 + l2``.
``graze``
    ``r1 = r_lo``: the target runs along the inner boundary circle.
``rotate``
    ``r1 = r0``: the target only turns, at the start's distance, and never leaves the
    safe set. On an arm with ``l1 != l2`` the standard MCBF certifies this rotation
    only up to the rate of Section 7 of the note, ``gamma sqrt(lam0 lam1)/(lam1 - lam0)``
    with ``lam0 <= lam1`` the eigenvalues of ``H``. That is at most
    ``gamma sqrt((l1^2 - sb^2)(l2^2 - sb^2))/|l1^2 - l2^2|``, reached at
    ``theta2 - theta1 = pi/2``. The script prints this cap, checks it against
    bisection on the filter's matrix inequality, and runs the standard MCBF just below
    and just above it.

While the target is outside the safe set, the closest safe point to it is on the
boundary circle at the target's angle. Staying there means sliding along the
boundary, rotating the whole arm about joint 1 (``theta1dot = theta2dot``) at the
target's rate. That motion has ``Z00 = 0``, ``Z01 = -lambda theta1dot``: matrix OD
certifies it (Theorem 1), scalar OD cannot for any ``omega`` (Proposition 1); see
Sections 5 and 6 of the note. Every command is held for ``--dt``, for which
``theta + dt u`` is the exact solution of ``thetadot = u``.

Examples, from the repository root (the script also runs from any other directory
when given its full path):

    python scripts/closed_loop_demo.py                         # the hero configuration
    python scripts/closed_loop_demo.py --case inner            # inner band, unit links
    python scripts/closed_loop_demo.py --case outer
    python scripts/closed_loop_demo.py --case inner --l1 1 --l2 0.6 --sigma-bar 0.3
    python scripts/closed_loop_demo.py --case graze --video figures/graze.mp4
    python scripts/closed_loop_demo.py --case rotate --l1 1 --l2 0.6 --sigma-bar 0.3
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))               # import odmcbf from the repository root

from odmcbf import (CONTROLLERS, SafetyFilter, TurningTarget, TwoLinkArm, figures,  # noqa: E402
                    inverse_kinematics, min_scalar_decay, paper, rollout, tracking_controller)

KEYS = tuple(name for name, _, _ in CONTROLLERS)          # nominal, standard, scalar, matrix
FILTERS = KEYS[1:]

#: Counter-clockwise quarter turn; v1 = QUARTER v0, and the rotation certificate.
QUARTER = np.array([[0.0, -1.0], [1.0, 0.0]])

#: Margins within this of zero count as on the boundary.
MARGIN_TOL = 1e-9

#: Relative tolerance on the blocks Z00, Z01 and on the eigenvalue gap of H.
BLOCK_TOL = 1e-9

#: Window, in seconds, over which the held quantities are averaged: the last SETTLE
#: seconds before the target starts to come back, or less if it arrived later.
SETTLE = 2.0

#: After --turn seconds of turning only, the target's distance moves over --move seconds,
#: stays for at least MIN_HOLD seconds, and comes back over --move seconds that end
#: RETURN_END before the end of the run (the paper's timeline, ``odmcbf.paper.TIMELINE``).
MIN_HOLD, RETURN_END = 1.0, paper.TIMELINE["return_end"]

#: Defaults of the hero case (the paper's, from ``odmcbf.paper``), and of every other case, for the flags that differ.
HERO = dict(**paper.HERO_ARM, **{key: paper.TIMELINE[key] for key in ("depth", "turn", "move", "duration")})
DEFAULTS = dict(l1=1.0, l2=1.0, sigma_bar=None, depth=0.3, turn=0.0, move=1.5, duration=8.0)

#: Hero subfigures (a) and (b): the paper sets them side by side in one column, each HERO_SHARE of the
#: ieeeconf column width (245.72 pt), so each is drawn HERO_SIZE, in inches, at its printed size.
HERO_SHARE = 0.49
HERO_SIZE = (HERO_SHARE * 245.71811 / 72.27, 1.785)       # TeX points: 72.27 to the inch
#: Size of the distance-to-target figure, which is not in the paper.
HERO_TRACKING_SIZE = (3.45, 2.45)

#: Candidate spacings of the arm snapshots, in seconds.
SNAPSHOT_SPACINGS = (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0)
MAX_SNAPSHOTS = 10

#: Video: frames per second and simulation steps per frame.
VIDEO_FPS, VIDEO_EVERY = 25, 4


def case_target(case: str, arm: TwoLinkArm, theta0, depth: float, rate: float,
                duration: float, turn: float = 0.0, move: float = 1.5) -> TurningTarget:
    """The target of a case, starting at the end effector of ``theta0``; its distance
    starts to move after ``turn`` seconds, and each move takes ``move`` seconds.

    The band cases are ``odmcbf.paper.target``, the paper's target, with these settings."""
    if case in ("hero", "outer", "inner"):
        return paper.target(arm, "outer" if case == "hero" else case, theta0, depth=depth, rate=rate,
                            duration=duration, turn=turn, move=move, return_end=RETURN_END)
    x0, y0 = arm.end_effector(theta0)
    r0, phi0 = math.hypot(x0, y0), math.atan2(y0, x0)
    r_lo, r_hi = arm.safe_radius_interval
    if case == "graze":
        r1 = r_lo
    elif case == "rotate":
        r1 = r0
    else:
        raise ValueError(case)
    end = duration - RETURN_END
    return TurningTarget(r0, r1, phi0, rate, t_in=(turn, turn + move), t_out=(end - move, end))


def outside(arm: TwoLinkArm, radius: float) -> bool:
    """Whether a point at this distance from joint 1 is on or outside the boundary of the safe set."""
    r_lo, r_hi = arm.safe_radius_interval
    return radius <= r_lo * (1 + BLOCK_TOL) or radius >= r_hi * (1 - BLOCK_TOL)


# ---------------------------------------------------------- eigenbasis blocks
def boundary_lambda(arm: TwoLinkArm) -> float:
    """``lambda = l1^2 + l2^2 - 2 sigma_bar^2``, the nonzero eigenvalue of ``H`` on the boundary."""
    return arm.l1 ** 2 + arm.l2 ** 2 - 2 * arm.sigma_bar_sq


def eigenbasis(arm: TwoLinkArm, theta):
    """Eigenvalues of ``H``, its unit eigenvector ``v0`` for the smaller one, and ``v1``,
    the counter-clockwise quarter turn of ``v0``."""
    lam, vectors = np.linalg.eigh(arm.H(theta))
    v0 = vectors[:, 0]
    return lam, v0, QUARTER @ v0


def blocks(arm: TwoLinkArm, theta, u) -> tuple[float, float, float]:
    """``Z00, Z01, Z11`` of ``V' Hdot(u) V``; NaN where ``H`` has a repeated eigenvalue."""
    lam, v0, v1 = eigenbasis(arm, theta)
    if lam[1] - lam[0] <= BLOCK_TOL * max(1.0, abs(lam[1])):
        return math.nan, math.nan, math.nan
    Hdot = arm.Hdot(theta, u)
    return float(v0 @ Hdot @ v0), float(v0 @ Hdot @ v1), float(v1 @ Hdot @ v1)


def scalar_decay_needed(arm: TwoLinkArm, theta, u) -> float:
    """Smallest ``omega >= 0`` with ``Hdot(u) + omega alpha(H) >= 0``, or ``inf`` if none.

    Inside the safe set this is the bound of Proposition 2 (``min_scalar_decay``).
    On the boundary it follows from the blocks: Proposition 1 rules out every
    ``omega`` unless ``Z00 > 0`` or ``Z00 = Z01 = 0``, and for ``Z00 > 0`` the
    singular case of Proposition 2 gives ``(Z01^2 / Z00 - Z11) / alpha(lambda)``.
    NaN outside the safe set.
    """
    margin = arm.margin(theta)
    if margin > MARGIN_TOL:
        return min_scalar_decay(arm.alpha_H(theta), arm.Hdot(theta, u))
    if margin < -MARGIN_TOL:
        return math.nan
    lam = np.linalg.eigvalsh(arm.H(theta))[1]
    z00, z01, z11 = blocks(arm, theta, u)
    tol = BLOCK_TOL * max(1.0, float(np.linalg.norm(arm.Hdot(theta, u))))
    if z00 > tol:
        return max(0.0, (z01 ** 2 / z00 - z11) / (arm.gamma * lam))
    if z00 < -tol or abs(z01) > tol:
        return math.inf
    return max(0.0, -z11 / (arm.gamma * lam))


# ------------------------------------------------------------------ closed loop
def turned(points: np.ndarray) -> np.ndarray:
    """Angle of each point about joint 1 minus that of the first, unwrapped: how far it has turned."""
    angle = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    return angle - angle[0]


@dataclass
class Run:
    """One closed loop from ``odmcbf.simulate.rollout`` with the quantities the figures need.

    ``theta[k]`` is the state at ``t[k] = k dt``; ``u[k]`` is held on ``[t[k], t[k+1])``
    and was filtered from ``u_nom[k]``, the tracking command at ``(t[k], theta[k])``.
    """

    key: str
    dt: float
    t: np.ndarray                     # (N + 1,)
    theta: np.ndarray                 # (N + 1, 2)
    u: np.ndarray                     # (N, 2) applied command
    u_nom: np.ndarray                 # (N, 2) nominal command at this run's own state
    margin: np.ndarray                # (N + 1,) lambda_min(H(theta))
    point: np.ndarray                 # (N + 1, 2) end effector
    target: np.ndarray                # (N + 1, 2) target point
    bound: np.ndarray                 # (N + 1,) distance from the target to the safe set
    target_outside: np.ndarray        # (N + 1,) the target is on or outside the boundary
    Omega: np.ndarray                 # (N, 2, 2) applied multiplier; (omega/2) I for scalar OD
    scalar_decay_needed: np.ndarray   # (N,) scalar_decay_needed(theta[k], u[k])
    safety_residual: np.ndarray       # (N,) lambda_min of the applied barrier LMI
    inaccurate: int                   # solves reported optimal_inaccurate

    @property
    def error(self) -> np.ndarray:
        """Distance from the end effector to the target, ``|p - p*|``."""
        return np.linalg.norm(self.point - self.target, axis=1)

    @property
    def lag(self) -> np.ndarray:
        """Signed angle about joint 1 from the end effector to the target, in ``(-pi, pi]``."""
        (x, y), (xs, ys) = self.point.T, self.target.T
        return np.arctan2(x * ys - y * xs, x * xs + y * ys)

    @property
    def decay_deviation(self) -> np.ndarray:
        """``||Omega - Omega_d||_F``; for scalar OD, ``Omega = omega I/2``. Zero for the standard MCBF."""
        return np.linalg.norm(self.Omega - 0.5 * np.eye(2), axis=(1, 2))

    @property
    def contact(self) -> np.ndarray:
        """Steps at which the target is on or outside the boundary of the safe set."""
        return self.target_outside[:-1]


def simulate(arm: TwoLinkArm, theta0, target: TurningTarget, gain: float, duration: float, dt: float,
             rho_omega: float) -> Dict[str, Run]:
    """``rollout`` of the tracking controller for every controller of ``odmcbf.simulate.CONTROLLERS``."""
    u_nom = tracking_controller(arm, target, gain)
    r_lo, r_hi = arm.safe_radius_interval
    runs = {}
    for name, decay, label in CONTROLLERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trace = rollout(arm, u_nom, theta0, dt=dt, horizon=duration, decay=decay, name=name,
                            label=label, **({} if decay is None else
                                            dict(omega_d=1.0, rho_omega=rho_omega, recovery=True)))
        inaccurate = sum("inaccurate" in str(w.message) for w in caught)
        runs[name] = trace, inaccurate
    t = runs["nominal"][0].t
    target_point = np.array([target.position(tk) for tk in t])
    target_radius = np.array([target.radius(tk)[0] for tk in t])
    bound = np.maximum(0.0, np.maximum(r_lo - target_radius, target_radius - r_hi))
    target_outside = np.array([outside(arm, r) for r in target_radius])
    out = {}
    for name, (trace, inaccurate) in runs.items():
        out[name] = Run(
            key=name, dt=dt, t=trace.t, theta=trace.theta, u=trace.u, u_nom=trace.u_nom,
            margin=trace.margin, point=np.array([arm.end_effector(th) for th in trace.theta]),
            target=target_point, bound=bound, target_outside=target_outside, Omega=trace.Omega,
            scalar_decay_needed=np.array([scalar_decay_needed(arm, th, u)
                                          for th, u in zip(trace.theta[:-1], trace.u)]),
            safety_residual=trace.safety_residual, inaccurate=inaccurate,
        )
    return out


def held_window(target: TurningTarget) -> tuple[float, float]:
    """The last ``SETTLE`` seconds before the target starts to come back, from when it arrived at the latest."""
    back = target.t_out[0]
    return max(back - SETTLE, target.t_in[1]), back


def held_steps(run: Run, target: TurningTarget) -> np.ndarray:
    """States in the held window."""
    start, back = held_window(target)
    return (run.t >= start - run.dt / 2) & (run.t < back - run.dt / 2)


def summarize(run: Run, arm: TwoLinkArm, target: TurningTarget) -> Dict[str, float]:
    """Numbers for the table; "held" means the held window."""
    held = held_steps(run, target)
    held_u = held[:-1]
    deviation = np.linalg.norm(run.u - run.u_nom, axis=1)
    sym = 0.5 * (run.Omega + np.transpose(run.Omega, (0, 2, 1)))
    filtered = run.key != "nominal"

    def reduce(values, how):
        values = values[~np.isnan(values)]
        return float(how(values)) if values.size else math.nan

    return {
        "min_margin": float(run.margin.min()),
        "held_margin": float(run.margin[held].mean()),
        "held_error": float(run.error[held].mean()),
        "held_excess": float((run.error - run.bound)[held].mean()) if filtered else math.nan,
        "held_lag": float(run.lag[held].mean()) + 0.0,
        "held_rotation_rate": float(np.polyfit(run.t[held], run.theta[held].mean(axis=1), 1)[0]),
        "end_error": float(run.error[-1]),
        "rotation": float(turned(run.point)[-1]),
        "deviation_l2": float(np.sqrt(run.dt * np.sum(deviation ** 2))),
        "held_decay_deviation": float(run.decay_deviation[held_u].mean()) if filtered else math.nan,
        "held_scalar_decay_needed": reduce(run.scalar_decay_needed[held_u], np.median),
        "inaccurate": float(run.inaccurate),
        "min_residual": reduce(run.safety_residual, np.min),
        "max_decay_rate_dt": (float((2 * arm.gamma * np.linalg.eigvalsh(sym)[:, -1] * run.dt).max())
                              if filtered else math.nan),
    }


ROWS = (
    ("min lambda_min(H)", "min_margin", "{:+.2e}"),
    ("held lambda_min(H)", "held_margin", "{:+.2e}"),
    ("held distance to the target |p - p*|", "held_error", "{:.4f}"),
    ("  beyond the least possible", "held_excess", "{:.2e}"),
    ("held angle behind the target [rad]", "held_lag", "{:+.4f}"),
    ("held rotation rate about joint 1 [rad/s]", "held_rotation_rate", "{:.4f}"),
    ("distance to the target at the end", "end_error", "{:.4f}"),
    ("rotation about joint 1 over the run [rad]", "rotation", "{:.3f}"),
    ("L2 norm of u - u_nom", "deviation_l2", "{:.4f}"),
    ("held ||Omega - Omega_d||_F", "held_decay_deviation", "{:.3f}"),
    ("held scalar decay for u (median)", "held_scalar_decay_needed", "{:.3g}"),
    ("optimal_inaccurate solves", "inaccurate", "{:.0f}"),
    ("min LMI residual", "min_residual", "{:+.1e}"),
    ("max decay rate x dt", "max_decay_rate_dt", "{:.3f}"),
)


def print_table(summaries: Dict[str, Dict[str, float]]) -> None:
    keys = list(summaries)
    print(f"{'':40s}" + "".join(f"{figures.TITLES[k]:>15s}" for k in keys))
    for label, field, fmt in ROWS:
        cells = []
        for k in keys:
            value = summaries[k][field]
            if math.isnan(value):
                text = "-"
            elif math.isinf(value):
                text = "inf"
            else:
                text = fmt.format(value)
                if text.startswith("-") and float(text) == 0.0:
                    text = text[1:]                              # no "-0.0000"
            cells.append(text.rjust(15))
        print(f"{label:40s}" + "".join(cells))


def contact_intervals(run: Run) -> List[tuple[float, float]]:
    """Maximal time intervals over which the target is on or outside the boundary."""
    intervals, start = [], None
    for k, on in enumerate(run.contact):
        if on and start is None:
            start = run.t[k]
        if not on and start is not None:
            intervals.append((start, run.t[k]))
            start = None
    if start is not None:
        intervals.append((start, run.t[-1]))
    return intervals


# ------------------------------------------------------ Proposition 1 check
def boundary_check(arm: TwoLinkArm, target: TurningTarget, duration: float, dt: float,
                   rho_omega: float) -> None:
    """Proposition 1 and Theorem 1 where the target first reaches the boundary.

    The configuration puts the end effector on the boundary circle at the target's
    angle, with ``0 < theta2 - theta1 < pi``, and the velocity is the rotation of the
    whole arm about joint 1 at the target's rate: the closest safe motion to the target.
    """
    lo, hi = arm.safe_theta_diff_interval
    r_lo, r_hi = arm.safe_radius_interval
    steps = round(duration / dt)
    hit = next((k for k in range(steps + 1) if outside(arm, target.radius(k * dt)[0])), None)
    if hit is None:
        print("\nThe target never reaches the boundary of the safe set.")
        return
    radius = target.radius(hit * dt)[0]
    phi, rate = target.phi0 + target.rate * hit * dt, target.rate
    inner = abs(radius - r_lo) < abs(radius - r_hi)
    diff = hi if inner else lo
    theta1 = inverse_kinematics(arm, (r_lo if inner else r_hi) * np.array([math.cos(phi), math.sin(phi)]))[0]
    theta = np.array([theta1, theta1 + diff])
    rate = rate if abs(rate) > 1e-9 else 1.0
    gamma, l1, l2, omega_d = arm.gamma, arm.l1, arm.l2, 1.0
    lam = boundary_lambda(arm)
    alpha_H = arm.alpha_H(theta)
    _, v0, v1 = eigenbasis(arm, theta)
    V = np.column_stack([v0, v1])
    u = np.full(2, rate)
    Hdot = arm.Hdot(theta, u)
    z00, z01, _ = blocks(arm, theta, u)
    note00 = (u[1] - u[0]) * l1 ** 2 * l2 ** 2 * math.sin(2 * diff) / lam
    note01 = -lam * u[0] - (u[1] - u[0]) * l2 ** 2 * (l2 ** 2 + l1 ** 2 * math.cos(2 * diff)) / lam
    side = "inner" if inner else "outer"
    print(f"\nProposition 1 where the target first reaches the {side} boundary (t = {hit * dt:.2f} s): "
          f"theta = ({theta[0]:+.4f}, {theta[1]:+.4f}), theta2 - theta1 = {diff:.6f}, "
          f"lambda_min(H) = {arm.margin(theta):+.1e}")
    print(f"  u = rotation about joint 1 at the target's rate, ({rate:.4g}, {rate:.4g}): "
          f"Z00 = {z00:+.2e} (note: {note00 + 0.0:+.1e}), Z01 = {z01:+.6f} (note: {note01:+.6f} = -lambda rate)")
    for omega in (omega_d, 1e2, 1e4):
        X = V.T @ (Hdot + omega * alpha_H) @ V
        print(f"  scalar, omega = {omega:>7g}: det V'(Hdot + omega alpha(H))V = {np.linalg.det(X):+.6f} "
              f"(-Z01^2 = {-z01 ** 2:+.6f}), lambda_min = {np.linalg.eigvalsh(X)[0]:+.2e}")
    print("  so no omega >= 0 certifies u: the determinant does not depend on omega.")
    Omega = 0.5 * omega_d * np.eye(2) + (rate / gamma) * QUARTER
    X = Hdot + Omega.T @ alpha_H + alpha_H @ Omega
    print(f"  matrix, Omega = Omega_d + (rate/gamma) [[0,-1],[1,0]] (the paper's rotation certificate): "
          f"lambda_min(Hdot + Omega' alpha(H) + alpha(H) Omega) = {np.linalg.eigvalsh(X)[0]:+.1e}, "
          f"lambda_min(sym Omega) = {np.linalg.eigvalsh(0.5 * (Omega + Omega.T))[0]:.3f} "
          f"(recovery needs {omega_d / 2:g}), ||Omega - Omega_d||_F = "
          f"{np.linalg.norm(Omega - 0.5 * omega_d * np.eye(2)):.4f}")
    print("  Filter outputs for this u at this state:")
    for key in FILTERS:
        res = SafetyFilter(2, 2, decay=key, omega_d=omega_d, rho_omega=rho_omega)(
            alpha_H, arm.LgH(theta), u)
        z00, z01, _ = blocks(arm, theta, res.u)
        print(f"    {figures.TITLES[key]:13s}: u = ({res.u[0]:+.4f}, {res.u[1]:+.4f}), |u - u_nom| = "
              f"{np.linalg.norm(res.u - u):.4f}, rotation (u1 + u2)/2 = {res.u.mean():+.4f}, "
              f"Z00 = {z00:+.4f}, Z01 = {z01:+.4f}")


# ------------------------------------------------ rotation cap (standard MCBF)
def rotation_cap(arm: TwoLinkArm, theta) -> float:
    """Largest rotation rate ``theta1dot = theta2dot`` the standard MCBF certifies at ``theta``.

    Section 7 of the note: ``sqrt(det alpha(H)) / (lam1 - lam0)`` with ``lam0 <= lam1``
    the eigenvalues of ``H``; ``inf`` where they coincide.
    """
    lam = np.linalg.eigvalsh(arm.H(theta))
    if lam[1] - lam[0] <= BLOCK_TOL * max(1.0, abs(lam[1])):
        return math.inf
    return math.sqrt(max(np.linalg.det(arm.alpha_H(theta)), 0.0)) / (lam[1] - lam[0])


def max_rotation_cap(arm: TwoLinkArm) -> float:
    """The largest of these rates over the safe set, reached at ``theta2 - theta1 = pi/2``.

    Section 7 of the note: ``gamma sqrt((l1^2 - sb^2)(l2^2 - sb^2)) / |l1^2 - l2^2|``;
    ``inf`` for ``l1 = l2``.
    """
    l1, l2, sb2 = arm.l1, arm.l2, arm.sigma_bar_sq
    if l1 == l2:
        return math.inf
    return arm.gamma * math.sqrt((l1 ** 2 - sb2) * (l2 ** 2 - sb2)) / abs(l1 ** 2 - l2 ** 2)


def rotation_limit_check(arm: TwoLinkArm, runs: Dict[str, Run], target: TurningTarget) -> None:
    """The standard MCBF's cap against the target's rate, and scalar OD's decay while the
    target only turns (``t < t_in[0]``)."""
    cap = max_rotation_cap(arm)
    rate = abs(target.rate)
    print(f"\nRotation about joint 1 (Section 7 of the note): the target turns at {rate:g} rad/s; anywhere in the "
          f"safe set the standard MCBF certifies the rotation of the whole arm about joint 1 at most at "
          f"{cap:.4f} rad/s, rate / cap = {rate / cap:.4f}")
    turning = runs["scalar"].t[:-1] < target.t_in[0] - runs["scalar"].dt / 2
    if turning.any():
        omega = runs["scalar"].Omega[turning, 0, 0] * 2          # scalar OD: Omega = (omega/2) I
        start = runs["scalar"].theta[0]
        print(f"  while the target only turns (t < {target.t_in[0]:g} s): scalar OD's omega from {omega.min():.4f} "
              f"to {omega.max():.4f}, against rate / (cap at theta0) = {rate / rotation_cap(arm, start):.4f}")
        for key in ("standard", "scalar", "matrix"):
            run = runs[key]
            k = int(np.count_nonzero(turning))
            print(f"    {figures.TITLES[key]:13s}: rotation about joint 1 by t = {run.t[k]:g} s: "
                  f"{turned(run.point)[k]:.4f} rad of the end effector (target {rate * run.t[k]:.4f})")


def bisect_rotation_cap(arm: TwoLinkArm, theta, hi: float = 1e3) -> float:
    """The same rate by bisection on ``lambda_min(Hdot(w, w) + alpha(H)) >= 0``."""
    def certified(w):
        return np.linalg.eigvalsh(arm.Hdot(theta, (w, w)) + arm.alpha_H(theta))[0] >= -1e-13
    if certified(hi):
        return math.inf
    a, b = 0.0, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        a, b = (mid, b) if certified(mid) else (a, mid)
    return a


def rotation_cap_check(arm: TwoLinkArm, theta0, target: TurningTarget, gain: float, duration: float,
                       dt: float, rho_omega: float) -> None:
    """The standard MCBF's cap on rotation about joint 1, in closed form and in closed loop."""
    cap0 = rotation_cap(arm, theta0)
    print("\nRotation cap of the standard MCBF (Section 7 of the note):")
    print(f"  at theta0: sqrt(det alpha(H))/(lam1 - lam0) = {cap0:.6f} rad/s, "
          f"bisection on its matrix inequality = {bisect_rotation_cap(arm, theta0):.6f} rad/s")
    if math.isinf(cap0):
        print("  the eigenvalues of H coincide at theta0 (l1 = l2, theta2 - theta1 = pi/2): no cap there")
        return
    if arm.l1 != arm.l2:
        print(f"  anywhere in the safe set: gamma sqrt((l1^2 - sb^2)(l2^2 - sb^2))/|l1^2 - l2^2| = "
              f"{max_rotation_cap(arm):.6f} rad/s")
    print(f"  standard MCBF with the target turning just below and just above the cap at theta0, "
          f"{duration:g} s:")
    for factor in (0.99, 1.01):
        rate = factor * cap0
        probe = TurningTarget(target.r0, target.r0, target.phi0, rate, t_in=target.t_in, t_out=target.t_out)
        trace = rollout(arm, tracking_controller(arm, probe, gain), theta0, dt=dt, horizon=duration,
                        decay="standard", omega_d=1.0, rho_omega=rho_omega, recovery=True)
        late = trace.t >= duration / 2
        achieved = np.polyfit(trace.t[late], trace.theta[late].mean(axis=1), 1)[0]
        behind = rate * duration - (trace.theta[-1].mean() - trace.theta[0].mean())
        diff = trace.theta[:, 1] - trace.theta[:, 0]
        print(f"    target at {factor:g} x cap = {rate:.5f}: rotation rate over the second half {achieved:.5f}, "
              f"behind by {behind:.4f} rad at the end, theta2 - theta1 in [{diff.min():.4f}, {diff.max():.4f}], "
              f"cap there at the end {rotation_cap(arm, trace.theta[-1]):.5f}")


# ---------------------------------------------------------------- drawing
def _violated_style() -> dict:
    from matplotlib.colors import to_rgba

    return dict(facecolor=to_rgba(figures.COLORS["standard"], 0.12),
                edgecolor=to_rgba(figures.COLORS["standard"], 0.55), hatch="////", lw=0, zorder=0)


OUTSIDE_STYLE = dict(facecolor="0.82", edgecolor="none", lw=0, zorder=0)


def _shade_bands(ax, arm: TwoLinkArm, lim: float, lw: float) -> None:
    """Outside the workspace (grey), the bands where H is violated (hatched), the boundary
    circles of the safe set (dashed) and of the workspace (solid), as in figures.workspace."""
    from matplotlib.patches import Circle, Wedge

    r_in, r_out = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    corner = 1.01 * math.hypot(lim, lim)
    ax.add_patch(Wedge((0, 0), corner, 0, 360, width=corner - r_out, **OUTSIDE_STYLE))
    if r_in > 0:
        ax.add_patch(Circle((0, 0), r_in, **OUTSIDE_STYLE))
    for outer, inner in ((r_out, r_hi), (r_lo, r_in)):
        if outer > inner:
            ax.add_patch(Wedge((0, 0), outer, 0, 360, width=outer - inner, **_violated_style()))
    for radius in (r_lo, r_hi):
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor=figures.COLORS["standard"],
                            linestyle="--", lw=lw, zorder=1))
    for radius in (r_in, r_out):
        if radius > 0:
            ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="0.2", lw=lw, zorder=1))


def _legend_entries(lw: float, marker_size: float, marker_width: float):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color="0.2", lw=lw),
        Patch(**{k: v for k, v in OUTSIDE_STYLE.items() if k != "zorder"}),
        Line2D([], [], color=figures.COLORS["standard"], linestyle="--", lw=lw),
        Patch(**{k: v for k, v in _violated_style().items() if k != "zorder"}),
        Line2D([], [], color="black", lw=lw * 0.8, ls=(0, (4, 3))),
        Line2D([], [], color="black", marker="x", ls="", ms=marker_size, mew=marker_width),
    ]
    labels = ["workspace boundary", "outside the workspace", r"$\lambda_{\min}(\mathbf{H})=0$",
              r"$\lambda_{\min}(\mathbf{H})<0$", "target path", "target point"]
    return handles, labels


def _snapshot_spacing(duration: float) -> float:
    for spacing in SNAPSHOT_SPACINGS:
        if duration / spacing <= MAX_SNAPSHOTS:
            return spacing
    return duration / MAX_SNAPSHOTS


def _title(case: str, target: TurningTarget, gain: float, arm: TwoLinkArm) -> str:
    return (f"{case}: target turning about joint 1 at {target.rate:g} rad/s, distance from joint 1 "
            f"{target.r0:.3g} $\\to$ {target.r1:.3g} $\\to$ {target.r0:.3g};  tracking gain $k={gain:g}$/s;  "
            f"$l_1={arm.l1:g}$, $l_2={arm.l2:g}$, $\\bar\\sigma={arm.sigma_bar:.4g}$")


def plot_paths(runs: Dict[str, Run], summaries: Dict[str, Dict[str, float]], arm: TwoLinkArm, title: str,
               outdir: Path, name: str):
    """One workspace panel per controller."""
    import matplotlib.pyplot as plt

    figures.apply_style()
    lim = 1.1 * arm.reach
    first = runs["nominal"]
    spacing = _snapshot_spacing(first.t[-1])
    fig = plt.figure(figsize=(6.9 * 2.3 * 0.73, 6.9 * 2.3 * 0.25), layout="constrained")
    grid = fig.add_gridspec(1, len(runs))
    for col, (key, run) in enumerate(runs.items()):
        ax = fig.add_subplot(grid[0, col])
        cmap = plt.get_cmap(figures.CMAPS[key])
        _shade_bands(ax, arm, lim, figures.REF_LW * 1.4)
        ax.plot(run.target[:, 0], run.target[:, 1], color="black", lw=figures.REF_LW,
                ls=(0, (4, 3)), zorder=2)
        ax.plot(run.point[:, 0], run.point[:, 1], color=figures.COLORS[key], lw=figures.LW, zorder=3)
        for tk in np.arange(0.0, run.t[-1] + 1e-9, spacing):
            k = min(int(round(tk / run.dt)), len(run.t) - 1)
            shade = cmap(0.35 + 0.6 * tk / run.t[-1])
            points = arm.arm_points(run.theta[k])
            ax.plot(points[:, 0], points[:, 1], "o-", color=shade, lw=0.8 * figures.SCALE,
                    ms=1.6 * figures.SCALE, alpha=0.8, zorder=4)
            ax.plot(*run.target[k], marker="x", color="black", ms=2.6 * figures.SCALE,
                    mew=0.5 * figures.SCALE, zorder=5)
        ax.plot(0, 0, marker="s", ms=2.4 * figures.SCALE, color="0.2", zorder=6)
        s = summaries[key]
        ax.set_title(figures.TITLES[key], fontsize=figures.LABEL_FS * 0.8)
        ax.text(0.02, 0.02, f"held margin {s['held_margin']:+.1e}\nheld distance to target {s['held_error']:.3f}",
                transform=ax.transAxes, fontsize=figures.LEGEND_FS * 0.6, va="bottom", zorder=7,
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round"))
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_color("0.75")

    fig.suptitle(f"{title}; arm snapshots every {spacing:g} s", fontsize=figures.LABEL_FS * 0.7)
    handles, labels = _legend_entries(figures.REF_LW * 1.4, 2.6 * figures.SCALE, 0.5 * figures.SCALE)
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels),
               fontsize=figures.LEGEND_FS * 0.75, frameon=False, handlelength=1.8, columnspacing=1.2)
    figures.save(fig, outdir, name)
    plt.close(fig)
    return fig


def _symlog_ticks(linthresh: float) -> List[float]:
    """0 and +-10^k every second decade from 1 down to 10 linthresh, clear of the linear band."""
    decades = [10.0 ** k for k in range(0, math.floor(math.log10(linthresh)) + 1, -2)
               if 10.0 ** k >= 10 * linthresh]
    return [-d for d in decades] + [0.0] + decades[::-1]


def _multiplier_lines(ax, runs: Dict[str, Run], lw: float):
    """||Omega - Omega_d||_F of scalar and matrix OD, and the scalar decay matrix OD's u would need."""
    lines = {}
    for key in ("scalar", "matrix"):
        lines[key], = ax.plot([], [], color=figures.COLORS[key], lw=lw,
                              label=rf"{figures.TITLES[key]}: $\|\Omega-\Omega_d\|_F$")
    lines["needed"], = ax.plot([], [], "k--", lw=lw * 0.8,
                               label=r"$\omega I/2$ that would certify matrix OD's $u$")
    ax.set_yscale("log")
    needed = (np.maximum(1.0, runs["matrix"].scalar_decay_needed) - 1.0) / math.sqrt(2)
    series = {"scalar": runs["scalar"].decay_deviation, "matrix": runs["matrix"].decay_deviation,
              "needed": needed}
    return lines, series


def plot_closed_loop(runs: Dict[str, Run], outdir: Path, name: str):
    """Distance to the target, margin, multiplier deviations and control deviation over time."""
    import matplotlib.pyplot as plt

    figures.apply_style()
    fig, axs = plt.subplots(4, 1, figsize=(6.9, 8.0), sharex=True)
    first = runs["nominal"]
    for ax in axs:
        for a, b in contact_intervals(first):
            ax.axvspan(a, b, color="0.93", zorder=0)
        ax.grid(alpha=0.3, linewidth=figures.GRID_LW)
        ax.tick_params(labelsize=figures.TICK_FS, width=0.5 * figures.SCALE, length=1.5 * figures.SCALE)
        ax.set_xlim(0, first.t[-1])
    for key, run in runs.items():
        style = dict(color=figures.COLORS[key], lw=figures.LW, label=figures.TITLES[key])
        axs[0].plot(run.t, run.error, **style)
        axs[1].plot(run.t, run.margin, **style)
        axs[3].plot(run.t[:-1], np.linalg.norm(run.u - run.u_nom, axis=1), **style)
    axs[0].plot(first.t, np.where(first.bound > 0, first.bound, np.nan), "k--", lw=figures.REF_LW * 1.3,
                label="distance from the target to the safe set")
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-6, 3 * max(run.error.max() for run in runs.values()))
    axs[0].set_ylabel(r"$|p-p^*|$", fontsize=figures.LABEL_FS)
    axs[0].legend(fontsize=figures.LEGEND_FS * 0.7, loc="lower center", ncols=5, handlelength=1.5,
                  columnspacing=0.8)
    smallest = min(runs[k].margin.min() for k in FILTERS)
    linthresh = (min(1e-4, 10 ** math.floor(math.log10(max(0.3 * smallest, 1e-9))))
                 if smallest > 0 else 1e-4)
    axs[1].set_yscale("symlog", linthresh=linthresh, linscale=0.5)
    axs[1].set_yticks(_symlog_ticks(linthresh))
    axs[1].axhline(0, color="k", lw=figures.REF_LW)
    axs[1].set_ylabel(r"$\lambda_{\min}(\mathbf{H})$", fontsize=figures.LABEL_FS)
    lines, series = _multiplier_lines(axs[2], runs, figures.LW)
    for key, line in lines.items():
        line.set_data(first.t[:-1], series[key])
    axs[2].relim()
    axs[2].autoscale_view()
    axs[2].set_ylabel(r"$\|\mathbf{\Omega}-\mathbf{\Omega}_d\|_F$", fontsize=figures.LABEL_FS)
    axs[2].legend(fontsize=figures.LEGEND_FS * 0.85, loc="lower center")
    axs[3].set_ylabel(r"$\|\mathbf{u}-\mathbf{u}_{\mathrm{nom}}\|$", fontsize=figures.LABEL_FS)
    axs[3].set_xlabel(r"$t$ [s]  (grey: the target is on or outside the boundary)",
                      fontsize=figures.LABEL_FS)
    for ax, tag in zip(axs, "abcd"):
        ax.text(0.99, 0.95, f"({tag})", transform=ax.transAxes, ha="right", va="top",
                fontsize=figures.LABEL_FS)
    fig.align_ylabels(axs)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.07, top=0.99, hspace=0.08)
    figures.save(fig, outdir, name)
    plt.close(fig)
    return fig


def render_video(runs: Dict[str, Run], arm: TwoLinkArm, title: str, out: Path) -> Path:
    """Animate the closed loops side by side: the arms on top, and underneath the margin, the
    distance to the target and the multiplier deviations. ``.mp4`` needs ffmpeg; any other
    suffix is written as an animated GIF."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    keys = list(runs)
    first = runs["nominal"]
    t = first.t
    with plt.rc_context({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 13}):
        fig = plt.figure(figsize=(16, 9), dpi=120)
        top = fig.add_gridspec(1, len(keys), left=0.02, right=0.98, top=0.88, bottom=0.5, wspace=0.06)
        bottom = fig.add_gridspec(1, 3, left=0.06, right=0.99, top=0.385, bottom=0.075, wspace=0.3)
        arm_axes = [fig.add_subplot(top[0, i]) for i in range(len(keys))]
        m_ax, err_ax, dec_ax = (fig.add_subplot(bottom[0, i]) for i in range(3))

        lim = 1.1 * arm.reach
        artists = {}
        for ax, key in zip(arm_axes, keys):
            run = runs[key]
            _shade_bands(ax, arm, lim, 1.2)
            ax.plot(run.target[:, 0], run.target[:, 1], color="k", lw=1.0,
                    ls=(0, (4, 3)), zorder=2)
            trace, = ax.plot([], [], color=figures.COLORS[key], lw=2.0, zorder=3)
            link, = ax.plot([], [], "o-", color=figures.COLORS[key], lw=4.5, ms=8, mec="white", mew=1.2,
                            zorder=5, solid_capstyle="round")
            mark, = ax.plot([], [], "x", color="k", ms=11, mew=2.2, zorder=6)
            ax.plot(0, 0, "s", color="0.15", ms=8, zorder=7)
            label = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=11, va="bottom",
                            family="monospace",
                            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round"))
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])
            ax.set_title(figures.TITLES[key] + (" (unfiltered)" if key == "nominal" else ""), fontsize=16)
            for spine in ax.spines.values():
                spine.set_color(figures.COLORS[key])
                spine.set_linewidth(2.5)
            artists[key] = (trace, link, mark, label)
        handles, labels = _legend_entries(1.2, 9, 2)
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.462), ncols=len(labels),
                   fontsize=11.5, frameon=False, handlelength=1.8, columnspacing=1.4)

        for ax in (m_ax, err_ax, dec_ax):
            for a, b in contact_intervals(first):
                ax.axvspan(a, b, color="0.93", zorder=0)
            ax.set_xlim(0, t[-1])
            ax.set_xlabel(r"$t$ [s]", fontsize=12)
            ax.grid(alpha=0.3)
        err_ax.plot(t, np.where(first.bound > 0, first.bound, np.nan), color="k", lw=1.2,
                    ls=(0, (4, 3)), zorder=4, label="target to safe set")
        margin_lines, error_lines = {}, {}
        for key in keys:
            style = dict(color=figures.COLORS[key], lw=2.0, label=figures.TITLES[key])
            margin_lines[key], = m_ax.plot([], [], **style)
            error_lines[key], = err_ax.plot([], [], **style)
        smallest = min(runs[k].margin.min() for k in FILTERS)
        linthresh = (min(1e-4, 10 ** math.floor(math.log10(max(0.3 * smallest, 1e-9))))
                     if smallest > 0 else 1e-4)
        m_ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
        margins = np.concatenate([runs[k].margin for k in keys])
        m_ax.set_ylim(min(1.3 * margins.min(), -linthresh), max(1.3 * margins.max(), 1.0))
        m_ax.axhline(0, color="k", lw=0.8)
        m_ax.set_ylabel(r"$\lambda_{\min}(\mathbf{H})$  (symlog)")
        m_ax.set_title("safety margin", fontsize=13)
        m_ax.legend(loc="lower left", fontsize=9.5, ncols=2)
        err_ax.set_yscale("log")
        err_ax.set_ylim(1e-4, 2 * max(runs[k].error.max() for k in keys))
        err_ax.set_ylabel(r"$|p-p^*|$, end effector to target", fontsize=10.5)
        err_ax.set_title("distance to the target", fontsize=13)
        err_ax.legend(loc="lower right", fontsize=9)
        dec_lines, series = _multiplier_lines(dec_ax, runs, 2.0)
        finite = np.concatenate([v[np.isfinite(v)] for v in series.values()])
        dec_ax.set_ylim(0.03, max(60.0, 2 * finite.max() if finite.size else 1.0))
        dec_ax.set_ylabel(r"multiplier deviation (scalar: $\Omega=\omega I/2$)", fontsize=10.5)
        dec_ax.set_title("decay multipliers", fontsize=13)
        dec_ax.legend(loc="lower center", fontsize=9)
        cursors = [ax.axvline(0, color="0.3", lw=1) for ax in (m_ax, err_ax, dec_ax)]
        header = fig.suptitle("", fontsize=17, y=0.985)
        fig.text(0.5, 0.925, title + ";  grey: the target is on or outside the boundary",
                 ha="center", fontsize=12.5)

        def phase(k):
            if first.target_outside[k]:
                return "the target is on or outside the boundary of the safe set"
            return "the target is inside the safe set"

        frames = list(range(0, len(t), VIDEO_EVERY))
        if frames[-1] != len(t) - 1:
            frames.append(len(t) - 1)

        def update(i):
            n = i + 1
            for key in keys:
                run = runs[key]
                trace, link, mark, label = artists[key]
                trace.set_data(run.point[:n, 0], run.point[:n, 1])
                points = arm.arm_points(run.theta[i])
                link.set_data(points[:, 0], points[:, 1])
                mark.set_data([run.target[i, 0]], [run.target[i, 1]])
                margin = run.margin[i] if abs(run.margin[i]) >= 1e-12 else 0.0
                label.set_text(f"to target {run.error[i]:6.3f}\nmargin    {margin:+.4f}")
                margin_lines[key].set_data(t[:n], run.margin[:n])
                error_lines[key].set_data(t[:n], run.error[:n])
            j = min(i, len(t) - 2) + 1
            for key, line in dec_lines.items():
                line.set_data(t[:j], series[key][:j])
            for cursor in cursors:
                cursor.set_xdata([t[i], t[i]])
            header.set_text(f"t = {t[i]:5.2f} s   ({phase(i)})")
            return []

        ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".mp4":
            writer = animation.FFMpegWriter(fps=VIDEO_FPS, bitrate=6000,
                                            extra_args=["-pix_fmt", "yuv420p"])
            ani.save(out, writer=writer, dpi=120)
        else:
            ani.save(out, writer=animation.PillowWriter(fps=VIDEO_FPS), dpi=60)
        plt.close(fig)
    return out


# ------------------------------------------------------------------ hero figure
#: Drawn at its printed size: 8 pt labels and colour-bar ticks, 7 pt axis ticks and contour labels, hairline axes.
HERO_RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.linewidth": 0.4, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.minor.width": 0.3, "ytick.minor.width": 0.3, "xtick.major.size": 1.8,
    "ytick.major.size": 1.8, "xtick.minor.size": 1.0, "ytick.minor.size": 1.0,
    "xtick.major.pad": 1.2, "ytick.major.pad": 1.2, "axes.labelpad": 1.0, "axes.titlepad": 2.0,
    "hatch.linewidth": 0.25, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 600,
    "savefig.bbox": "standard",                             # exactly HERO_SIZE
}
HERO_LW = 0.9
HERO_LS = {"standard": (0, (5.0, 1.2, 1.0, 1.2)), "scalar": (0, (4.0, 1.4)), "matrix": "-"}
HERO_Z = {"standard": 3, "matrix": 4, "scalar": 5}           # scalar OD over matrix OD where they overlap
HERO_TARGET = dict(color="black", lw=0.45, ls=(0, (3.0, 2.0)))
HERO_BOUND = dict(color="black", lw=0.8, ls=(0, (0.8, 1.2)))
HERO_VIOLATED = dict(facecolor="white", edgecolor="0.6", hatch="////////////", lw=0)
HERO_CONTACT = "0.92"
HERO_COLORS = {**figures.COLORS, "standard": "#6a3d9a"}      # the hero's standard MCBF is purple
HERO_INITIAL = "#d62728"                                     # the initial configuration
HERO_FINAL = "#2ca02c"                                       # the target at the end
HERO_ZOOM = "#ff00ff"                                        # the zoomed region and the inset's frame
HERO_POINT = dict(marker="*", color="black", ms=6.5, mec="white", mew=0.45, ls="")   # the configuration of (b)
#: Layout of (a), in points: the radius of the workspace circle, against the right edge; the height of
#: its centre above the bottom edge; the width and height of the inset left of it; its magnification;
#: and the white margin kept at the left and right edges.
HERO_A = dict(radius=44.6, centre=65.2, inset=(27.0, 50.0), magnify=9.0, edge=0.5)
#: Layout of (b), in points: the margins of the plot (left, right, bottom, top), the height of the
#: unsafe strip below the blue line (two lines of text), and the top of the plot in rad/s.
HERO_B = dict(margins=(18.0, 31.5, 20.5, 3.4), strip=18.5, top=2.0)
#: Where each contour of (b) is labelled: at this |theta2dot - theta1dot|, rad/s, on the branch to the right
#: (+1) or left (-1), clear of the others; omega = 1 is labelled level, just beyond its set.
HERO_LABEL_AT = {2.0: (1.5, 1), 4.0: (0.75, 1), 10.0: (0.5, -1), 40.0: (0.1, -1)}
HERO_THETA2_AT = (0.06, 0.3)                                  # the theta2 label: this far along its arc, this far out
HERO_HALO = 1.6                                               # pt: the white outline behind some labels
HERO_SCRIPT = 0.875                                           # sub- and superscripts: 7 pt on 8 pt text


def _halo(draw, *args, zorder: float = 3, halo: float = HERO_HALO, **kwargs):
    """Draw text with ``draw`` (``ax.text`` or ``ax.annotate``) over a white outline of itself, which
    knocks out the hatching or lines beneath it. Outline and glyphs are drawn from the same path, so
    they stay in register (a separate white copy under mathtext text lands about 0.6 pt low)."""
    from matplotlib import patheffects

    outline = [patheffects.withStroke(linewidth=halo, foreground="white")]
    return draw(*args, zorder=zorder, path_effects=outline, **kwargs)


@contextlib.contextmanager
def _script_scale(factor: float):
    """Draw sub- and superscripts at ``factor`` of their base size: mathtext's own 0.7 would print the
    subscripts of 8 pt labels at 5.6 pt. A no-op where matplotlib lays out mathtext otherwise."""
    from matplotlib import mathtext

    try:
        from matplotlib import _mathtext
    except ImportError:
        yield
        return
    cache = getattr(getattr(mathtext.MathTextParser, "_parse_cached", None), "cache_clear", lambda: None)
    old = getattr(_mathtext, "SHRINK_FACTOR", None)
    if old is None:
        yield
        return
    _mathtext.SHRINK_FACTOR = factor
    cache()                                                  # layouts cached at the old size
    try:
        yield
    finally:
        _mathtext.SHRINK_FACTOR = old
        cache()


def _hero_bands(ax, arm: TwoLinkArm) -> None:
    """The bands where H is violated (hatched), bounded by the workspace circles and the
    boundary circles of the safe set."""
    from matplotlib.patches import Circle, Wedge

    r_in, r_out = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    for outer, inner in ((r_out, r_hi), (r_lo, r_in)):
        if outer > inner:
            ax.add_patch(Wedge((0, 0), outer, 0, 360, width=outer - inner, zorder=0, **HERO_VIOLATED))
    for radius in (r_in, r_lo, r_hi, r_out):
        if radius > 0:
            ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="0.3", lw=0.3, zorder=1))


def _hero_certified(ax, arm: TwoLinkArm, boundary: str, span: float = 2.0, safe: float = 3.0,
                    strip: float = 0.4, grid: int = 401):
    """The joint velocities each filter certifies at a boundary configuration, as
    ``scripts/certified_velocities_demo.py`` draws them: the smallest scalar decay omega*
    over the rotation of the whole arm and ``theta2dot - theta1dot``, from ``strip`` on the
    unsafe side of the blue line to ``safe`` on the other."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    import certified_velocities_demo as cert                 # the same closed forms (note, Section 5)

    standard_color, blue = HERO_COLORS["standard"], HERO_COLORS["matrix"]
    diff = cert.boundary_value(arm, boundary)
    W, Sv = np.meshgrid(np.linspace(-span, span, grid), np.linspace(-safe, safe, grid))
    z00, z01 = cert.note_blocks(arm, diff, W, Sv)
    omega_star = cert.scalar_bound_boundary(arm, z00, z01)
    cmap = plt.get_cmap("Oranges").copy()
    cmap.set_bad("0.86")
    image = ax.pcolormesh(W, Sv, np.where(z00 > 0, np.maximum(omega_star, 1.0), np.nan), norm=LogNorm(1, 1e3),
                          cmap=cmap, shading="auto", rasterized=True)
    standard = [omega == 1.0 for omega in cert.DISK_OMEGAS]
    lines = ax.contour(W, Sv, np.where(z00 > 0, omega_star, np.nan), levels=list(cert.DISK_OMEGAS),
                       colors=[standard_color if s else "0.15" for s in standard],
                       linewidths=[0.8 if s else 0.35 for s in standard])
    up = math.sin(2 * diff) < 0                              # Z00 > 0 below the line: unsafe above it
    side = -1.0 if up else 1.0                               # the sign of theta2dot - theta1dot on the safe side
    across = np.linspace(-span, span, 4001)
    spots = []                                               # each label where its contour crosses its height
    for omega, (height, end) in HERO_LABEL_AT.items():
        blocks_at = cert.note_blocks(arm, diff, across, np.full_like(across, side * height))
        cross = np.flatnonzero(np.diff(np.sign(cert.scalar_bound_boundary(arm, *blocks_at) - omega)))
        if cross.size:
            spots.append((across[cross[-1] if end > 0 else cross[0]], side * height))
    ax.clabel(lines, levels=list(HERO_LABEL_AT), fmt="%g", fontsize=7,
              inline_spacing=1.5, manual=spots)
    ax.axhline(0, color=blue, lw=1.6, zorder=3)
    low, high = (-safe, strip) if up else (-strip, safe)
    ax.set(xlim=(-span, span), ylim=(low, high))
    inside = (z00 > 0) & (omega_star <= 1.0)
    if inside.any():                                         # along the set's long axis, as printed
        points = ax.transData.transform(np.column_stack([W[inside], Sv[inside]]))
        centre = points.mean(axis=0)
        long_axis = np.linalg.eigh(np.cov((points - centre).T))[1][:, -1]
        angle = (math.degrees(math.atan2(long_axis[1], long_axis[0])) + 90.0) % 180.0 - 90.0
        ax.text(*ax.transData.inverted().transform(centre), "Standard MCBF", rotation=angle,
                rotation_mode="anchor", ha="center", va="center_baseline", fontsize=8, color=standard_color,
                zorder=5)
        tip = np.argmax(Sv[inside] * side)                   # omega = 1 just beyond the set's far end, level
        ax.annotate(r"$\omega=1$", (W[inside][tip], Sv[inside][tip]), xytext=(0.0, 2.0 * side),
                    textcoords="offset points", ha="center", va="bottom" if side > 0 else "top", fontsize=7,
                    color=standard_color, zorder=5)
    ax.annotate("Matrix OD only", (-span, 0.0), xytext=(1.5, -1.6 * side), textcoords="offset points",
                ha="left", va="top" if side > 0 else "bottom", fontsize=8, color=blue, zorder=4)
    # below the line the rates violate the paper's matrix tangent-cone condition Z00 >= 0;
    # Z00 is 1 x 1 here, so that is Z00 < 0
    ax.annotate(r"Unsafe: $\mathbf{Z}_{00}\prec\mathbf{0}$", (span, low if side > 0 else high),
                xytext=(-1.5, 1.4 * side), textcoords="offset points", ha="right",
                va="bottom" if side > 0 else "top", fontsize=8, color="0.25", zorder=4)
    ax.set_xticks(np.arange(-math.floor(span), math.floor(span) + 1))
    ax.set_yticks(np.arange(0, math.floor(safe) + 1) * side)
    ax.set_xlabel(r"$(\dot\theta_1+\dot\theta_2)/2$")
    ax.set_ylabel(r"$\dot\theta_2-\dot\theta_1$")
    return image                                             # the caller adds the colour bar after layout


def _hero_angles(ax, arm: TwoLinkArm, theta, color) -> None:
    """Mark the absolute joint angles of one arm: theta1 at joint 1 and theta2 at joint 2, each
    counter-clockwise from a horizontal reference (dashed) to its link, with an arrowhead on the link."""
    from matplotlib.patches import Arc

    for centre, angle, name, radius, (at, gap) in ((np.zeros(2), theta[0], r"$\theta_1$", 0.30, (0.5, 0.2)),
                                                   (arm.joint2(theta), theta[1], r"$\theta_2$", 0.17, HERO_THETA2_AT)):
        a = math.degrees(angle) % 360.0                                   # counter-clockwise, in [0, 360)
        ax.plot([centre[0], centre[0] + 1.6 * radius], [centre[1], centre[1]], color=color, lw=0.4,
                ls=(0, (2.5, 1.5)), zorder=12)
        ax.add_patch(Arc(centre, 2 * radius, 2 * radius, theta1=0.0, theta2=a, color=color, lw=0.5, zorder=12))
        tip = centre + radius * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        back = math.radians(a - 10.0)                                     # a short step back along the arc
        ax.annotate("", xy=tip, xytext=centre + radius * np.array([math.cos(back), math.sin(back)]), zorder=12,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.5, mutation_scale=4, shrinkA=0, shrinkB=0))
        mid = math.radians(at * a)                                        # the label, this far along the arc
        _halo(ax.text, *(centre + (radius + gap) * np.array([math.cos(mid), math.sin(mid)])), name, ha="center",
              va="center", fontsize=8, color=color, zorder=12)


def _hero_point(arm: TwoLinkArm, boundary: str) -> np.ndarray:
    """The end effector of the configuration of (b): ``theta1 = 0`` and ``theta2 - theta1`` on the boundary."""
    import certified_velocities_demo as cert

    return arm.end_effector(np.array([0.0, cert.boundary_value(arm, boundary)]))


def _hero_merged(ax, arm: TwoLinkArm, runs: Dict[str, Run], lim: float, boundary: str) -> None:
    """The three filters in one workspace: the target's path and each end effector's path,
    the initial configuration, and each filter's arm with the target at the end; the joint
    angles are marked on the standard MCBF's arm, and the configuration of (b) by a star."""
    first = runs["nominal"]
    _hero_bands(ax, arm)
    ax.plot(first.target[:, 0], first.target[:, 1], zorder=2, **HERO_TARGET)
    start = arm.arm_points(first.theta[0])
    ax.plot(start[:, 0], start[:, 1], "-o", color=HERO_INITIAL, lw=1.1, ms=2.2, mew=0, solid_capstyle="round",
            zorder=3)
    widths = {"standard": (0.6, 1.2), "matrix": (0.9, 1.7), "scalar": (0.5, 1.0)}   # scalar OD drawn over matrix OD
    for key in FILTERS:
        run = runs[key]
        color = HERO_COLORS[key]
        path_width, arm_width = widths[key]
        ax.plot(run.point[:, 0], run.point[:, 1], color=color, lw=path_width, solid_capstyle="round",
                zorder=HERO_Z[key])
        end = arm.arm_points(run.theta[-1])
        ax.plot(end[:, 0], end[:, 1], "-o", color=color, lw=arm_width, ms=arm_width + 1.0, mew=0,
                solid_capstyle="round", zorder=HERO_Z[key] + 3)
    _hero_angles(ax, arm, runs["standard"].theta[-1], "0.15")
    ax.plot(*first.target[-1], "x", color=HERO_FINAL, ms=3.8, mew=0.9, zorder=10)
    ax.plot(0, 0, "s", color="0.2", ms=1.6, zorder=11)
    point = _hero_point(arm, boundary)
    ax.plot(*point, zorder=13, **HERO_POINT)
    _halo(ax.annotate, "(b)", point, xytext=(-2.6, -5.6), textcoords="offset points", ha="center",
          va="top", fontsize=8, color="black", zorder=13)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    ax.axis("off")


def _hero_zoom(runs: Dict[str, Run], k: int) -> np.ndarray:
    """The centre of the inset: the middle of the box around scalar OD, matrix OD and the target at step ``k``."""
    points = np.array([runs["matrix"].point[k], runs["scalar"].point[k], runs["nominal"].target[k]])
    return (points.min(axis=0) + points.max(axis=0)) / 2


def _hero_inset(ins, ax, arm: TwoLinkArm, runs: Dict[str, Run], k: int, centre, half) -> None:
    """Magnify, in ``ins``, the rectangle of half-sizes ``half`` around ``centre`` and box it on
    the workspace ``ax``: at step ``k`` matrix OD slides along the boundary next to the target, scalar
    OD stands off it and trails."""
    from matplotlib.patches import Circle, Rectangle, Wedge

    first = runs["nominal"]
    _, r_out = arm.workspace_radii
    _, r_hi = arm.safe_radius_interval
    ins.add_patch(Wedge((0, 0), r_out, 0, 360, width=r_out - r_hi, zorder=0, **HERO_VIOLATED))
    ins.add_patch(Circle((0, 0), r_hi, fill=False, edgecolor="0.3", lw=0.4, zorder=1))
    ins.plot(first.target[:, 0], first.target[:, 1], zorder=2, **HERO_TARGET)
    for key, width in (("matrix", 1.0), ("scalar", 0.7)):
        run = runs[key]
        ins.plot(run.point[:, 0], run.point[:, 1], color=HERO_COLORS[key], lw=width, zorder=HERO_Z[key])
        ins.plot(*run.point[k], "o", color=HERO_COLORS[key], ms=2.6, mew=0, zorder=6)
    ins.plot(*first.target[k], "x", color="black", ms=3.2, mew=0.7, zorder=7)
    half = np.asarray(half, dtype=float)
    ins.set(xlim=(centre[0] - half[0], centre[0] + half[0]), ylim=(centre[1] - half[1], centre[1] + half[1]),
            aspect="equal", xticks=[], yticks=[])
    for spine in ins.spines.values():
        spine.set(color=HERO_ZOOM, linewidth=0.7)
    ax.add_patch(Rectangle(centre - half, *(2 * half), fill=False, edgecolor=HERO_ZOOM, lw=0.6, zorder=20))


def _hero_distance(ax, runs: Dict[str, Run], summaries: Dict[str, Dict[str, float]]) -> None:
    """Distance from each end effector to the target, and the least any safe arm can have:
    the distance from the target to the safe set, nonzero only while the target is outside it."""
    first = runs["nominal"]
    t = first.t
    for a, b in contact_intervals(first):
        ax.axvspan(a, b, color=HERO_CONTACT, lw=0, zorder=0)
    for key in FILTERS:
        ax.plot(t, runs[key].error, color=HERO_COLORS[key], lw=HERO_LW, ls=HERO_LS[key], zorder=HERO_Z[key])
    ax.plot(t, first.bound, zorder=6, **HERO_BOUND)
    top = 2.0 * max(summaries["scalar"]["held_error"], first.bound.max())
    ax.set_ylim(0, top)
    if runs["standard"].error.max() > top:
        ax.text(0.97, 0.95, "Standard MCBF\noff scale", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="0.25", linespacing=1.0)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel("Distance to the target")
    ax.grid(alpha=0.35, lw=0.35)


def plot_hero(runs: Dict[str, Run], summaries: Dict[str, Dict[str, float]], arm: TwoLinkArm,
              target: TurningTarget, outdir: Path, name: str) -> List[Path]:
    """The paper's figure as separate subfigures, each HERO_SIZE at its printed size and without
    a title, for LaTeX to import: ``name_a`` the three closed loops in one workspace, with the
    legend and a magnified inset of the boundary in the middle of the held window; ``name_b`` the joint
    velocities each filter certifies at the boundary the target pushes into. Also ``name_tracking``,
    the distance to the target, which is not in the paper. Each is written as ``.pdf`` and ``.png``; returns the
    PDF paths."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _, r_hi = arm.safe_radius_interval
    boundary = "outer" if target.r1 > r_hi else "inner"
    zoom = int(round(sum(held_window(target)) / 2 / runs["nominal"].dt))   # the middle of the held window
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, stem):
        for suffix in (".pdf", ".png"):
            fig.savefig(outdir / f"{stem}{suffix}", dpi=600 if suffix == ".png" else None)
        plt.close(fig)
        written.append(outdir / f"{stem}.pdf")

    with plt.rc_context(HERO_RC), _script_scale(HERO_SCRIPT):
        width, height = HERO_SIZE
        across, tall = 72.0 * width, 72.0 * height            # points
        fig = plt.figure(figsize=HERO_SIZE)                  # (a): the workspace right, legend and inset left
        radius, centre = HERO_A["radius"], HERO_A["centre"]
        _, r_out = arm.workspace_radii
        scale = radius / r_out                               # points per unit length
        lim = r_out + 0.3 / scale                            # the outer circle's line just inside the axes
        half = lim * scale
        ax = fig.add_axes([(across - HERO_A["edge"] - 2 * half) / across, (centre - half) / tall, 2 * half / across,
                           2 * half / tall])
        _hero_merged(ax, arm, runs, lim, boundary)
        handles = [Line2D([], [], color=HERO_COLORS[key], lw=1.2) for key in FILTERS]
        labels = [figures.TITLES[key] for key in FILTERS]
        style = dict(frameon=False, borderpad=0.1, borderaxespad=0.0, handlelength=1.1, handletextpad=0.35,
                     labelspacing=0.22)
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.0), **style)   # the filters
        handles = [Line2D([], [], marker="x", ms=3.2, mew=0.7, **HERO_TARGET),
                   Line2D([], [], color=HERO_FINAL, marker="x", ms=3.8, mew=0.9, ls=""),
                   Line2D([], [], color=HERO_INITIAL, lw=1.1, marker="o", ms=2.2, mew=0), Patch(**HERO_VIOLATED)]
        labels = ["Target", "Final target", "Initial configuration", r"$\lambda_{\min}(\mathbf{H})<0$"]
        fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.0), **style)   # the rest
        spot = _hero_zoom(runs, zoom)
        wide, high = HERO_A["inset"]                         # the inset, left of the zoomed spot
        low = centre + spot[1] * scale - high / 2
        ins = fig.add_axes([(HERO_A["edge"] + 0.7) / across, low / tall, wide / across, high / tall])
        _hero_inset(ins, ax, arm, runs, zoom, spot, np.array([wide, high]) / 2 / HERO_A["magnify"] / scale)
        save(fig, f"{name}_a")

        fig = plt.figure(figsize=HERO_SIZE)                  # (b): certified joint rates and a colour bar
        left, right, bottom, top = HERO_B["margins"]
        ax = fig.add_axes([left / across, bottom / tall, (across - left - right) / across,
                           (tall - bottom - top) / tall])
        up = tall - bottom - top                             # the height of the plot, points
        image = _hero_certified(ax, arm, boundary, span=2.0, safe=HERO_B["top"],
                                strip=HERO_B["top"] * HERO_B["strip"] / (up - HERO_B["strip"]))
        box = ax.get_position()
        bar = fig.colorbar(image, cax=fig.add_axes([box.x1 + 2.5 / across, box.y0, 4.0 / across, box.height]),
                           extend="max", extendfrac=0.04)
        bar.ax.set_ylabel(r"Smallest certifying $\omega$", labelpad=1.5)   # Proposition 2's bound; no symbol
        bar.ax.tick_params(which="both", width=0.4, pad=1.0, labelsize=8)   # exponents at 7 pt
        bar.outline.set_linewidth(0.4)
        save(fig, f"{name}_b")

        fig = plt.figure(figsize=HERO_TRACKING_SIZE)         # the distance to the target (not in the paper)
        ax = fig.add_axes([0.16, 0.17, 0.8, 0.78])
        _hero_distance(ax, runs, summaries)
        handles = [Line2D([], [], color=HERO_COLORS[key], lw=HERO_LW, ls=HERO_LS[key]) for key in FILTERS]
        handles += [Line2D([], [], **HERO_BOUND), Patch(facecolor=HERO_CONTACT, edgecolor="none")]
        labels = [figures.TITLES[key] for key in FILTERS] + ["Least possible", "Target outside the safe set"]
        ax.legend(handles, labels, loc="upper left", fontsize=7, handlelength=2.0, borderaxespad=0.3,
                  labelspacing=0.3, frameon=True, framealpha=0.9, edgecolor="0.8")
        save(fig, f"{name}_tracking")
    return written


def write_hero_tex(path: Path, arm: TwoLinkArm, theta0, target: TurningTarget, gain: float, duration: float,
                   summaries: Dict[str, Dict[str, float]], name: str) -> Path:
    """A LaTeX float for the hero figure, in the paper's macros, with the run's key numbers as comments."""
    s = summaries
    cap, start = max_rotation_cap(arm), rotation_cap(arm, theta0)
    r_lo, r_hi = arm.safe_radius_interval
    start_held, back = held_window(target)
    distance = max(0.0, r_lo - target.r1, target.r1 - r_hi)
    band = "outer" if target.r1 > r_hi else "inner"
    rate = abs(target.rate)
    omega = 1 + math.sqrt(2) * s["scalar"]["held_decay_deviation"]     # scalar OD: omega >= 1
    notes = [
        "Generated by scripts/closed_loop_demo.py (hero case; scripts/reproduce_paper.py runs it); numbers from this run.",
        (f"Arm l1 = {arm.l1:g}, l2 = {arm.l2:g}, sigma_bar = {arm.sigma_bar:g}, gamma = {arm.gamma:g}; "
         "H = J J^T - sigma_bar^2 I, J the Jacobian in absolute joint angles."),
        (f"Target: turns about joint 1 at {target.rate:g} rad/s; distance from joint 1 {target.r0:.4f} -> "
         f"{target.r1:.4f} over [{target.t_in[0]:g}, {target.t_in[1]:g}] s, back over "
         f"[{target.t_out[0]:g}, {target.t_out[1]:g}] s; {distance:.4f} beyond the safe set ({band} band)."),
        f"Nominal: u = J^-1 (pdot* + k (p* - p)), k = {gain:g}/s.",
        (f"Fastest rotation of the whole arm about joint 1 the standard MCBF certifies: {cap:.4f} rad/s "
         f"anywhere in the safe set, {start:.4f} rad/s at theta0."),
        (f"Held, {start_held:g} <= t < {back:g} s: distance to the target standard {s['standard']['held_error']:.3f}, "
         f"scalar OD {s['scalar']['held_error']:.4f}, matrix OD {s['matrix']['held_error']:.4f}, "
         f"least possible {distance:.4f};"),
        (f"  lambda_min(H) standard {s['standard']['held_margin']:.2e}, scalar OD {s['scalar']['held_margin']:.2e}, "
         f"matrix OD {s['matrix']['held_margin']:.2e}; scalar OD's omega {omega:.2f}."),
        (f"Rotation about joint 1 over the run: target {rate * duration:.2f}, standard {s['standard']['rotation']:.2f}, "
         f"scalar OD {s['scalar']['rotation']:.2f}, matrix OD {s['matrix']['rotation']:.2f} rad."),
    ]
    boundary = "outer" if band == "outer" else "inner"
    caption = (
        rf"Two-link arm with $l_1={arm.l1:g}$, $l_2={arm.l2:g}$ and $\bar\sigma={arm.sigma_bar:g}$. "
        rf"(a)~The end effectors (solid paths) track a target $\bp^\star$ (dashed; green $\times$ at the end) turning "
        rf"about joint~1 at ${rate:g}$\,rad/s. The target first turns inside the safe set, then moves "
        rf"${distance:.3f}$ beyond it into the {band} band where $\bH\not\succeq\bzero$ (hatched), and returns; "
        r"inside the safe set its path lies under those of scalar and matrix OD. Red: the initial configuration; "
        rf"colours: each arm at the end, $t={duration:g}$\,s, with the absolute joint angles $\theta_1$ and "
        r"$\theta_2$ marked on the standard MCBF's; the star marks the configuration of (b). "
        rf"The inset magnifies the magenta box at $t={0.5 * sum(held_window(target)):g}$\,s: matrix OD slides along "
        r"the boundary next to the target, while scalar OD stands off it and trails. "
        + (rf"The standard MCBF certifies rotating the whole arm about joint~1 at no more than ${cap:.2f}$\,rad/s "
           r"anywhere in the safe set and falls behind. " if cap < rate else "")
        + rf"(b)~Joint rates (rad/s) at the {boundary} boundary of the safe set ($\theta_1=0$), coloured by the "
        r"smallest scalar decay $\omega$ that certifies each, the bound of \cref{prop:scalarcost} (shown from "
        r"$\omega=1$). The standard MCBF ($\omega=1$) certifies only the purple set. Scalar OD certifies a rate on the "
        r"safe side of the blue line with a decay that grows without bound toward the line. The nonzero rates on the "
        r"blue line, rotations of the whole arm about joint~1, are certified by matrix OD only (\cref{prop:scalargap}). "
        r"Rates in grey violate the matrix tangent-cone condition~\eqref{eq:tangentcone}, here "
        r"$\bZ_{00}\prec\bzero$, and leave the safe set."
    )
    share = rf"{HERO_SHARE:g}\columnwidth"                   # the width HERO_SIZE is drawn at
    lines = [f"% {note}" for note in notes] + [
        r"\begin{figure}[t]",
        r"    \centering",
        rf"    \begin{{subfigure}}[t]{{{share}}}",
        r"        \centering",
        rf"        \includegraphics[width=\linewidth]{{Figures/{name}_a.pdf}}",
        r"        \caption{Closed loops}",
        r"        \label{fig:hero_closed_loop}",
        r"    \end{subfigure}\hfill",
        rf"    \begin{{subfigure}}[t]{{{share}}}",
        r"        \centering",
        rf"        \includegraphics[width=\linewidth]{{Figures/{name}_b.pdf}}",
        rf"        \caption{{Certified joint rates at the {boundary} boundary}}",
        r"        \label{fig:hero_rates}",
        r"    \end{subfigure}",
        rf"    \caption{{{caption}}}",
        r"    \label{fig:hero}",
        r"\end{figure}",
    ]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Figures are written to OUT/NAME_paths.png and OUT/NAME_closed_loop.png, and for the "
               "hero case also OUT/NAME_a, OUT/NAME_b and OUT/NAME_tracking (.pdf and .png) and OUT/NAME.tex; "
               "NAME defaults to fig_<case>. "
               "The hero case sets its own defaults for --l1 --l2 --sigma-bar --depth --turn --move --duration "
               f"({', '.join(f'{k} {v:g}' for k, v in HERO.items())}); the other cases use unit links.",
    )
    parser.add_argument("--case", choices=("hero", "inner", "outer", "graze", "rotate"), default="hero",
                        help="where the target goes; see the module docstring (default: hero)")
    parser.add_argument("--theta0", type=float, nargs=2, default=(0.0, math.pi / 2),
                        metavar=("THETA1", "THETA2"),
                        help="initial absolute joint angles, rad; the target starts at this end "
                             "effector (default: 0, pi/2)")
    parser.add_argument("--gain", type=float, default=paper.GAIN,
                        help="tracking gain k of the nominal controller, 1/s (default: 5)")
    parser.add_argument("--rate", type=float, default=paper.TIMELINE["rate"],
                        help="rate at which the target turns about joint 1, rad/s (default: 1)")
    parser.add_argument("--depth", type=float, default=None,
                        help="hero, inner and outer cases: how far into the band the target goes, as a "
                             "fraction of the band's width, in [0, 1) (default: 0.3)")
    parser.add_argument("--turn", type=float, default=None,
                        help="seconds the target only turns before its distance starts to move "
                             "(default: 1 hero, 0 otherwise)")
    parser.add_argument("--move", type=float, default=None,
                        help="seconds each move of the target's distance takes (default: 0.75 hero, 1.5 otherwise)")
    parser.add_argument("--duration", type=float, default=None,
                        help=f"duration, s, at least --turn + 2 --move + {MIN_HOLD + RETURN_END:g} "
                             "(default: 4.5 hero, 8 otherwise)")
    parser.add_argument("--l1", type=float, default=None, help="length of link 1 (default: 1)")
    parser.add_argument("--l2", type=float, default=None,
                        help="length of link 2 (default: 0.75 hero, 1 otherwise)")
    parser.add_argument(
        "--sigma-bar", type=float, default=None,
        help="lower bound on sigma_min(J), in (0, min(l1, l2)); default 0.6 in the hero case; in the "
             "other cases the boundary of the safe set is at theta2 - theta1 = pi/3 and 2 pi/3")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="rate of the class-K term alpha(H) = gamma H (default: 1)")
    parser.add_argument("--dt", type=float, default=paper.DT, help="time step, s (default: 0.01)")
    parser.add_argument("--rho-omega", type=float, default=paper.FILTER["rho_omega"],
                        help="penalty on omega - omega_d; rho_Omega = 2 rho_omega (default: 5e-3)")
    parser.add_argument("--video", type=Path, default=None,
                        help="also write an animation to this file (.mp4 needs ffmpeg; other "
                             "suffixes give a GIF)")
    parser.add_argument("--out", type=Path, default=ROOT / "figures",
                        help="output directory (default: figures/ in the repository)")
    parser.add_argument("--name", default=None,
                        help="file-name prefix without extension (default: fig_<case>)")
    args = parser.parse_args(argv)
    for key, value in (HERO if args.case == "hero" else DEFAULTS).items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    try:
        arm = TwoLinkArm(args.l1, args.l2, gamma=args.gamma, sigma_bar=args.sigma_bar)
    except ValueError as err:
        parser.error(str(err))
    for flag, value in (("--duration", args.duration), ("--dt", args.dt), ("--rho-omega", args.rho_omega)):
        if not (np.isfinite(value) and value > 0):
            parser.error(f"{flag} must be positive and finite; got {value:g}")
    steps = round(args.duration / args.dt)
    if not np.isclose(steps * args.dt, args.duration):
        parser.error("--dt must divide --duration")
    if args.gamma * args.dt >= 1:
        parser.error(f"--gamma * --dt must be below 1, since the recovery constraint makes the "
                     f"decay rate at least gamma; got {args.gamma * args.dt:g}")
    if not (np.isfinite(args.turn) and args.turn >= 0):
        parser.error(f"--turn must be nonnegative and finite; got {args.turn:g}")
    if not (np.isfinite(args.move) and args.move > 0):
        parser.error(f"--move must be positive and finite; got {args.move:g}")
    if args.duration < args.turn + 2 * args.move + MIN_HOLD + RETURN_END:
        parser.error(f"--duration must be at least --turn + 2 --move + {MIN_HOLD + RETURN_END:g} s")
    if not (np.isfinite(args.gain) and args.gain >= 0 and args.gain * args.dt < 1):
        parser.error(f"--gain must be nonnegative with --gain * --dt below 1; got {args.gain:g}")
    if not np.isfinite(args.rate):
        parser.error("--rate must be finite")
    if not 0 <= args.depth < 1:
        parser.error(f"--depth must lie in [0, 1); got {args.depth:g}")
    theta0 = np.asarray(args.theta0, dtype=float)
    if not np.all(np.isfinite(theta0)):
        parser.error("--theta0 must be finite")
    if arm.margin(theta0) < -MARGIN_TOL:
        parser.error(f"theta0 must lie in the safe set; its margin is {arm.margin(theta0):+.3e}")
    target = case_target(args.case, arm, theta0, args.depth, args.rate, args.duration, args.turn, args.move)
    name = args.name or f"fig_{args.case}"
    title = _title(args.case, target, args.gain, arm)

    lam = boundary_lambda(arm)
    lo, hi = arm.safe_theta_diff_interval
    r_lo, r_hi = arm.safe_radius_interval
    print(arm)
    print(f"safe set: {lo:.5f} <= theta2 - theta1 <= {hi:.5f}, {r_lo:.5f} <= |p| <= {r_hi:.5f}; "
          f"lambda = tr H = {lam:.5f} on the boundary")
    held_bound = max(0.0, r_lo - target.r1, target.r1 - r_hi)
    print(f"case {args.case}: from theta0 = ({theta0[0]:.4f}, {theta0[1]:.4f}), the target turns about "
          f"joint 1 at {target.rate:g} rad/s; its distance from joint 1 goes {target.r0:.4f} -> "
          f"{target.r1:.4f} over t in {target.t_in}, and back over {target.t_out}. While held there it is "
          f"{held_bound:.4f} from the safe set. {args.duration:g} s, dt = {args.dt:g} s")
    print(f"nominal: u_nom = J^-1 (pdot* + k (p* - p)), k = {args.gain:g}/s (odmcbf.tracking.tracking_controller)")
    print(f"filters: omega_d = 1, rho_omega = {args.rho_omega:g}, rho_Omega = {2 * args.rho_omega:g}, "
          f"recovery on; odmcbf.simulate.rollout")

    boundary_check(arm, target, args.duration, args.dt, args.rho_omega)
    if args.case == "rotate":
        rotation_cap_check(arm, theta0, target, args.gain, args.duration, args.dt, args.rho_omega)

    runs = simulate(arm, theta0, target, args.gain, args.duration, args.dt, args.rho_omega)
    summaries = {key: summarize(run, arm, target) for key, run in runs.items()}
    start_held, back = held_window(target)
    print(f"\nClosed loop ({len(runs['nominal'].u)} steps each; held = t in [{start_held:g}, {back:g}) s, "
          f"at most the last {SETTLE:g} s before the target comes back):")
    print_table(summaries)
    print("Beyond the least possible: |p - p*| minus the distance from the target to the safe set, which "
          "no safe arm can beat.")
    print("Angle behind the target: the signed angle about joint 1 from the end effector to the target.")
    print("||Omega - Omega_d||_F uses Omega = omega I/2 for scalar OD, for which it equals "
          "|omega - omega_d| / sqrt 2; the two penalties then agree.")
    print("Scalar decay for u: the smallest omega with Hdot(u) + omega alpha(H) >= 0 for the control "
          "applied (Proposition 2 inside the safe set, Proposition 1 on the boundary; inf if none).")
    print("Max decay rate x dt: 2 gamma lambda_max(sym Omega) dt, a first-order check of the sampling.")
    if args.case == "hero":
        rotation_limit_check(arm, runs, target)

    plot_paths(runs, summaries, arm, title, args.out, name + "_paths")
    plot_closed_loop(runs, args.out, name + "_closed_loop")
    print(f"\nwrote {args.out / (name + '_paths.png')}")
    print(f"wrote {args.out / (name + '_closed_loop.png')}")
    if args.case == "hero":
        for pdf in plot_hero(runs, summaries, arm, target, args.out, name):
            print(f"wrote {pdf} and {pdf.with_suffix('.png')}")
        print(f"wrote {write_hero_tex(args.out / (name + '.tex'), arm, theta0, target, args.gain, args.duration, summaries, name)}")
    if args.video is not None:
        print(f"wrote {render_video(runs, arm, title, args.video)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
