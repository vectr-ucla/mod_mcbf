"""Which joint velocities scalar OD and matrix OD certify at the boundary of the safe set, and at what decay.

The arm is ``TwoLinkArm(l1, l2, sigma_bar=...)`` from ``odmcbf/two_link.py`` with
the barrier ``H = J J^T - sigma_bar^2 I`` of ``two_link_manipulator_kinematics.md``
and ``alpha(H) = gamma H``. Velocities ``thetadot`` are drawn in the two directions
that note separates: horizontally the rotation of the whole arm about joint 1,
``w = (theta1dot + theta2dot)/2``, and vertically ``s = theta2dot - theta1dot``,
the only part that changes the margin. The configuration has ``theta1 = 0`` and
``theta2 - theta1`` on the inner boundary (``--boundary outer`` for the outer one).

The figure comes from the note's closed forms at a boundary configuration
(Section 5), with ``theta1dot = w - s/2`` and ``theta2dot = w + s/2``:

    Z00 = s l1^2 l2^2 sin 2(theta2 - theta1) / lambda,
    Z01 = -lambda (w - s/2) - s l2^2 (l2^2 + l1^2 cos 2(theta2 - theta1)) / lambda,
    Z11 = -Z00,                      lambda = l1^2 + l2^2 - 2 sigma_bar^2,

and from the note's scalar bound ``omega >= (Z00^2 + Z01^2) / (alpha(lambda) Z00)``
(Proposition 2 at a boundary point, valid for ``Z00 > 0``). It shows the smallest
scalar decay ``omega*`` that certifies each velocity. Its level set ``omega* <= omega``
is the disk ``(Z00 - omega gamma lambda/2)^2 + Z01^2 <= (omega gamma lambda/2)^2``,
and the disk ``omega = 1`` is what the standard MCBF certifies. Every disk touches
the line ``Z00 = 0`` only at the origin, so ``omega*`` is infinite on it and scalar
OD certifies ``Z00 > 0`` and the zero velocity (Proposition 1). Matrix OD certifies ``Z00 >= 0``
(Theorem 1): the extra line is the rotation about joint 1. Velocities with
``Z00 < 0`` lower the margin below 0: unsafe (grey). The script checks each closed
form against an independent numerical computation and prints the errors.

Examples, from the repository root (the script also runs from any other directory
when given its full path):

    python scripts/certified_velocities_demo.py
    python scripts/certified_velocities_demo.py --l1 1 --l2 0.6 --sigma-bar 0.3
    python scripts/certified_velocities_demo.py --boundary outer
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))               # import odmcbf from the repository root

import cvxpy as cp                                            # noqa: E402

from odmcbf import TwoLinkArm, figures                        # noqa: E402

#: Counter-clockwise quarter turn; v1 = QUARTER v0.
QUARTER = np.array([[0.0, -1.0], [1.0, 0.0]])
#: Nominal decay rate of both OD filters; Omega_d = (omega_d/2) I.
OMEGA_D = 1.0
#: Scalar decays whose certified disks are drawn as contours of omega*.
DISK_OMEGAS = (1.0, 2.0, 4.0, 10.0, 40.0)


# ------------------------------------------------------------ configurations
def boundary_value(arm: TwoLinkArm, boundary: str) -> float:
    lo, hi = arm.safe_theta_diff_interval
    return hi if boundary == "inner" else lo


def boundary_lambda(arm: TwoLinkArm) -> float:
    """``lambda = l1^2 + l2^2 - 2 sigma_bar^2``, the nonzero eigenvalue of ``H`` on the boundary."""
    return arm.l1 ** 2 + arm.l2 ** 2 - 2 * arm.sigma_bar_sq


def velocity(w, s) -> np.ndarray:
    """``(theta1dot, theta2dot)`` from the rotation ``w`` and ``s = theta2dot - theta1dot``."""
    return np.stack([np.asarray(w) - np.asarray(s) / 2, np.asarray(w) + np.asarray(s) / 2], axis=-1)


# ------------------------------------------------------------ closed forms (the note)
def note_blocks(arm: TwoLinkArm, diff: float, w, s):
    """``Z00, Z01`` at a boundary configuration from the note's Section 5 closed forms."""
    l1, l2, lam = arm.l1, arm.l2, boundary_lambda(arm)
    z00 = s * l1 ** 2 * l2 ** 2 * math.sin(2 * diff) / lam
    z01 = -lam * (w - s / 2) - s * l2 ** 2 * (l2 ** 2 + l1 ** 2 * math.cos(2 * diff)) / lam
    return z00, z01


def note_blocks_section4(arm: TwoLinkArm, diff: float, w, s):
    """``Z00, Z01`` from the note's Section 4 forms (before the boundary identity is used)."""
    l1, l2, lam = arm.l1, arm.l2, boundary_lambda(arm)
    t1, t2 = w - s / 2, w + s / 2
    z00 = (t2 - t1) * l1 ** 2 * l2 ** 2 * math.sin(2 * diff) / lam
    z01 = -(l1 ** 4 * t1 + l2 ** 4 * t2 + l1 ** 2 * l2 ** 2 * (t1 + t2) * math.cos(2 * diff)) / lam
    return z00, z01


def scalar_bound_boundary(arm: TwoLinkArm, z00, z01):
    """The note's ``omega* = (Z00^2 + Z01^2) / (alpha(lambda) Z00)``; ``inf`` for ``Z00 <= 0``."""
    z00 = np.asarray(z00, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z00 > 0, (z00 ** 2 + np.asarray(z01) ** 2) / (arm.gamma * boundary_lambda(arm) * z00),
                        np.inf)


# ------------------------------------------------------------ numerics (independent)
def numeric_blocks(arm: TwoLinkArm, theta, U: np.ndarray):
    """``Z00, Z01`` from the numerical eigenbasis of ``H`` and ``Hdot(u)``."""
    _, vec = np.linalg.eigh(arm.H(theta))
    v0 = vec[:, 0]
    V = np.column_stack([v0, QUARTER @ v0])
    G = [V.T @ Gj @ V for Gj in arm.LgH(theta)]
    return (U[..., 0] * G[0][0, 0] + U[..., 1] * G[1][0, 0],
            U[..., 0] * G[0][0, 1] + U[..., 1] * G[1][0, 1])


def lmi_min_eig(arm: TwoLinkArm, theta, u, omega: float) -> float:
    """``lambda_min(Hdot(u) + omega alpha(H))`` in the Cartesian basis."""
    return float(np.linalg.eigvalsh(arm.Hdot(theta, u) + omega * arm.alpha_H(theta))[0])


def bisect_scalar(arm: TwoLinkArm, theta, u, hi: float = 1e9) -> float:
    """Smallest ``omega >= 0`` with ``lambda_min(Hdot + omega alpha(H)) >= 0``, by bisection.

    ``inf`` if no ``omega`` up to ``hi`` works.
    """
    if lmi_min_eig(arm, theta, u, 0.0) >= 0:
        return 0.0
    if lmi_min_eig(arm, theta, u, hi) < 0:
        return math.inf
    a, b = 0.0, hi
    for _ in range(300):
        mid = 0.5 * (a + b) if a == 0.0 or b / a <= 4 else math.sqrt(a * b)
        if lmi_min_eig(arm, theta, u, mid) >= 0:
            b = mid
        else:
            a = mid
        if b - a <= 1e-14 * b:
            break
    return b


class MatrixMultiplier:
    """Matrix OD's smallest multiplier deviation at one state and velocity.

    ``min ||Omega - Omega_d||_F`` s.t. ``Hdot + Omega' alpha(H) + alpha(H) Omega >= 0`` and
    ``sym(Omega) >= Omega_d``.
    """

    def __init__(self) -> None:
        self.A = cp.Parameter((2, 2), symmetric=True)
        self.Hdot = cp.Parameter((2, 2), symmetric=True)
        Omega = cp.Variable((2, 2))
        X = self.Hdot + Omega.T @ self.A + self.A @ Omega
        Omega_d = 0.5 * OMEGA_D * np.eye(2)
        self.problem = cp.Problem(cp.Minimize(cp.norm(Omega - Omega_d, "fro")),
                                  [(X + X.T) / 2 >> 0, (Omega + Omega.T) / 2 >> Omega_d])

    def __call__(self, arm: TwoLinkArm, theta, u) -> float:
        A, Hdot = arm.alpha_H(theta), arm.Hdot(theta, u)
        self.A.value = 0.5 * (A + A.T)
        self.Hdot.value = 0.5 * (Hdot + Hdot.T)
        # Tight tolerances first; on the boundary alpha(H) is singular and the problem can be
        # nearly degenerate, so fall back to the solver's defaults, then give up (NaN).
        attempts = (("CLARABEL", dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)),
                    ("CLARABEL", {}), ("SCS", dict(eps=1e-9, max_iters=200000)))
        for solver, options in attempts:
            try:
                self.problem.solve(solver=solver, **options)
            except cp.error.SolverError:
                continue
            if self.problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return max(0.0, float(self.problem.value))
        return math.nan


# ------------------------------------------------------------ checks
def run_checks(arm: TwoLinkArm, boundary: str, span: float, rate: float, rng) -> dict:
    """Closed forms against numerics; returns the numbers the figure and the printout use."""
    diff = boundary_value(arm, boundary)
    theta_b = np.array([0.0, diff])
    lam = boundary_lambda(arm)
    out = {"theta_b": theta_b, "lambda": lam}

    # 1. The Z blocks: Section 5 and Section 4 forms against the numerical eigenbasis.
    W, S = np.meshgrid(np.linspace(-span, span, 201), np.linspace(-span, span, 201))
    z00, z01 = note_blocks(arm, diff, W, S)
    z00_4, z01_4 = note_blocks_section4(arm, diff, W, S)
    z00_n, z01_n = numeric_blocks(arm, theta_b, velocity(W, S))
    out["block_error"] = max(np.abs(z00 - z00_n).max(), np.abs(z01 - z01_n).max())
    out["section4_error"] = max(np.abs(z00 - z00_4).max(), np.abs(z01 - z01_4).max())

    # 2. The certified sets: disk and half-plane membership against the LMI eigenvalue test.
    mismatches, tested = 0, 0
    for omega in DISK_OMEGAS:
        r = omega * arm.gamma * lam / 2
        disk = (z00 - r) ** 2 + z01 ** 2 - r ** 2
        for k in rng.choice(W.size, 3000, replace=False):
            if abs(disk.flat[k]) < 1e-9 * max(r * r, 1.0):
                continue                                   # on the curve itself, to rounding
            tested += 1
            u = velocity(W.flat[k], S.flat[k])
            mismatches += (disk.flat[k] <= 0) != (lmi_min_eig(arm, theta_b, u, omega) >= 0)
    for k in rng.choice(W.size, 3000, replace=False):         # the union over omega: Z00 > 0
        if abs(z00.flat[k]) < 1e-9:
            continue
        tested += 1
        u = velocity(W.flat[k], S.flat[k])
        mismatches += (z00.flat[k] > 0) != (lmi_min_eig(arm, theta_b, u, 1e8) >= 0)
    out["set_mismatches"], out["set_tested"] = mismatches, tested

    # 3. Proposition 2 on the boundary: the note's bound against bisection.
    errors = []
    while len(errors) < 300:
        w, s = rng.uniform(-span, span), rng.uniform(-span, span)
        z0, z1 = note_blocks(arm, diff, w, s)
        if z0 <= 1e-3:
            continue
        closed = float(scalar_bound_boundary(arm, z0, z1))
        errors.append(abs(bisect_scalar(arm, theta_b, velocity(w, s)) - closed) / closed)
    out["boundary_bound_error"] = max(errors)

    # 4. The rotation on the boundary: det = -Z01^2 for every omega; matrix multiplier 2|rate|/(sqrt 3 gamma).
    u_rot = velocity(rate, 0.0)
    out["rotation_blocks"] = note_blocks(arm, diff, rate, 0.0)
    matrix = MatrixMultiplier()
    out["rotation_matrix"] = matrix(arm, theta_b, u_rot)
    out["rotation_matrix_limit"] = 2 * abs(rate) / (math.sqrt(3) * arm.gamma)
    out["rotation_lmi"] = [(omega, lmi_min_eig(arm, theta_b, u_rot, omega)) for omega in (1.0, 1e2, 1e4, 1e6)]

    out["side"] = -1.0 if note_blocks(arm, diff, 0.0, 1.0)[0] < 0 else 1.0   # the side with Z00 > 0
    return out


# ------------------------------------------------------------ figure
def plot(arm: TwoLinkArm, boundary: str, span: float, grid: int, checks: dict, outdir: Path, name: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    figures.apply_style()
    S = figures.SCALE
    blue, red = figures.COLORS["matrix"], figures.COLORS["standard"]
    fs = figures.LABEL_FS
    diff = boundary_value(arm, boundary)
    W, Sv = np.meshgrid(np.linspace(-span, span, grid), np.linspace(-span, span, grid))
    z00, z01 = note_blocks(arm, diff, W, Sv)
    omega_star = scalar_bound_boundary(arm, z00, z01)

    fig, ax = plt.subplots(figsize=(6.9 * 0.85, 6.9 * 0.72), layout="constrained")
    shown = np.where(z00 > 0, np.maximum(omega_star, 1.0), np.nan)
    cmap = plt.get_cmap("Oranges").copy()
    cmap.set_bad("0.86")
    image = ax.pcolormesh(W, Sv, shown, norm=LogNorm(1, 1e3), cmap=cmap, shading="auto", rasterized=True)
    # Level sets omega* = omega are the scalar disks; omega = 1 is the standard MCBF.
    standard = [omega == 1.0 for omega in DISK_OMEGAS]
    lines = ax.contour(W, Sv, np.where(z00 > 0, omega_star, np.nan), levels=list(DISK_OMEGAS),
                       colors=[red if s else "0.15" for s in standard],
                       linewidths=[1.1 * S if s else 0.45 * S for s in standard])
    ax.clabel(lines, fmt=lambda v: "$\\omega$=1" if v == 1.0 else f"{v:g}", fontsize=figures.TICK_FS * 0.55)
    inside = (z00 > 0) & (omega_star <= 1.0)
    if inside.any():
        ax.text(W[inside].mean(), Sv[inside].mean(), "standard\nMCBF", ha="center", va="center",
                fontsize=figures.LEGEND_FS * 0.6, color=red)
    ax.axhline(0, color=blue, lw=1.4 * S, zorder=3)
    into = checks["side"] < 0                      # Z00 > 0 below the line: unsafe above it
    ax.text(0.5, 0.75 if into else 0.25, "unsafe", transform=ax.transAxes, ha="center", va="center",
            fontsize=figures.LABEL_FS * 0.9, color="0.3")
    ax.text(0.02, 0.52 if into else 0.48, "blue line: $\\omega^*=\\infty$, certified by matrix OD only "
            "(Prop. 1, Thm. 1)", transform=ax.transAxes, ha="left", va="bottom" if into else "top",
            fontsize=figures.LEGEND_FS * 0.55, color=blue)
    ax.set(xlim=(-span, span), ylim=(-span, span), aspect="equal")
    bar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02, extend="max")
    bar.set_label("$\\omega^*=(Z_{00}^2+Z_{01}^2)/(\\alpha(\\lambda)Z_{00})$, smallest scalar decay\n"
                  "that certifies the velocity (shown from $\\omega_d=1$)", fontsize=figures.LABEL_FS * 0.48)
    bar.ax.tick_params(labelsize=figures.TICK_FS * 0.55)
    ax.set_xlabel("rotation $(\\dot\\theta_1+\\dot\\theta_2)/2$ [rad/s]", fontsize=fs * 0.6)
    ax.set_ylabel("$\\dot\\theta_2-\\dot\\theta_1$ [rad/s]", fontsize=fs * 0.6)
    ax.tick_params(labelsize=figures.TICK_FS * 0.55)
    ax.set_title(f"Proposition 2 at the {boundary} boundary: $l_1={arm.l1:g}$, $l_2={arm.l2:g}$, "
                 f"$\\bar\\sigma={arm.sigma_bar:.4g}$, $\\gamma={arm.gamma:g}$, $\\theta_1=0$", fontsize=fs * 0.62)
    figures.save(fig, outdir, name)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="The figure is written to OUT/NAME.png.",
    )
    parser.add_argument("--l1", type=float, default=1.0, help="length of link 1 (default: 1)")
    parser.add_argument("--l2", type=float, default=1.0, help="length of link 2 (default: 1)")
    parser.add_argument(
        "--sigma-bar", type=float, default=None,
        help="lower bound on sigma_min(J), in (0, min(l1, l2)); by default the boundary of "
             "the safe set is at theta2 - theta1 = pi/3 and 2 pi/3")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="rate of the class-K term alpha(H) = gamma H (default: 1)")
    parser.add_argument("--boundary", choices=("inner", "outer"), default="inner",
                        help="which boundary of the safe set (default: inner)")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="rotation rate about joint 1 of the printed Proposition 1 check, rad/s "
                             "(default: 1)")
    parser.add_argument("--span", type=float, default=2.0,
                        help="half-width of the velocity window, rad/s (default: 2)")
    parser.add_argument("--grid", type=int, default=401,
                        help="grid points per axis of the figure (default: 401)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random checks (default: 0)")
    parser.add_argument("--out", type=Path, default=ROOT / "figures",
                        help="output directory (default: figures/ in the repository)")
    parser.add_argument("--name", default="fig_certified_velocities",
                        help="file name without extension (default: fig_certified_velocities)")
    args = parser.parse_args(argv)

    try:
        arm = TwoLinkArm(args.l1, args.l2, gamma=args.gamma, sigma_bar=args.sigma_bar)
    except ValueError as err:
        parser.error(str(err))
    if args.span <= 0 or args.grid < 11 or args.rate == 0:
        parser.error("--span must be positive, --grid at least 11 and --rate nonzero")
    warnings.filterwarnings("ignore", message="Solution may be inaccurate")

    checks = run_checks(arm, args.boundary, args.span, args.rate, np.random.default_rng(args.seed))
    lam, theta_b = checks["lambda"], checks["theta_b"]
    z0, z1 = checks["rotation_blocks"]
    print(arm)
    print(f"{args.boundary} boundary: theta = ({theta_b[0]:.4f}, {theta_b[1]:.6f}), "
          f"lambda_min(H) = {arm.margin(theta_b):+.1e}, lambda = l1^2 + l2^2 - 2 sigma_bar^2 = {lam:.6f}")
    print("\nThe note's closed forms against numerics:")
    rows = [
        ("Z00, Z01 (Section 5) vs the numerical eigenbasis (201 x 201 velocities)",
         f"max abs error {checks['block_error']:.1e}"),
        ("Z00, Z01 (Section 5) vs the Section 4 forms on the boundary",
         f"max abs error {checks['section4_error']:.1e}"),
        ("disks omega* <= omega, Z00 > 0 vs lambda_min(Hdot + omega alpha(H)) >= 0",
         f"{checks['set_mismatches']} mismatches in {checks['set_tested']} tests"),
        ("omega* = (Z00^2 + Z01^2)/(alpha(lambda) Z00) vs bisection (300 velocities)",
         f"max rel error {checks['boundary_bound_error']:.1e}"),
    ]
    for label, value in rows:
        print(f"  {label:78s} {value}")
    print(f"\nRotation ({args.rate:g}, {args.rate:g}) on the boundary: Z00 = {z0:+.1e}, Z01 = {z1:+.6f} "
          f"(-lambda rate = {-lam * args.rate:+.6f})")
    eigenvalues = ", ".join(f"{v:+.2e} (omega {w:g})" for w, v in checks["rotation_lmi"])
    print(f"  lambda_min(Hdot + omega alpha(H)) = {eigenvalues}: negative for every omega, tending to 0 "
          "(Proposition 1)")
    print(f"  matrix OD: smallest ||Omega - Omega_d||_F = {checks['rotation_matrix']:.6f} "
          f"(closed form 2|rate|/(sqrt 3 gamma) = {checks['rotation_matrix_limit']:.6f})")

    plot(arm, args.boundary, args.span, args.grid, checks, args.out, args.name)
    print(f"\nwrote {args.out / (args.name + '.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
