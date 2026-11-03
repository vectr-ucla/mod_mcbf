"""Draw the two-link arm in its workspace, with the region where H is violated.

The arm is ``TwoLinkArm(l1, l2, sigma_bar=...)`` from ``odmcbf/two_link.py``
and ``H = J J^T - sigma_bar^2 I`` is the barrier of
``two_link_manipulator_kinematics.md``. Each configuration gets one panel with
the workspace ``|l1 - l2| <= |p| <= l1 + l2`` and everything outside it in
grey, the part of it where ``lambda_min(H) < 0`` (hatched), the boundary
``lambda_min(H) = 0`` (dashed), and the arm.

Without ``--theta`` the demo places four configurations relative to the arm's
safe set: ``theta2 - theta1 = pi/2``, where the margin is largest, one on its
boundary, and one in each band where ``H`` is violated, stretched and folded.

Examples, from the repository root (the script also runs from any other directory
when given its full path):

    python scripts/workspace_demo.py
    python scripts/workspace_demo.py --l1 1 --l2 0.6 --sigma-bar 0.3
    python scripts/workspace_demo.py --l1 1.5 --l2 0.8 --sigma-bar 0.78 --theta 0 1.2 --theta 0.4 2.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))               # import odmcbf from the repository root

from odmcbf import TwoLinkArm, figures      # noqa: E402
from odmcbf.figures import margin_status    # noqa: E402


def demo_configurations(arm: TwoLinkArm) -> np.ndarray:
    """Largest margin, on the boundary, stretched past it, and folded past it."""
    lo, hi = arm.safe_theta_diff_interval
    return np.array([
        [0.0, np.pi / 2],                   # theta2 - theta1 = pi/2: the largest margin
        [0.5, 0.5 + lo],                    # on the boundary
        [0.9, 0.9 + lo / 2],                # stretched past the boundary: H is violated
        [1.0, 1.0 + (np.pi + hi) / 2],      # folded past the boundary: H is violated
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="The figure is written to OUT/NAME.png.",
    )
    parser.add_argument("--l1", type=float, default=1.0, help="length of link 1 (default: 1)")
    parser.add_argument("--l2", type=float, default=1.0, help="length of link 2 (default: 1)")
    parser.add_argument(
        "--sigma-bar", type=float, default=None,
        help="lower bound on sigma_min(J), in (0, min(l1, l2)); by default the "
             "boundary of the safe set is at theta2 - theta1 = pi/3")
    parser.add_argument(
        "--theta", type=float, nargs=2, action="append", metavar=("THETA1", "THETA2"),
        help="absolute joint angles in radians; repeat for one panel each "
             "(default: four configurations placed relative to the safe set)")
    parser.add_argument("--out", type=Path, default=ROOT / "figures",
                        help="output directory (default: figures/ in the repository)")
    parser.add_argument("--name", default="fig_workspace",
                        help="file name without extension (default: fig_workspace)")
    args = parser.parse_args(argv)

    try:
        arm = TwoLinkArm(args.l1, args.l2, sigma_bar=args.sigma_bar)
    except ValueError as err:
        parser.error(str(err))
    thetas = np.array(args.theta) if args.theta else demo_configurations(arm)
    if not np.all(np.isfinite(thetas)):
        parser.error("--theta values must be finite")

    r_in, r_out = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    print(arm)
    print(f"workspace   {r_in:.4g} <= |p| <= {r_out:.4g}")
    print(f"safe set    {r_lo:.4g} <= |p| <= {r_hi:.4g}; H is violated elsewhere in the workspace")
    for theta in thetas:
        radius = float(np.linalg.norm(arm.end_effector(theta)))
        margin, status = margin_status(arm, theta)
        print(f"theta = ({theta[0]:+.3f}, {theta[1]:+.3f})   |p| = {radius:.4g}   "
              f"lambda_min(H) = {margin:+.4g}   {status}")

    figures.workspace(arm, thetas, outdir=args.out, name=args.name)
    print(f"wrote {args.out / (args.name + '.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
