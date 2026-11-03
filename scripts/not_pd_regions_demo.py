"""Plot where H = J J^T - sigma_bar^2 I is not positive definite, for one sigma_bar.

``H`` is the barrier of ``two_link_manipulator_kinematics.md``, in the absolute
joint angles ``theta = (theta1, theta2)`` of ``odmcbf/two_link.py``. The left
panel hatches the configurations with ``lambda_min(H) <= 0`` in the
``(theta1, theta2)`` plane, dashes the lines with ``lambda_min(H) = 0`` and
dots the singular configurations ``theta2 - theta1 = 0`` and ``pi``. The right
panel plots both eigenvalues of ``H`` against ``theta2 - theta1``.

Any ``sigma_bar >= 0`` works: ``0`` leaves only the singular lines, and
``sigma_bar >= min(l1, l2)`` covers every configuration. ``--closed-loop``
overlays the notebook's four closed loops, which needs
``0 < sigma_bar < min(l1, l2)`` so that the filters have a safe set to keep.

Examples, from the repository root (the script also runs from any other directory
when given its full path):

    python scripts/not_pd_regions_demo.py
    python scripts/not_pd_regions_demo.py --l1 1.3 --l2 0.8 --sigma-bar 0.5
    python scripts/not_pd_regions_demo.py --sigma-bar 0
    python scripts/not_pd_regions_demo.py --closed-loop
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))               # import odmcbf from the repository root

from odmcbf import TwoLinkArm, TwoLinkKinematics, run_comparison   # noqa: E402
from odmcbf import regions                                          # noqa: E402

#: The notebook's closed-loop setup.
U_NOM = np.array([1.0, 0.4])
THETA0 = np.array([0.0, np.pi / 2])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="The figure is written to OUT/NAME.png.",
    )
    parser.add_argument("--l1", type=float, default=1.0, help="length of link 1 (default: 1)")
    parser.add_argument("--l2", type=float, default=1.0, help="length of link 2 (default: 1)")
    parser.add_argument(
        "--sigma-bar", type=float, default=None,
        help="threshold sigma_bar >= 0 (default: the TwoLinkArm default, which puts "
             "the boundary of the safe set at theta2 - theta1 = pi/3)")
    parser.add_argument(
        "--closed-loop", action="store_true",
        help="overlay the notebook's nominal, standard, scalar OD and matrix OD closed loops")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="time step of the closed loops (default: 0.01)")
    parser.add_argument("--out", type=Path, default=ROOT / "figures",
                        help="output directory (default: figures/ in the repository)")
    parser.add_argument("--name", default="fig_not_pd_regions",
                        help="file name without extension (default: fig_not_pd_regions)")
    args = parser.parse_args(argv)

    try:
        kinematics = TwoLinkKinematics(args.l1, args.l2)
        sigma_bar = (TwoLinkArm(args.l1, args.l2).sigma_bar
                     if args.sigma_bar is None else args.sigma_bar)
        region = regions.NotPDRegion(kinematics, sigma_bar)
        arm = TwoLinkArm(args.l1, args.l2, sigma_bar=sigma_bar) if args.closed_loop else None
        if args.closed_loop and not (np.isfinite(args.dt) and args.dt > 0):
            raise ValueError(f"--dt must be positive; got {args.dt:g}")
    except ValueError as err:
        parser.error(str(err))

    print(region)
    if region.everywhere:
        print("H is not positive definite anywhere: sigma_bar >= min(l1, l2)")
    elif region.singular_only:
        print("H is not positive definite only where J is singular: "
              "theta2 - theta1 = 0 or pi (mod 2 pi)")
    else:
        w = region.half_width
        print(f"H is not positive definite for |theta2 - theta1| <= {w:.6g} "
              f"or >= pi - {w:.6g} (mod 2 pi)")
    if region.boundary_theta_diffs:
        values = ", ".join(f"{v:+.6g}" for v in region.boundary_theta_diffs)
        print(f"lambda_min(H) = 0 at theta2 - theta1 = {values}")

    paths = None
    if arm is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate")
                paths = run_comparison(arm, u_nom=U_NOM, theta0=THETA0, dt=args.dt, horizon=8.0,
                                       omega_d=1.0, rho_omega=5e-3, recovery=True)
        except ValueError as err:
            parser.error(str(err))
        for trace in paths.values():
            inside = region.contains(trace.theta)
            print(f"{trace.label:<10} min lambda_min(H) = {trace.margin.min():+.3e}   "
                  f"steps with lambda_min(H) <= 0: {int(inside.sum())} of {len(inside)}")

    regions.plot_not_pd_regions(kinematics, sigma_bar, paths=paths,
                                outdir=args.out, name=args.name)
    print(f"wrote {args.out / (args.name + '.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
