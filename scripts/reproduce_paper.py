"""Reproduce every figure and number of the paper's Section VIII, Singularity Avoidance.

Runs the three experiments of ``odmcbf/paper.py`` through the scripts that draw them and
saves the figure files under the names the paper uses:

    paper                                      script                         file under OUT
    Fig. 1(a), (b), and its numbers            scripts/closed_loop_demo.py    fig_hero_a.pdf, fig_hero_b.pdf
    Fig. 2(a), (b) with its legend, and the    scripts/tracking_figure.py     fig_tracking_a.pdf,
      numbers of Sec. VIII-C: the medians                                     fig_tracking_b.pdf,
      of the smallest certifying scalar decay,                                fig_tracking_legend.pdf
      scalar OD's mean omega, the smallest
      margins, and the solve times
    Fig. 3 and the recovery paragraph of       scripts/recovery_demo.py       fig_recovery.pdf
      Sec. VIII-C

Each script prints its own tables, and the lines marked ``[paper]`` at the end give the numbers
the paper quotes. ``odmcbf/tests/test_paper.py`` checks the same numbers.

    python scripts/reproduce_paper.py                                # figures/paper/
    python scripts/reproduce_paper.py --out /tmp/paper
    python scripts/reproduce_paper.py --copy-to path/to/paper/Figures # also copy into the paper's figure directory

The solve times depend on the machine; everything else is deterministic. The script first prints
the versions of the running environment, to compare with the ones the paper states
(``requirements-lock.txt`` pins them).
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                   # import odmcbf from the repository root
sys.path.insert(0, str(ROOT / "scripts"))       # and the figure scripts

import closed_loop_demo  # noqa: E402  Fig. 1
import recovery_demo  # noqa: E402  Fig. 3
import tracking_figure  # noqa: E402  Fig. 2 and the numbers of Sec. VIII-C

#: The paper's file names: (script, name the script writes, name the paper includes).
FIGURES = (
    ("closed_loop_demo", "fig_hero_a.pdf", "fig_hero_a.pdf"),
    ("closed_loop_demo", "fig_hero_b.pdf", "fig_hero_b.pdf"),
    ("tracking_figure", "fig_tracking_a.pdf", "fig_tracking_a.pdf"),
    ("tracking_figure", "fig_tracking_b.pdf", "fig_tracking_b.pdf"),
    ("tracking_figure", "fig_tracking_legend.pdf", "fig_tracking_legend.pdf"),
    ("recovery_demo", "fig_recovery_outer_column.pdf", "fig_recovery.pdf"),
)


def versions() -> str:
    import cvxpy
    import matplotlib
    import numpy

    solver = "CLARABEL"
    try:
        import clarabel
        solver = f"Clarabel {clarabel.__version__}"
    except ImportError:                          # pragma: no cover - clarabel ships with cvxpy
        pass
    return (f"Python {platform.python_version()}, numpy {numpy.__version__}, cvxpy {cvxpy.__version__}, "
            f"{solver}, matplotlib {matplotlib.__version__}")


def banner(text: str) -> None:
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "paper",
                        help="directory for the paper's figure files (default: figures/paper/)")
    parser.add_argument("--copy-to", type=Path, default=None, metavar="DIR",
                        help="also copy the paper's figure files into this directory, such as the paper's Figures/")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"environment: {versions()}")
    print("the paper states the solver and cvxpy versions in Sec. VIII-A; solve times depend on the machine")

    banner("Fig. 1: the hero run (scripts/closed_loop_demo.py, its default case)")
    closed_loop_demo.main(["--out", str(out)])
    banner("Fig. 2 and the numbers of Sec. VIII-C: the six tracking cases (scripts/tracking_figure.py)")
    tracking_figure.main(["--out", str(out)])
    banner("Fig. 3 and the recovery paragraph of Sec. VIII-C (scripts/recovery_demo.py)")
    recovery_demo.main(["--out", str(out)])

    banner("The paper's figure files")
    for script, written, name in FIGURES:
        if written != name:
            shutil.copyfile(out / written, out / name)
        print(f"  {name:26s} <- scripts/{script}.py")
        if args.copy_to is not None:
            args.copy_to.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / name, args.copy_to / name)
    if args.copy_to is not None:
        print(f"  copied into {args.copy_to}")
    print("\nWhere the numbers quoted in Sec. VIII-C are printed above:")
    print("  [paper] 'smallest scalar decay ... a median of 24 to 36': the row 'smallest omega for matrix's u'")
    print("  [paper] 'penalty distance ... of 17 to 25': (omega - 1) / sqrt 2 of that row")
    print("  [paper] 'scalar OD settles for a mean omega of only 6.3 to 8.3': the row 'omega, scalar (mean)'")
    print("  [paper] 'keep lambda_min(H) > 0 in every run': 'Smallest margin over each run'")
    print("  [paper] 'the standard MCBF stays farthest from the target, followed by scalar OD': the rows 'Delta d'")
    print("  [paper] 'Clarabel takes a mean of ... ms per solve': the 'Solver table', column 'solve ... mean'")
    print("  [paper] recovery: lambda_min(H) at t = 0, and the recovery rates against 2 beta, in the recovery table")
    print("  [paper] Fig. 1's comments: the hero run's 'Closed loop' table and rotation cap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
