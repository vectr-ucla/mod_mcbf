"""The paper's Fig. 2 and the numbers of Sec. VIII-C, from the six tracking cases of ``odmcbf/paper.py``.

The script reruns the hero configuration of ``scripts/closed_loop_demo.py`` on the arms
``l2 = 0.75, 1, 1.25``, with the target pushed into the outer band and into the inner band. In
the hero configuration ``l1 = 1`` and ``sigma_bar = 0.6``, and the target turns at 1 rad/s, moves
30% of the way into a band over ``1 <= t <= 1.75`` s, stays there until 3.25 s and is back by
4 s. These are the six cases ``odmcbf.paper.TRACKING_CASES``, with the same plant, nominal
controller and filters as the hero case. ``scripts/reproduce_paper.py`` runs this script with the
other figures.

It writes the paper's figure as two subfigures, ``OUT/fig_tracking_a`` (the hero arm's outer
crossing) and ``OUT/fig_tracking_b`` (its inner crossing), each as ``.pdf`` and ``.png``. Each
plots the distance from every filter's end effector to the target, in the hero figure's colours
and line widths (solid lines, scalar OD drawn over matrix OD). The least possible distance, from
the target to the safe set, is dotted, and the window where the target is outside the safe set
is shaded. ``OUT/fig_tracking_legend`` is the legend the two share, at the column width. Each
file is drawn at its printed size so LaTeX can include it at 100%; the subcaptions are in the paper.

For the paper's closed-loop table, it prints each case's held-window means (``held`` as in
``closed_loop_demo.summarize``): the distance beyond the least possible, the margin, scalar OD's
omega, the median smallest scalar decay certifying matrix OD's applied input, and each OD
filter's ``||Omega - Omega_d||_F``. For the solver table, it prints per filter, over all six runs,
Clarabel's solve time and iterations and the wall time of a full filter call.

    python scripts/tracking_figure.py                 # figures/fig_tracking_{a,b,legend}.{pdf,png} and the numbers
    python scripts/tracking_figure.py --out /tmp/x    # elsewhere
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import closed_loop_demo as cld                                   # noqa: E402
import odmcbf.filters as filters                                 # noqa: E402
from odmcbf import figures, paper                                # noqa: E402

#: The six cases: link 2 and the band the target enters. Link 1, the threshold, the target's timing
#: and depth, the gain, the step and the penalty are the hero case's (``odmcbf.paper``).
CASES = list(paper.TRACKING_CASES)
THETA0 = paper.THETA0
GAIN, RATE, DT, RHO_OMEGA = paper.GAIN, paper.TIMELINE["rate"], paper.DT, paper.FILTER["rho_omega"]

#: The figure shows this arm's outer and inner crossings, one subfigure each, drawn at
#: SUBFIGURE_SIZE, inches: 0.49 of the ieeeconf column width (245.72 TeX points) by this height.
FIGURE_L2 = paper.HERO_ARM["l2"]
SUBFIGURE_SIZE = (0.49 * 245.71811 / 72.27, 1.32)
#: Margins of the plot within a subfigure, points: left, right, bottom, top.
SUBFIGURE_MARGINS = (23.0, 3.0, 17.0, 3.0)
#: The shared legend, drawn at the column width by this height, inches.
LEGEND_SIZE = (245.71811 / 72.27, 0.30)
#: The hero figure's line widths of the paths, points, and its drawing order (scalar OD on top).
PATH_WIDTHS = {"standard": 0.6, "scalar": 0.5, "matrix": 0.9}

# ------------------------------------------------------------- solver statistics
#: One record per filter solve: wall time of the call, Clarabel's solve time and iterations, status.
STATS: list[dict] = []
_original_call = filters.SafetyFilter.__call__


def _timed_call(self, *args, **kwargs):
    tic = time.perf_counter()
    res = _original_call(self, *args, **kwargs)
    call_ms = 1e3 * (time.perf_counter() - tic)
    st = self._problem.solver_stats
    STATS.append(dict(decay=self.decay, call_ms=call_ms, solve_ms=1e3 * float(st.solve_time),
                      iters=int(st.num_iters) if st.num_iters is not None else -1,
                      status=res.status, residual=res.safety_residual))
    return res


filters.SafetyFilter.__call__ = _timed_call


# ------------------------------------------------------------------------ runs
def run_cases() -> dict:
    """Simulate the six cases; each value holds the runs, their summaries, the held window and the
    solver records of that case."""
    out = {}
    for l2, band in CASES:
        arm = paper.arm(l2)
        target = paper.target(arm, band)
        first = len(STATS)
        runs = cld.simulate(arm, THETA0, target, GAIN, paper.TIMELINE["duration"], DT, RHO_OMEGA)
        r_lo, r_hi = arm.safe_radius_interval
        out[(l2, band)] = dict(
            runs=runs, summaries={k: cld.summarize(r, arm, target) for k, r in runs.items()},
            held=cld.held_window(target), d_min=max(0.0, r_lo - target.r1, target.r1 - r_hi),
            stats=STATS[first:])
        print(f"ran l2 = {l2:g}, {band}: r1 = {target.r1:.4f}, least possible distance "
              f"{out[(l2, band)]['d_min']:.4f}, held t in {out[(l2, band)]['held']}", flush=True)
    return out


def print_tables(data: dict) -> None:
    filt = cld.FILTERS
    print("\nClosed-loop table, means over the held window (Delta d and lambda_min(H) in 1e-3):")
    head = "".join(f"{f'l2={l2:g} {band}':>14s}" for l2, band in CASES)
    print(f"{'':28s}{head}")

    def row(label, values, fmt):
        print(f"{label:28s}" + "".join(f"{fmt.format(v):>14s}" for v in values))

    for key in filt:
        row(f"Delta d, {key}", [1e3 * data[c]["summaries"][key]["held_excess"] for c in CASES], "{:.1f}")
    for key in filt:
        row(f"lambda_min(H), {key}", [1e3 * data[c]["summaries"][key]["held_margin"] for c in CASES], "{:.1f}")
    omegas = []
    for c in CASES:
        r = data[c]["runs"]["scalar"]
        start, back = data[c]["held"]
        held = (r.t >= start - r.dt / 2) & (r.t < back - r.dt / 2)
        omegas.append(float((2.0 * r.Omega[held[:-1], 0, 0]).mean()))            # Omega = (omega/2) I
    row("omega, scalar (mean)", omegas, "{:.2f}")
    row("smallest omega for matrix's u", [data[c]["summaries"]["matrix"]["held_scalar_decay_needed"]
                                          for c in CASES], "{:.1f}")
    for key in ("scalar", "matrix"):
        row(f"||Omega - Omega_d||_F, {key}", [data[c]["summaries"][key]["held_decay_deviation"]
                                              for c in CASES], "{:.2f}")
    for key in filt:
        row(f"rotation [rad], {key}", [data[c]["summaries"][key]["rotation"] for c in CASES], "{:.2f}")
    print("Smallest margin over each run:")
    for c in CASES:
        s = data[c]["summaries"]
        print(f"  l2 = {c[0]:g}, {c[1]}: " + ", ".join(f"{k} {s[k]['min_margin']:+.2e}" for k in cld.KEYS))

    print("\nSolver table, over all six runs:")
    for decay in ("standard", "scalar", "matrix"):
        rows = [st for d in data.values() for st in d["stats"] if st["decay"] == decay]
        solve = np.array([st["solve_ms"] for st in rows])
        iters = np.array([st["iters"] for st in rows])
        call = np.array([st["call_ms"] for st in rows])
        inaccurate = [st for st in rows if st["status"] != "optimal"]
        print(f"  {decay:>8s}: {len(rows)} solves; solve {solve.mean():.3f} mean, {solve.max():.3f} max ms; "
              f"iterations {iters.mean():.1f} mean, {iters.max()} max; call {call.mean():.2f} mean ms; "
              f"{len(inaccurate)} optimal_inaccurate (least residual "
              f"{min((st['residual'] for st in inaccurate), default=math.nan):+.1e}); "
              f"least residual overall {min(st['residual'] for st in rows):+.1e}")


# ---------------------------------------------------------------------- figure
def plot(data: dict, outdir: Path) -> list[Path]:
    """The outer and the inner crossing of the arm ``FIGURE_L2`` as subfigures ``fig_tracking_a`` and
    ``fig_tracking_b``, in the hero figure's colours and line widths; returns the PDF paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    width, height = SUBFIGURE_SIZE
    across, tall = 72.0 * width, 72.0 * height                     # points
    left, right, bottom, top = SUBFIGURE_MARGINS
    with plt.rc_context(cld.HERO_RC), cld._script_scale(cld.HERO_SCRIPT):
        for suffix, band in (("a", "outer"), ("b", "inner")):
            fig = plt.figure(figsize=SUBFIGURE_SIZE)
            ax = fig.add_axes([left / across, bottom / tall, (across - left - right) / across,
                               (tall - bottom - top) / tall])
            runs, s = data[(FIGURE_L2, band)]["runs"], data[(FIGURE_L2, band)]["summaries"]
            first = runs["nominal"]
            t = first.t
            for a, b in cld.contact_intervals(first):
                ax.axvspan(a, b, color=cld.HERO_CONTACT, lw=0, zorder=0)
            for key in cld.FILTERS:
                ax.plot(t, runs[key].error, color=cld.HERO_COLORS[key], lw=PATH_WIDTHS[key],
                        solid_capstyle="round", zorder=cld.HERO_Z[key])
            ax.plot(t, first.bound, zorder=6, **cld.HERO_BOUND)
            peak = 1.22 * max(s["scalar"]["held_error"], first.bound.max())   # scalar OD's plateau at 82% of the height
            ax.set_ylim(0, peak)
            ax.set_yticks(np.arange(0, peak, _round_step(peak / 3.2)))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.set_xlim(0, t[-1])
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.tick_params(length=1.5, pad=1.0)
            ax.grid(alpha=0.35, lw=0.35)
            ax.set_xlabel(r"$t$ [s]")
            if band == "outer":
                ax.set_ylabel("Distance to the target")
            for ext in (".pdf", ".png"):
                fig.savefig(outdir / f"fig_tracking_{suffix}{ext}", dpi=600 if ext == ".png" else None)
            plt.close(fig)
            written.append(outdir / f"fig_tracking_{suffix}.pdf")

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        fig = plt.figure(figsize=LEGEND_SIZE)                   # the legend the two subfigures share
        handles = [Line2D([], [], color=cld.HERO_COLORS[k], lw=PATH_WIDTHS[k]) for k in cld.FILTERS]
        handles += [Line2D([], [], **cld.HERO_BOUND), Patch(facecolor=cld.HERO_CONTACT, edgecolor="none")]
        labels = [figures.TITLES[k] for k in cld.FILTERS] + ["Least possible", r"Target outside $\mathcal{C}$"]
        fig.legend(handles, labels, loc="center", ncol=3, handlelength=1.8, handletextpad=0.5,
                   columnspacing=1.2, labelspacing=0.3, borderaxespad=0.0, frameon=False)
        for ext in (".pdf", ".png"):
            fig.savefig(outdir / f"fig_tracking_legend{ext}", dpi=600 if ext == ".png" else None)
        plt.close(fig)
        written.append(outdir / "fig_tracking_legend.pdf")
    return written


def _round_step(x: float) -> float:
    """The smallest of 1, 2, 2.5, 5 times a power of ten that is at least ``x``."""
    e = 10 ** math.floor(math.log10(x))
    return next(m * e for m in (1, 2, 2.5, 5, 10) if m * e >= x)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "figures",
                        help="output directory (default: figures/ in the repository)")
    args = parser.parse_args(argv)
    data = run_cases()
    print_tables(data)
    for pdf in plot(data, args.out):
        print(f"wrote {pdf} and {pdf.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
