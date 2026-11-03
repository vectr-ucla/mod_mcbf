"""Recovery from outside the safe set, at the rate the recovery constraint sets.

The arm starts at rest in a band where ``H = J J^T - sigma_bar^2 I`` is violated,
``--depth`` of the band's width beyond the boundary of the safe set, and its
nominal command tracks a target that stays at the start's end effector,

    u_nom(t, theta) = J(theta)^-1 k (p* - p(theta)),    p* = p(theta0),

the controller of ``odmcbf/tracking.py`` with a target that neither turns nor moves
(``odmcbf.simulate.hold_at_start``). The command is zero at the start and, once the
filter has moved the arm, points back toward the start, so it never raises the
margin: unfiltered, the arm stays in the band. Whatever brings it back to the safe
set is the filter's decay term alone.

The filter is matrix OD with the recovery constraint of the paper's Section VII. With ``alpha(H) = gamma H``,
where ``lambda_min(H) < 0`` its matrix inequality along the unit eigenvector ``v0``
of ``lambda_min(H)`` reads

    d/dt lambda_min(H) = v0' Hdot v0 >= 2 gamma |lambda_min(H)| v0' sym(Omega) v0.

The recovery constraint, eq. (25), ``sym(Omega) >= beta I``, bounds the decay along
``v0`` from below, so the margin rises at least at the relative rate ``2 beta gamma``.
With the nominal command pulling back, the filter takes exactly that rate: for
``beta >= omega_d/2`` the penalty and the matrix inequality both push
``v0' sym(Omega) v0`` down onto ``beta``, and per step ``lambda_min(H)`` shrinks by the
factor ``1 - 2 beta gamma dt`` (to order ``dt^2``). Without the recovery constraint the
filter drops the decay along ``v0`` instead, and the arm comes to rest in the band.

The script runs ``odmcbf.simulate.run_recovery``: matrix OD with the recovery
constraint at each of ``--betas``, and once without it. The default betas are
``omega_d/2 = 0.5``, the paper's design and the largest value for which
Proposition 3(b) keeps the filter no more invasive than the standard MCBF, and
``0.25``. It prints a table and writes ``NAME.png``, which shows the arm in its
workspace at the start and end of each run, and ``lambda_min(H)`` against time over
the slowest recovery each ``beta`` allows, ``lambda_min(H(0)) exp(-2 beta gamma t)``.
It also writes ``NAME_column.pdf`` (and ``.png``), the second panel alone at the
paper's column width, which is the paper's Fig. 3. The defaults are the experiment
``odmcbf.paper.RECOVERY``. ``scripts/reproduce_paper.py`` runs it with the other
figures and saves it as ``fig_recovery.pdf``, the name the paper uses.

Examples, from the repository root:

    python scripts/recovery_demo.py                       # the hero arm, outer band
    python scripts/recovery_demo.py --band inner
    python scripts/recovery_demo.py --gamma 2 --gain 20   # the rate follows gamma, not the gain
    python scripts/recovery_demo.py --betas 0.5 1         # a beta above omega_d/2 recovers faster still
    python scripts/recovery_demo.py --l1 1 --l2 1 --sigma-bar 0.7071
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))               # import odmcbf from the repository root

from odmcbf import (BANDS, NO_RECOVERY, TwoLinkArm, band_start, figures, paper,  # noqa: E402
                    recovery_rate, run_recovery)

import closed_loop_demo as loops            # noqa: E402  the hero figure's bands, colours and line styles

OMEGA_D = paper.FILTER["omega_d"]
#: The arm of the hero configuration, and the paper's recovery experiment (``odmcbf.paper``).
ARM = paper.HERO_ARM
RECOVERY = paper.RECOVERY
#: Times, in seconds, at which the table reports the margin.
REPORT_TIMES = (1.0, 2.0)


def beta_of(key: str) -> float:
    """The recovery margin of a run of ``run_recovery``; NaN for the run without the constraint."""
    return math.nan if key == NO_RECOVERY else float(key.split("=", 1)[1])


def decay_along_v0(arm: TwoLinkArm, theta: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """``v0' sym(Omega) v0`` at each step, ``v0`` the unit eigenvector of ``lambda_min(H)``."""
    values = np.empty(len(Omega))
    for k, (theta_k, Omega_k) in enumerate(zip(theta, Omega)):
        v0 = np.linalg.eigh(arm.H(theta_k))[1][:, 0]
        values[k] = v0 @ (0.5 * (Omega_k + Omega_k.T)) @ v0
    return values


def simulate(arm: TwoLinkArm, band: str, depth: float, gain: float, duration: float, dt: float,
             rho_omega: float, betas: List[float]):
    """``run_recovery``; returns the traces, their guaranteed rates ``2 beta gamma`` (NaN without the
    recovery constraint), their legend labels, and the count of solves reported ``optimal_inaccurate``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = run_recovery(arm, band, depth, gain, dt, duration, betas=betas, omega_d=OMEGA_D,
                            rho_omega=rho_omega)
    inaccurate = sum("inaccurate" in str(w.message) for w in caught)
    rates = {key: 2 * beta_of(key) * arm.gamma for key in runs}
    labels = {key: (r"Matrix OD, no recovery constraint" if key == NO_RECOVERY
                    else rf"Matrix OD, $\beta={beta_of(key):g}$") for key in runs}
    return runs, rates, labels, inaccurate


def summarize(arm: TwoLinkArm, trace, rate: float) -> Dict[str, float]:
    """Numbers for the table."""
    dt = float(trace.t[1] - trace.t[0])
    relative = recovery_rate(trace)
    ratio = trace.margin[1:] / trace.margin[:-1]
    decay = decay_along_v0(arm, trace.theta[:-1], trace.Omega)
    nominal_rate = max(arm.margin_rate(theta, u) for theta, u in zip(trace.theta[:-1], trace.u_nom))

    def at(t):
        return float(trace.margin[min(int(round(t / dt)), len(trace.t) - 1)])

    def finite(values, how):
        values = values[np.isfinite(values)]
        return float(how(values)) if values.size else math.nan

    out = {"margin_0": float(trace.margin[0])}
    out.update({f"margin_{t:g}": at(t) for t in REPORT_TIMES})
    out.update({
        "margin_end": float(trace.margin[-1]),
        "shrunk": float(trace.margin[-1] / trace.margin[0]),
        "predicted": (1 - rate * dt) ** len(trace.u) if np.isfinite(rate) else math.nan,
        "guaranteed": rate,
        "rate_min": finite(relative, np.min),
        "rate_max": finite(relative, np.max),
        "rate_end": float(relative[-1]),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "decay_start": float(decay[0]),
        "decay_end": float(decay[-1]),
        "nominal_rate_max": nominal_rate,
        "deviation_max": float(np.linalg.norm(trace.u - trace.u_nom, axis=1).max()),
        "u_end": float(np.linalg.norm(trace.u[-1])),
        "min_residual": finite(trace.safety_residual, np.min),
        "min_recovery_residual": finite(trace.recovery_residual, np.min),
    })
    return out


ROWS = (
    ("lambda_min(H) at t = 0", "margin_0", "{:+.4e}"),
    *((f"lambda_min(H) at t = {t:g} s", f"margin_{t:g}", "{:+.4e}") for t in REPORT_TIMES),
    ("lambda_min(H) at the end", "margin_end", "{:+.4e}"),
    ("  relative to t = 0", "shrunk", "{:.5f}"),
    ("  (1 - 2 beta gamma dt)^N", "predicted", "{:.5f}"),
    ("guaranteed recovery rate 2 beta gamma", "guaranteed", "{:.4f}"),
    ("recovery rate, min over the run", "rate_min", "{:.4f}"),
    ("recovery rate, max over the run", "rate_max", "{:.4f}"),
    ("recovery rate at the end", "rate_end", "{:.2e}"),
    ("lambda_min(H) step ratio, min", "ratio_min", "{:.6f}"),
    ("lambda_min(H) step ratio, max", "ratio_max", "{:.6f}"),
    ("v0' sym(Omega) v0 at the start", "decay_start", "{:.4f}"),
    ("v0' sym(Omega) v0 at the end", "decay_end", "{:.2e}"),
    ("max d/dt lambda_min(H) under u_nom", "nominal_rate_max", "{:+.1e}"),
    ("max |u - u_nom|", "deviation_max", "{:.4f}"),
    ("|u| at the end", "u_end", "{:.1e}"),
    ("min LMI residual", "min_residual", "{:+.1e}"),
    ("min lambda_min(sym Omega) - beta", "min_recovery_residual", "{:+.1e}"),
)


def print_table(summaries: Dict[str, Dict[str, float]]) -> None:
    keys = list(summaries)
    heads = ["no recovery" if k == NO_RECOVERY else f"beta = {beta_of(k):g}" for k in keys]
    print(f"{'':40s}" + "".join(f"{h:>15s}" for h in heads))
    for label, field, fmt in ROWS:
        cells = []
        for k in keys:
            value = summaries[k][field]
            text = "-" if math.isnan(value) else fmt.format(value)
            if text.startswith("-") and text != "-" and float(text) == 0.0:
                text = text[1:]                              # no "-0.0000"
            cells.append(text.rjust(15))
        print(f"{label:40s}" + "".join(cells))


# ---------------------------------------------------------------- drawing
#: Every simulated trajectory is a solid line, one hue per run: the first beta (the paper's design,
#: omega_d/2) in matrix OD's blue from the hero figure, then green, brown and orange; the run without the
#: recovery constraint in red. Dashes are reserved for the theoretical bounds.
RUN_COLORS = (loops.HERO_COLORS["matrix"], "#009e73", "#8c564b", "#e69f00")
NO_RECOVERY_COLOR = "#d62728"
#: Dash patterns of the bound lines in the overview figure, one per beta.
BOUND_LS = ((0, (4, 2)), (0, (1.5, 1.5)), (0, (4, 1.5, 1, 1.5)), (0, (6, 2)))
#: The paper's figure is panel (b) alone at its printed size: the ieeeconf column width (245.72 TeX
#: points) by this height in inches. Also the plot margins in points (left, right, bottom, top), the
#: line width of the runs in points, and the style of the bound lines, which are drawn thin and black
#: over the runs they coincide with.
COLUMN_SIZE = (245.71811 / 72.27, 1.45)
COLUMN_MARGINS = (31.0, 3.0, 17.0, 3.0)
COLUMN_LW, COLUMN_BOUND = 1.6, dict(color="black", lw=0.8, ls=(0, (2.5, 1.5)))
FRACTIONS = {0.5: "1/2", 0.25: "1/4", 0.75: "3/4", 1.0: "1"}


def _style(key: str, betas: List[float], lw: float = figures.LW) -> dict:
    """Solid, in the run's hue; the last beta drawn first."""
    if key == NO_RECOVERY:
        return dict(color=NO_RECOVERY_COLOR, ls="-", lw=lw, zorder=6)
    order = betas.index(beta_of(key))
    return dict(color=RUN_COLORS[order % len(RUN_COLORS)], ls="-", lw=lw, zorder=5 - 0.1 * order)


def plot(runs, rates, labels, arm: TwoLinkArm, betas: List[float], title: str, outdir: Path, name: str):
    """(a) The arm in its workspace, at the start and at the end of each run, with the end-effector
    paths; (b) ``lambda_min(H)`` against time, over the slowest recovery each beta allows,
    ``lambda_min(H(0)) exp(-2 beta gamma t)``."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figures.apply_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9 * 2, 6.9 * 0.85), layout="constrained",
                                 width_ratios=(1.0, 1.25))
    first = next(iter(runs.values()))
    t = first.t
    start = first.theta[0]

    # (a) the arm at the start (black) and at the end of each run, and the end-effector paths
    loops._hero_bands(ax, arm)
    shapes = [arm.arm_points(start)] + [arm.arm_points(run.theta[-1]) for run in runs.values()]
    for key, end in zip(runs, shapes[1:]):
        style = _style(key, betas)
        path = np.array([arm.end_effector(theta) for theta in runs[key].theta])
        ax.plot(path[:, 0], path[:, 1], **style)
        ax.plot(end[:, 0], end[:, 1], "-o", color=style["color"], lw=0.45 * style["lw"], ms=1.4 * figures.SCALE,
                mew=0, alpha=0.9, zorder=style["zorder"], solid_capstyle="round")
    ax.plot(shapes[0][:, 0], shapes[0][:, 1], "-o", color="0.15", lw=figures.LW * 1.4,
            ms=2.6 * figures.SCALE, mew=0, zorder=10)
    ax.plot(0, 0, "s", color="0.2", ms=2.4 * figures.SCALE, zorder=11)
    corners = np.vstack(shapes)
    low, high = corners.min(axis=0), corners.max(axis=0)
    centre, half = 0.5 * (low + high), 0.6 * (high - low).max() + 0.1 * arm.reach
    ax.set(xlim=(centre[0] - half, centre[0] + half), ylim=(centre[1] - half, centre[1] + half),
           aspect="equal", xticks=[], yticks=[])
    ax.set_title("(a) the arm at the start (black) and at the end of each run", fontsize=figures.LABEL_FS * 0.8)
    handles = [Line2D([], [], color="0.15", lw=figures.LW * 1.4, marker="o", ms=2.6 * figures.SCALE, mew=0),
               Patch(**loops.HERO_VIOLATED)]
    ax.legend(handles, ["the start; the target stays here", r"$\lambda_{\min}(\mathbf{H})<0$"],
              loc="lower left", fontsize=figures.LEGEND_FS * 0.8)

    # (b) the margin against time, over the slowest recovery each beta allows
    margin0 = first.margin[0]
    for key, run in runs.items():
        bx.plot(t, run.margin, label=labels[key], **_style(key, betas))
    for order, beta in enumerate(betas):
        rate = 2 * beta * arm.gamma
        exponent = "-t" if rate == 1 else f"-{rate:g}t"
        bx.plot(t, margin0 * np.exp(-rate * t), color="black", lw=figures.LW * 0.75,
                ls=BOUND_LS[order % len(BOUND_LS)], zorder=8,
                label=rf"slowest allowed at $\beta={beta:g}$: $\lambda_{{\min}}(\mathbf{{H}}(0))\,e^{{{exponent}}}$")
    bx.axhline(0, color="0.3", lw=figures.REF_LW, zorder=1)
    bx.set_xlim(0, t[-1])
    bx.set_ylim(1.08 * margin0, -0.08 * margin0)
    bx.set_xlabel(r"$t$ [s]", fontsize=figures.LABEL_FS)
    bx.set_ylabel(r"$\lambda_{\min}(\mathbf{H})$", fontsize=figures.LABEL_FS)
    bx.grid(alpha=0.3, linewidth=figures.GRID_LW)
    bx.tick_params(labelsize=figures.TICK_FS)
    bx.set_title("(b) recovery of the margin, against the slowest the recovery constraint allows",
                 fontsize=figures.LABEL_FS * 0.8)
    bx.legend(loc="center right", fontsize=figures.LEGEND_FS * 0.8)
    fig.suptitle(title, fontsize=figures.LABEL_FS * 0.8)
    figures.save(fig, outdir, name)
    plt.close(fig)
    return fig


def plot_column(runs, arm: TwoLinkArm, betas: List[float], outdir: Path, name: str) -> Path:
    """Panel (b) for the paper, at its printed size and in the hero figure's fonts: ``lambda_min(H)``
    against time for each run, over ``lambda_min(H(0)) exp(-2 beta gamma t)`` for each beta. Written as
    ``name.pdf`` and ``name.png``; returns the PDF path."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    width, height = COLUMN_SIZE
    across, tall = 72.0 * width, 72.0 * height                     # points
    left, right, bottom, top = COLUMN_MARGINS
    first = next(iter(runs.values()))
    t, margin0 = first.t, first.margin[0]
    with plt.rc_context(loops.HERO_RC), loops._script_scale(loops.HERO_SCRIPT):
        fig = plt.figure(figsize=COLUMN_SIZE)
        ax = fig.add_axes([left / across, bottom / tall, (across - left - right) / across,
                           (tall - bottom - top) / tall])
        handles, labels = [], []
        for key, run in runs.items():
            style = _style(key, betas, COLUMN_LW)
            ax.plot(t, run.margin, solid_capstyle="round", **style)
            handles.append(Line2D([], [], **{k: v for k, v in style.items() if k != "zorder"}))
            labels.append("No recovery constraint" if key == NO_RECOVERY
                          else rf"$\beta={FRACTIONS.get(beta_of(key), f'{beta_of(key):g}')}$")
        for beta in betas:
            ax.plot(t, margin0 * np.exp(-2 * beta * arm.gamma * t), zorder=8, **COLUMN_BOUND)
        handles.append(Line2D([], [], **COLUMN_BOUND))
        labels.append(r"$\lambda_{\min}(\mathbf{H}(0))\,e^{-2\beta t}$" if arm.gamma == 1
                      else r"$\lambda_{\min}(\mathbf{H}(0))\,e^{-2\beta\gamma t}$")
        ax.axhline(0, color="0.3", lw=0.4, zorder=1)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(1.08 * margin0, -0.08 * margin0)
        ax.set_xticks(np.arange(0, t[-1] + 1e-9, 1.0))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.tick_params(length=1.5, pad=1.0)
        ax.grid(alpha=0.35, lw=0.35)
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$\lambda_{\min}(\mathbf{H})$")
        ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=7,
                  handlelength=1.7, handletextpad=0.5, labelspacing=0.25, borderaxespad=0.2, frameon=True,
                  framealpha=0.9, edgecolor="0.85", fancybox=False)
        for ext in (".pdf", ".png"):
            fig.savefig(outdir / f"{name}{ext}", dpi=600 if ext == ".png" else None)
        plt.close(fig)
    return outdir / f"{name}.pdf"


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--band", choices=BANDS, default="outer",
                        help="the band the arm starts in: outer (stretched) or inner (folded) (default: outer)")
    parser.add_argument("--depth", type=float, default=RECOVERY["depth"],
                        help="how far into the band the arm starts, as a fraction of its width, in (0, 1) "
                             "(default: 0.3)")
    parser.add_argument("--gain", type=float, default=RECOVERY["gain"],
                        help="tracking gain k of the nominal controller, 1/s (default: 5)")
    parser.add_argument("--duration", type=float, default=RECOVERY["horizon"], help="duration, s (default: 5)")
    parser.add_argument("--dt", type=float, default=RECOVERY["dt"], help="time step, s (default: 0.01)")
    parser.add_argument("--l1", type=float, default=ARM["l1"], help=f"length of link 1 (default: {ARM['l1']:g})")
    parser.add_argument("--l2", type=float, default=ARM["l2"], help=f"length of link 2 (default: {ARM['l2']:g})")
    parser.add_argument("--sigma-bar", type=float, default=ARM["sigma_bar"],
                        help=f"lower bound on sigma_min(J), in (0, min(l1, l2)) (default: {ARM['sigma_bar']:g})")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="rate of the class-K term alpha(H) = gamma H (default: 1)")
    parser.add_argument("--betas", type=float, nargs="+", default=list(RECOVERY["betas"]),
                        help="recovery margins of matrix OD, sym(Omega) >= beta I, one run each "
                             f"(default: {' '.join(f'{b:g}' for b in RECOVERY['betas'])}; omega_d/2 = {OMEGA_D / 2:g} "
                             "is the paper's design)")
    parser.add_argument("--rho-omega", type=float, default=paper.FILTER["rho_omega"],
                        help="penalty on omega - omega_d; rho_Omega = 2 rho_omega (default: 5e-3)")
    parser.add_argument("--out", type=Path, default=ROOT / "figures",
                        help="output directory (default: figures/ in the repository)")
    parser.add_argument("--name", default=None, help="file name without extension (default: fig_recovery_BAND)")
    args = parser.parse_args(argv)

    try:
        arm = TwoLinkArm(args.l1, args.l2, gamma=args.gamma, sigma_bar=args.sigma_bar)
        theta0 = band_start(arm, args.band, args.depth)
    except ValueError as err:
        parser.error(str(err))
    for flag, value in (("--duration", args.duration), ("--dt", args.dt), ("--rho-omega", args.rho_omega),
                        *(("--betas", b) for b in args.betas)):
        if not (np.isfinite(value) and value > 0):
            parser.error(f"{flag} must be positive and finite; got {value:g}")
    if len(set(args.betas)) != len(args.betas):
        parser.error("--betas must be distinct")
    steps = round(args.duration / args.dt)
    if not np.isclose(steps * args.dt, args.duration):
        parser.error("--dt must divide --duration")
    fastest = 2 * max(args.betas) * args.gamma
    if fastest * args.dt >= 1:
        parser.error(f"2 beta gamma dt must be below 1 for every beta, or a step overshoots the "
                     f"recovery; got {fastest * args.dt:g}")
    if not (np.isfinite(args.gain) and args.gain >= 0 and args.gain * args.dt < 1):
        parser.error(f"--gain must be nonnegative with --gain * --dt below 1; got {args.gain:g}")
    betas = list(args.betas)
    name = args.name or f"fig_recovery_{args.band}"

    r_lo, r_hi = arm.safe_radius_interval
    start = arm.end_effector(theta0)
    print(arm)
    print(f"safe set: {r_lo:.5f} <= |p| <= {r_hi:.5f}; the arm starts at rest {args.depth:g} of the way into "
          f"the {args.band} band, theta0 = ({theta0[0]:.4f}, {theta0[1]:.4f}), |p| = {np.linalg.norm(start):.5f}, "
          f"lambda_min(H) = {arm.margin(theta0):+.5f}")
    print(f"nominal: u_nom = J^-1 k (p* - p), p* = p(theta0) fixed, k = {args.gain:g}/s; "
          f"{args.duration:g} s, dt = {args.dt:g} s")
    print(f"filter: matrix OD, omega_d = {OMEGA_D:g}, rho_Omega = {2 * args.rho_omega:g}; the recovery constraint "
          f"sym(Omega) >= beta I at beta = {', '.join(f'{b:g}' for b in betas)}, and once without it; "
          f"odmcbf.simulate.run_recovery")

    runs, rates, labels, inaccurate = simulate(arm, args.band, args.depth, args.gain, args.duration, args.dt,
                                               args.rho_omega, betas)
    summaries = {key: summarize(arm, run, rates[key]) for key, run in runs.items()}
    print(f"\nClosed loop ({steps} steps each; {inaccurate} solves optimal_inaccurate):")
    print_table(summaries)
    print("Recovery rate: d/dt lambda_min(H) / |lambda_min(H)| under the applied u, the relative rate at which the "
          "margin rises (odmcbf.recovery_rate); the recovery constraint guarantees 2 beta gamma.")
    print("Step ratio: lambda_min(H(theta[k+1])) / lambda_min(H(theta[k])); at exactly the guaranteed rate it is "
          "1 - 2 beta gamma dt, up to terms of order dt^2.")
    print("v0' sym(Omega) v0: the decay along the unit eigenvector v0 of lambda_min(H); the recovery constraint "
          "keeps it at least beta.")
    print("\nWith the recovery constraint the margin shrinks per step by the factor 1 - 2 beta gamma dt:")
    for key in runs:
        if key == NO_RECOVERY:
            continue
        s = summaries[key]
        print(f"  beta = {beta_of(key):<5g}: step ratio in [{s['ratio_min']:.6f}, {s['ratio_max']:.6f}], "
              f"1 - 2 beta gamma dt = {1 - s['guaranteed'] * args.dt:.6f}; recovery rate in "
              f"[{s['rate_min']:.5f}, {s['rate_max']:.5f}], 2 beta gamma = {s['guaranteed']:g}")
    s = summaries[NO_RECOVERY]
    print(f"  without it: the margin ends at {s['shrunk']:.3f} of its start; the recovery rate falls from "
          f"{s['rate_max']:.3f} to {s['rate_end']:.1e}, and v0' sym(Omega) v0 from {s['decay_start']:.3f} to "
          f"{s['decay_end']:.1e}")

    title = (f"Matrix OD recovering from the {args.band} band: the arm starts {args.depth:g} of the way in, "
             f"with its target at the start;  $l_1={arm.l1:g}$, $l_2={arm.l2:g}$, "
             rf"$\bar\sigma={arm.sigma_bar:.4g}$, $\gamma={arm.gamma:g}$, $\omega_d={OMEGA_D:g}$, $k={args.gain:g}$/s")
    plot(runs, rates, labels, arm, betas, title, args.out, name)
    print(f"\nwrote {args.out / (name + '.png')}")
    pdf = plot_column(runs, arm, betas, args.out, name + "_column")
    print(f"wrote {pdf} and {pdf.with_suffix('.png')} (the paper's figure, at its printed size)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
