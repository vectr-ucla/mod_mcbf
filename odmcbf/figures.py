"""Figures and animation for the singularity-avoidance example.

``snapshots``    timelapse of the closed loop, one panel per controller.
``closed_loop``  safety margin, control deviation, and decay-variable deviation.
``animate``      side-by-side playback of the four closed loops.
``workspace``    the arm in given configurations, its workspace, and where H is violated.
``margin_status`` the margin of a configuration and whether it is safe, on the boundary,
                 or violates H, as ``workspace`` labels it.

The arm drawn by ``snapshots`` and ``animate`` is the one the traces were run
with unless another is passed explicitly; axis limits follow its reach.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch, Wedge
import matplotlib.pyplot as plt

from .simulate import Trace
from .two_link import TwoLinkArm

ORDER = ("nominal", "standard", "scalar", "matrix")
COLORS = {
    "nominal": "tab:gray",
    "standard": "tab:red",
    "scalar": "tab:orange",
    "matrix": "tab:blue",
}
CMAPS = {"nominal": "Greys", "standard": "Reds", "scalar": "YlOrBr", "matrix": "Blues"}
TITLES = {
    "nominal": "Nominal",
    "standard": "Standard MCBF",
    "scalar": "Scalar OD",
    "matrix": "Matrix OD",
}
#: Snapshot spacing per panel, in seconds. All four closed loops keep rotating
#: the arm about joint 1, so a common spacing gives comparable panels.
SNAPSHOT_SPACING = {"nominal": 1.0, "standard": 1.0, "scalar": 1.0, "matrix": 1.0}

SCALE = 2.0            # static figures are drawn at 2x their printed size
LABEL_FS = 8 * SCALE
TICK_FS = 7 * SCALE
LEGEND_FS = 6.5 * SCALE
LW = 1.1 * SCALE
REF_LW = 0.65 * SCALE
GRID_LW = 0.4 * SCALE
DRAW_SCALE = 0.5       # visual scale of the ellipses and threshold circles
SNAPSHOT_MARGIN = 1.14 # axis half-width of the snapshots, in units of the arm's reach
ANIMATE_MARGIN = 1.35  # axis half-width of the animation, in units of the arm's reach
WORKSPACE_COLUMNS = 4  # panels per row of the workspace figure


def apply_style() -> None:
    """Serif text, STIX math, embeddable fonts; applied by every static figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.6 * SCALE,
        "grid.alpha": 0.3,
        "grid.linewidth": GRID_LW,
        "legend.framealpha": 0.82,
        "legend.edgecolor": "0.8",
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, outdir: Optional[Path], name: str) -> None:
    """Write ``name.png`` into ``outdir``; a ``None`` outdir skips saving."""
    if outdir is None:
        return
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png")


def snapshots(
    traces: Dict[str, Trace],
    outdir: Optional[Path] = None,
    arm: Optional[TwoLinkArm] = None,
    name: str = "fig_snapshots",
    width: float = 6.9,
):
    """Timelapse of every controller, with time shown by colour.

    Each sampled instant draws the arm, the singular-value ellipse at the
    end-effector, and the threshold circle of radius proportional to
    ``sigma_bar``. The constraint holds exactly when the ellipse contains the
    circle. Colour darkens with time on a common 0-to-horizon scale.
    """
    apply_style()
    arm = _resolve_arm(traces, arm)
    horizon = float(traces[ORDER[0]].t[-1])
    lim = SNAPSHOT_MARGIN * arm.reach

    fig, axs = plt.subplots(1, len(ORDER), figsize=(width, width * 2.6 / 7.05 * 0.80))
    for ax, key in zip(axs, ORDER):
        trace, cmap = traces[key], plt.get_cmap(CMAPS[key])
        spacing = SNAPSHOT_SPACING[key]
        for t in np.arange(0, horizon + spacing / 2, spacing):
            idx = int(np.argmin(abs(trace.t - t)))
            if not np.isclose(trace.t[idx], t):
                raise ValueError(f"no sample at t = {t} s; check dt against the spacing")
            theta = trace.theta[idx]
            shade = cmap(0.30 + 0.62 * t / horizon)     # common time scale across panels
            points = arm.arm_points(theta)
            width_, height_, angle = _ellipse_axes(arm, theta)
            ax.add_patch(Ellipse(points[-1], width_, height_, angle=angle,
                                 facecolor=shade, edgecolor=shade, alpha=0.20,
                                 lw=0.6 * SCALE, zorder=1))
            ax.add_patch(Circle(points[-1], DRAW_SCALE * arm.sigma_bar,
                                fill=False, edgecolor="black", linestyle="--",
                                lw=0.55 * SCALE, alpha=0.55, zorder=3))
            ax.plot(points[:, 0], points[:, 1], marker="o", color=shade,
                    lw=1.4 * SCALE, ms=2.4 * SCALE, solid_capstyle="round", zorder=2)
        ax.plot(0, 0, marker="s", ms=2.4 * SCALE, color="0.25", zorder=4)   # joint 1
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[],
               aspect="equal")
        ax.set_title(TITLES[key], fontsize=LABEL_FS, pad=4 * SCALE)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.75")
            spine.set_linewidth(0.5 * SCALE)
        bar = fig.colorbar(ScalarMappable(norm=Normalize(0, horizon), cmap=cmap),
                           ax=ax, orientation="horizontal", fraction=0.045,
                           pad=0.03, aspect=28, shrink=0.82)
        bar.set_ticks([0, horizon])
        bar.set_ticklabels(["$0$", f"${horizon:g}$"])
        bar.ax.tick_params(labelsize=6 * SCALE, length=2 * SCALE, pad=1 * SCALE)
        bar.set_label("$t$ [s]", fontsize=6.5 * SCALE, labelpad=1 * SCALE)
        bar.outline.set_linewidth(0.4 * SCALE)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.90, wspace=0.05)
    save(fig, outdir, name)
    plt.close(fig)          # pyplot lets go of it; the caller gets the figure
    return fig


def closed_loop(
    traces: Dict[str, Trace],
    u_nom,
    outdir: Optional[Path] = None,
    omega_d: float = 1.0,
    name: str = "fig_closed_loop",
):
    """Safety margin, control deviation, and decay-variable deviation.

    Panel (c) overlays the smallest scalar decay that certifies the applied
    control, which the scalar filter tracks once the constraint is active.
    """
    apply_style()
    u_nom = np.asarray(u_nom, dtype=float)
    horizon = float(traces[ORDER[0]].t[-1])

    fig, axs = plt.subplots(3, 1, figsize=(6.9, 6.0), sharex=True)
    for ax in axs:
        ax.grid(alpha=0.3, linewidth=GRID_LW)
        ax.set_xlim(0, horizon)
        ax.tick_params(axis="both", labelsize=TICK_FS,
                       width=0.5 * SCALE, length=1.5 * SCALE)

    # (a) safety margin
    for key in ORDER:
        trace = traces[key]
        axs[0].plot(trace.t, trace.margin, color=COLORS[key], lw=LW, label=trace.label)
    axs[0].axhline(0, color="k", lw=REF_LW)
    axs[0].set_ylabel(r"$\lambda_{\min}(\mathbf{H})$", fontsize=LABEL_FS)
    _method_legend(axs[0], loc="lower right")

    # (b) control deviation
    for key in ORDER:
        trace = traces[key]
        axs[1].plot(trace.t[:-1], trace.control_deviation(u_nom),
                    color=COLORS[key], lw=LW)
    axs[1].set_ylabel(r"$\|\mathbf{u}-\mathbf{k}_{\mathrm{nom}}\|$", fontsize=LABEL_FS)

    # (c) decay-variable deviations, with the smallest certifying scalar decay
    scalar, matrix = traces["scalar"], traces["matrix"]
    bound = np.maximum(omega_d, scalar.scalar_certificate) - omega_d
    axs[2].plot(scalar.t[:-1], scalar.decay_deviation, color=COLORS["scalar"],
                lw=LW, zorder=3, label=r"$\omega-\omega_d$")
    axs[2].plot(matrix.t[:-1], matrix.decay_deviation, color=COLORS["matrix"],
                lw=LW, zorder=3, label=r"$\|\mathbf{\Omega}-\mathbf{\Omega}_d\|_F$")
    # Drawn last and on top: it nearly coincides with the scalar curve.
    axs[2].plot(scalar.t[:-1], bound, "k--", lw=LW, zorder=4,
                label="Prop. 2 lower bound")
    axs[2].set_ylabel("decay-variable deviation", fontsize=LABEL_FS)
    axs[2].set_xlabel(r"$t$ [s]", fontsize=LABEL_FS)
    axs[2].legend(loc="center right", ncols=1, fontsize=LEGEND_FS, frameon=True,
                  framealpha=0.82, handlelength=1.5, borderpad=0.25, labelspacing=0.2)

    for ax, tag in zip(axs, "abc"):
        ax.text(0.008, 0.97, f"({tag})", transform=ax.transAxes,
                ha="left", va="top", fontsize=LABEL_FS)

    fig.align_ylabels(axs)
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.075, top=0.99, hspace=0.09)
    save(fig, outdir, name)
    plt.close(fig)          # pyplot lets go of it; the caller gets the figure
    return fig


def animate(
    traces: Dict[str, Trace],
    arm: Optional[TwoLinkArm] = None,
    skip: int = 5,
    interval: int = 80,
):
    """Play the four closed loops side by side.

    Each panel shows the arm, the singular-value ellipse at the end-effector,
    and the dashed threshold circle. The constraint is active exactly when the
    ellipse's minor axis touches the circle. In the constant-command run of
    ``main.ipynb``, the nominal arm drives through it
    and collapses at the singularity. The three filters all end up rotating
    the whole arm about joint 1: the standard filter from well inside the safe
    set, the scalar OD filter from just inside it, and the matrix OD filter
    along the boundary itself, at the rate closest to the command.

    Display in a notebook with ``HTML(animate(traces).to_jshtml(fps=12))``.
    """
    arm = _resolve_arm(traces, arm)
    frames = np.arange(0, len(traces[ORDER[0]].t), skip)
    lim = ANIMATE_MARGIN * arm.reach

    fig, axs = plt.subplots(1, len(ORDER), figsize=(3.4 * len(ORDER), 4.0))
    arms, ellipses, circles, titles = [], [], [], []
    for ax, key in zip(axs, ORDER):
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[], aspect="equal")
        ax.grid(alpha=0.25)
        (line,) = ax.plot([], [], "o-", lw=3, color="k", ms=5, zorder=3)
        ellipse = Ellipse((0, 0), 1, 1, angle=0, facecolor=COLORS[key],
                          alpha=0.35, edgecolor=COLORS[key], lw=1.5, zorder=1)
        circle = Circle((0, 0), DRAW_SCALE * arm.sigma_bar, fill=False,
                        edgecolor="k", linestyle="--", lw=1.2, zorder=2)
        ax.add_patch(ellipse)
        ax.add_patch(circle)
        arms.append(line)
        ellipses.append(ellipse)
        circles.append(circle)
        titles.append(ax.set_title(TITLES[key], fontsize=10))

    def update(frame):
        k = frames[frame]
        for j, key in enumerate(ORDER):
            trace = traces[key]
            points = arm.arm_points(trace.theta[k])
            arms[j].set_data(points[:, 0], points[:, 1])
            width, height, angle = _ellipse_axes(arm, trace.theta[k])
            ellipses[j].set_center(points[-1])
            ellipses[j].width, ellipses[j].height = width, height
            ellipses[j].angle = angle
            circles[j].center = points[-1]
            titles[j].set_text(
                f"{TITLES[key]}\n$t$ = {trace.t[k]:.1f} s,  "
                f"$\\lambda_{{\\min}}(\\mathbf{{H}})$ = {trace.margin[k]:+.3f}")
        return arms + ellipses + circles

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=interval, blit=False)
    plt.close(fig)                                      # the animation is the output
    return ani


def workspace(
    arm: TwoLinkArm,
    theta,
    outdir: Optional[Path] = None,
    name: str = "fig_workspace",
    panel_width: float = 3.6,
):
    """The arm in its workspace, with the region where ``H`` is violated.

    ``theta`` is one configuration ``(theta1, theta2)`` or a stack of them,
    one panel each, at most ``WORKSPACE_COLUMNS`` to a row. Link lengths and
    threshold are those of ``arm``, so
    ``workspace(TwoLinkArm(l1, l2, sigma_bar=s), theta)`` draws any
    ``l1, l2 > 0`` and ``0 < s < min(l1, l2)``; they are printed above the
    panels.

    Each panel shows the workspace ``|l1 - l2| <= |p| <= l1 + l2`` (solid
    circles; the inner one is omitted when ``l1 = l2``) with everything outside
    it in grey, the part of it where ``lambda_min(H) < 0`` (hatched), the
    boundary of the safe set, ``lambda_min(H) = 0`` (dashed circles), and the
    arm. Whether ``H >= 0`` depends on the end-effector position alone, see
    ``TwoLinkArm.safe_radius_interval``, so the shading is the same for every
    configuration. Everything is drawn to scale: with a small ``sigma_bar`` or
    very unequal links a band can be thinner than a line. Each title gives the
    configuration and its margin and status from ``margin_status``.
    """
    apply_style()
    thetas = np.atleast_2d(np.asarray(theta, dtype=float))
    if thetas.ndim != 2 or thetas.shape[1] != 2 or len(thetas) == 0:
        raise ValueError(f"theta must have shape (2,) or (k, 2), k >= 1; got {np.shape(theta)}")
    if not np.all(np.isfinite(thetas)):
        raise ValueError("theta must be finite")
    r_in, r_out = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    lim = SNAPSHOT_MARGIN * arm.reach

    violated = dict(facecolor=to_rgba(COLORS["standard"], 0.12),
                    edgecolor=to_rgba(COLORS["standard"], 0.55),
                    hatch="////", lw=0, zorder=0)
    # The safe-set boundary sits above the workspace circles, which it nears as sigma_bar -> 0.
    boundary = dict(fill=False, edgecolor=COLORS["standard"], linestyle="--",
                    lw=REF_LW * 1.4, zorder=2)
    reach = dict(fill=False, edgecolor="0.2", lw=REF_LW * 1.4, zorder=1)
    outside = dict(facecolor="0.82", edgecolor="none", lw=0, zorder=0)
    corner = 1.01 * np.hypot(lim, lim)         # the grey reaches the corners of the axes

    count = len(thetas)
    ncols = min(count, WORKSPACE_COLUMNS)
    nrows = -(-count // ncols)
    fig, axs = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                            figsize=(panel_width * ncols, panel_width * 1.3 * nrows))
    for ax in axs.flat[count:]:
        fig.delaxes(ax)
    for ax, th in zip(axs.flat, thetas):
        ax.add_patch(Wedge((0, 0), corner, 0, 360, width=corner - r_out, **outside))
        if r_in > 0:
            ax.add_patch(Circle((0, 0), r_in, **outside))
        for outer, inner in ((r_out, r_hi), (r_lo, r_in)):   # outer band, inner band
            if outer > inner:
                ax.add_patch(Wedge((0, 0), outer, 0, 360, width=outer - inner, **violated))
        for radius in (r_lo, r_hi):
            ax.add_patch(Circle((0, 0), radius, **boundary))
        for radius in (r_in, r_out):
            if radius > 0:
                ax.add_patch(Circle((0, 0), radius, **reach))

        points = arm.arm_points(th)
        ax.plot(points[:, 0], points[:, 1], color="0.1", lw=1.4 * SCALE,
                solid_capstyle="round", zorder=3)
        ax.plot(points[1:, 0], points[1:, 1], "o", color="0.1", mec="white",
                mew=0.4 * SCALE, ms=3.2 * SCALE, zorder=4)
        ax.plot(0, 0, marker="s", ms=3.2 * SCALE, color="0.1", mec="white",
                mew=0.4 * SCALE, zorder=4)                                   # joint 1

        margin, status = margin_status(arm, th)
        ax.set_title(f"$\\theta=({th[0]:.2f},\\,{th[1]:.2f})$\n"
                     f"$\\lambda_{{\\min}}(\\mathbf{{H}})={margin:+.3g}$  ({status})",
                     fontsize=LABEL_FS * 0.8, pad=3 * SCALE)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
        ax.tick_params(labelsize=TICK_FS * 0.8, width=0.5 * SCALE, length=1.5 * SCALE)
        ax.set_xlabel("$x$", fontsize=LABEL_FS * 0.8, labelpad=1 * SCALE)
        for spine in ax.spines.values():
            spine.set_color("0.75")
            spine.set_linewidth(0.5 * SCALE)
    for ax in axs[:, 0]:
        ax.set_ylabel("$y$", fontsize=LABEL_FS * 0.8, labelpad=1 * SCALE)

    handles = [
        Line2D([], [], color="0.2", lw=REF_LW * 1.4),
        Patch(**{k: v for k, v in outside.items() if k != "zorder"}),
        Line2D([], [], color=COLORS["standard"], linestyle="--", lw=REF_LW * 1.4),
        Patch(**{k: v for k, v in violated.items() if k != "zorder"}),
    ]
    labels = ["workspace boundary", "outside the workspace",
              r"$\lambda_{\min}(\mathbf{H})=0$", r"$\lambda_{\min}(\mathbf{H})<0$"]
    fig.legend(handles, labels, loc="outside lower center", ncols=4 if ncols > 1 else 2,
               fontsize=LEGEND_FS * 0.8, frameon=False, handlelength=1.8,
               columnspacing=1.2)
    fig.suptitle(f"$l_1={arm.l1:g}$,  $l_2={arm.l2:g}$,  "
                 f"$\\bar\\sigma={arm.sigma_bar:.4g}$", fontsize=LABEL_FS * 0.8)
    save(fig, outdir, name)
    plt.close(fig)          # pyplot lets go of it; the caller gets the figure
    return fig


def margin_status(arm: TwoLinkArm, theta) -> tuple[float, str]:
    """Margin ``lambda_min(H(theta))`` and ``"safe"``, ``"on boundary"`` or ``"violated"``.

    A margin within ``1e-12 (l1**2 + l2**2)`` of zero, far above the rounding
    error of ``H`` at any scale of the arm, counts as on the boundary and is
    returned as exactly zero.
    """
    margin = arm.margin(theta)
    if abs(margin) <= 1e-12 * (arm.l1 ** 2 + arm.l2 ** 2):
        return 0.0, "on boundary"
    return margin, "safe" if margin > 0 else "violated"


def _resolve_arm(traces: Dict[str, Trace], arm: Optional[TwoLinkArm]) -> TwoLinkArm:
    """The arm to draw: the explicit one, else the one the traces were run with."""
    if arm is not None:
        return arm
    for trace in traces.values():
        if trace.arm is not None:
            return trace.arm
    return TwoLinkArm()


def _ellipse_axes(arm: TwoLinkArm, theta):
    """Full axes and orientation of the end-effector singular-value ellipse."""
    lam, vectors = np.linalg.eigh(arm.M(theta))
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    return (2 * DRAW_SCALE * np.sqrt(max(0.0, lam[1])),
            2 * DRAW_SCALE * np.sqrt(max(0.0, lam[0])),
            angle)


def _method_legend(ax, loc="best", **kwargs):
    """Two-column method legend, ordered so the rows read across."""
    handles, text = ax.get_legend_handles_labels()
    by_label = dict(zip(text, handles))
    # Matplotlib fills multi-column legends column first.
    display = ["nominal", "scalar OD", "standard", "matrix OD"]
    ordered = [label for label in display if label in by_label]
    opts = dict(fontsize=LEGEND_FS, frameon=True, framealpha=0.82, handlelength=1.3,
                columnspacing=0.65, borderpad=0.25, labelspacing=0.2)
    opts.update(kwargs)
    return ax.legend([by_label[k] for k in ordered], ordered, loc=loc, ncols=2, **opts)
