"""Where the singularity-avoidance barrier is not positive definite.

``H = J J^T - sigma_bar^2 I`` is the barrier of ``two_link_manipulator_kinematics.md``
in the absolute joint angles ``theta = (theta1, theta2)`` of
:class:`~odmcbf.two_link.TwoLinkArm`. It fails to be positive definite exactly
where ``lambda_min(H) <= 0``, that is where ``sigma_min(J) <= sigma_bar``. From

    tr H  = l1**2 + l2**2 - 2 sigma_bar**2,
    det H = (l1**2 - sigma_bar**2) (l2**2 - sigma_bar**2) - l1**2 l2**2 cos(theta2 - theta1)**2,

the set depends on ``theta`` only through ``theta2 - theta1``, and it takes one of
three forms:

``sigma_bar = 0``
    the singular configurations ``theta2 - theta1 = 0`` (stretched) and ``pi``
    (folded), mod ``2 pi``: two families of lines in the ``(theta1, theta2)`` plane;
``0 < sigma_bar < min(l1, l2)``
    ``cos(theta2 - theta1)**2 >= (l1**2 - sigma_bar**2) (l2**2 - sigma_bar**2) / (l1**2 l2**2)``:
    one band around each of those lines, bounded by lines on which
    ``lambda_min(H) = 0``;
``sigma_bar >= min(l1, l2)``
    every configuration, because ``sigma_min(J)`` never exceeds ``min(l1, l2)``.

:class:`TwoLinkArm` accepts only ``0 < sigma_bar < min(l1, l2)``, the thresholds
whose safe set has an interior. Everything here accepts any ``sigma_bar >= 0``.

:class:`NotPDRegion`
    The exact set for one arm and one ``sigma_bar``; no plotting.
:func:`plot_joint_space`, :func:`plot_eigenvalues`, :func:`plot_joint_path`
    Draw onto a given matplotlib ``Axes``, so they combine with other figures.
:func:`plot_not_pd_regions`
    Both panels in one figure, optionally with paths of configurations on top.

The same set in the workspace, as radii of the end effector, is drawn by
:func:`odmcbf.figures.workspace`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.ticker import AutoLocator, FuncFormatter, MultipleLocator, ScalarFormatter

from .figures import COLORS, GRID_LW, LABEL_FS, LEGEND_FS, LW, REF_LW, SCALE, TICK_FS
from .figures import apply_style, save
from .two_link import TwoLinkArm, TwoLinkKinematics, band_half_width

TWO_PI = 2 * np.pi

#: Default window of the joint-space panel: one period in each joint angle.
DEFAULT_WINDOW = ((-np.pi, np.pi), (-np.pi, np.pi))

#: Styles shared by both panels. The colours match ``figures.workspace``; the
#: hatch runs across the bands, which have slope 1 in the joint-space panel.
NOT_PD_STYLE = dict(facecolor=to_rgba(COLORS["standard"], 0.12),
                    edgecolor=to_rgba(COLORS["standard"], 0.55),
                    hatch="\\\\\\\\", lw=0)
BOUNDARY_STYLE = dict(color=COLORS["standard"], linestyle="--", lw=REF_LW * 1.4)
DEGENERATE_STYLE = dict(color=COLORS["standard"], linestyle="-", lw=REF_LW * 2.2)
SINGULAR_STYLE = dict(color="0.35", linestyle=":", lw=REF_LW * 1.2)
LAMBDA_MIN_COLOR, LAMBDA_MAX_COLOR = "0.1", "0.45"

#: Line widths of overlaid paths, as multiples of ``LW``: the first path is the
#: widest and each later one, drawn on top, is thinner, so a path that runs
#: under a later one still shows along its edges. The topmost path is also
#: dashed, so a path it covers completely still shows through the gaps.
PATH_WIDTHS = (1.7, 0.9)
#: Dash-dot, unlike the dashed ``lambda_min(H) = 0`` lines.
TOP_PATH_DASHES = (0, (6.0, 1.5, 1.5, 1.5))
#: Colours for paths that do not name a controller; none is a controller colour.
PATH_FALLBACK_COLORS = ("tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan")
#: Paths with larger angles, in radians, are refused rather than wrapped.
MAX_PATH_ANGLE = 1e12


class NotPDRegion:
    """The configurations where ``H = J J^T - sigma_bar^2 I`` is not positive definite.

    Parameters
    ----------
    kinematics:
        Supplies the link lengths ``l1`` and ``l2``; a :class:`TwoLinkArm`
        works too, and its own ``sigma_bar`` is then ignored.
    sigma_bar:
        Any threshold ``sigma_bar >= 0``.

    The set is closed: configurations with ``lambda_min(H) = 0`` belong to it.
    """

    def __init__(self, kinematics: TwoLinkKinematics, sigma_bar: float) -> None:
        sigma_bar = float(sigma_bar)
        if not np.isfinite(sigma_bar) or sigma_bar < 0:
            raise ValueError(f"sigma_bar must be finite and nonnegative; got {sigma_bar:g}")
        self.kinematics = kinematics
        self.l1 = float(kinematics.l1)
        self.l2 = float(kinematics.l2)
        self.sigma_bar = sigma_bar

    @classmethod
    def from_arm(cls, arm: TwoLinkArm) -> NotPDRegion:
        """The region for the arm's own threshold ``arm.sigma_bar``."""
        return cls(arm, arm.sigma_bar)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(l1={self.l1:g}, l2={self.l2:g}, "
                f"sigma_bar={self.sigma_bar:.6g})")

    # ------------------------------------------------------------- the set
    @property
    def everywhere(self) -> bool:
        """Whether ``H`` is not positive definite anywhere, ``sigma_bar >= min(l1, l2)``."""
        return self.sigma_bar >= min(self.l1, self.l2)

    @property
    def singular_only(self) -> bool:
        """Whether the set is only the singular lines, ``sigma_bar = 0``."""
        return self.sigma_bar == 0.0

    @property
    def half_width(self) -> float:
        """Half-width ``w`` in ``[0, pi/2]`` of each band, in ``theta2 - theta1``.

        The set is ``|theta2 - theta1| <= w`` or ``|theta2 - theta1| >= pi - w``
        (mod ``2 pi``), with ``cos(w)**2`` the right-hand side of the ``det H``
        condition in the module docstring; see
        :func:`~odmcbf.two_link.band_half_width`. It is ``0`` for
        ``sigma_bar = 0`` and ``pi/2`` (the whole plane) for
        ``sigma_bar >= min(l1, l2)``.
        """
        if self.everywhere:
            return np.pi / 2
        return band_half_width(self.l1, self.l2, self.sigma_bar)

    @property
    def theta_diff_intervals(self) -> list[tuple[float, float]]:
        """Closed intervals of ``theta2 - theta1`` in ``[-pi, pi]`` that make up the set.

        The band around ``pi`` wraps, so it appears as two intervals, one at
        each end. With ``sigma_bar = 0`` the intervals have zero length.
        """
        if self.everywhere:
            return [(-np.pi, np.pi)]
        w = self.half_width
        return [(-np.pi, -np.pi + w), (-w, w), (np.pi - w, np.pi)]

    @property
    def boundary_theta_diffs(self) -> list[float]:
        """Values of ``theta2 - theta1`` in ``(-pi, pi]`` where ``lambda_min(H) = 0``.

        Four values when ``0 < sigma_bar < min(l1, l2)``; the singular values
        ``0`` and ``pi`` when ``sigma_bar = 0``; ``-pi/2`` and ``pi/2``, where
        ``sigma_min(J)`` peaks, when ``sigma_bar = min(l1, l2)``; none above it.
        """
        low = min(self.l1, self.l2)
        if self.sigma_bar > low:
            return []
        if self.sigma_bar == low:
            return [-np.pi / 2, np.pi / 2]
        if self.singular_only:
            return [0.0, np.pi]
        w = self.half_width
        return [-np.pi + w, -w, w, np.pi - w]

    def _bands(self) -> list[tuple[float, float]]:
        """The bands as intervals of ``theta2 - theta1`` mod ``2 pi``, without wrapping."""
        w = self.half_width
        return [(-w, w), (np.pi - w, np.pi + w)]

    def contains(self, theta) -> np.ndarray:
        """Whether each configuration ``theta = (theta1, theta2)`` lies in the set.

        ``theta`` has shape ``(2,)`` or ``(..., 2)``. The test is exact in
        ``theta2 - theta1``; a configuration on the boundary counts as inside,
        up to rounding in ``theta2 - theta1`` and in its reduction mod ``2 pi``.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape[-1:] != (2,):
            raise ValueError(f"theta must have shape (2,) or (..., 2); got {theta.shape}")
        if self.everywhere:
            return np.ones(theta.shape[:-1], dtype=bool)
        diff = np.abs(_wrap(theta[..., 1] - theta[..., 0]))       # in [0, pi]
        w = self.half_width
        return (diff <= w) | (diff >= np.pi - w)

    # ----------------------------------------------------- the spectrum of H
    def eigenvalues(self, theta_diff):
        """Eigenvalues ``(lambda_min, lambda_max)`` of ``H`` versus ``theta2 - theta1``."""
        mu_min, mu_max = self.kinematics.singular_values_sq(theta_diff)
        with np.errstate(over="ignore"):
            r = np.square(np.float64(self.sigma_bar))  # inf rather than OverflowError
        return mu_min - r, mu_max - r

    def min_eigenvalue(self, theta_diff):
        """``lambda_min(H) = sigma_min(J)**2 - sigma_bar**2`` versus ``theta2 - theta1``."""
        return self.eigenvalues(theta_diff)[0]


# ------------------------------------------------------------------ drawing
def plot_joint_space(
    region: NotPDRegion,
    ax=None,
    *,
    window=DEFAULT_WINDOW,
    show_boundary: bool = True,
    show_singular: bool = True,
):
    """Draw the set in the ``(theta1, theta2)`` plane onto ``ax``.

    The set is hatched, lines with ``lambda_min(H) = 0`` are dashed, and the
    singular lines ``theta2 - theta1 = 0`` and ``pi``, where ``J`` has rank
    one, are dotted. All of them are exact: each band is clipped to
    ``window = ((theta1_lo, theta1_hi), (theta2_lo, theta2_hi))`` as a polygon.
    Lines of zero width (``sigma_bar = 0``) are drawn solid. Returns ``ax``.
    """
    ax = plt.gca() if ax is None else ax
    (x0, x1), (y0, y1) = window = _check_window(window)

    if region.everywhere:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, zorder=0, **NOT_PD_STYLE))
    elif region.singular_only:
        for value in (0.0, np.pi):
            for seg in _diagonal_segments(value, window):
                ax.plot(*seg, zorder=2, **DEGENERATE_STYLE)
    else:
        for lo, hi in region._bands():
            for poly in _strip_polygons(lo, hi, window):
                ax.add_patch(Polygon(poly, closed=True, zorder=0, **NOT_PD_STYLE))
    if show_singular and not region.singular_only:
        for value in (0.0, np.pi):
            for seg in _diagonal_segments(value, window):
                ax.plot(*seg, zorder=1, **SINGULAR_STYLE)
    if show_boundary and not region.singular_only:
        for value in region.boundary_theta_diffs:
            for seg in _diagonal_segments(value, window):
                ax.plot(*seg, zorder=2, **BOUNDARY_STYLE)

    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal")
    step = _pi_step(max(x1 - x0, y1 - y0))          # equal aspect: one step for both axes
    _pi_axis(ax.xaxis, x0, x1, step)
    _pi_axis(ax.yaxis, y0, y1, step)
    ax.set_xlabel(r"$\theta_1$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\theta_2$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS, width=0.5 * SCALE, length=1.5 * SCALE)
    return ax


def plot_joint_path(ax, theta, *, window=DEFAULT_WINDOW, outline: bool = True, **line_kwargs):
    """Draw a path of configurations ``theta`` (shape ``(N, 2)``) onto ``ax``.

    Both angles are wrapped into the window. Where the path leaves through one
    side it is drawn up to that side and resumes from the opposite one, and a
    window wider than one period shows every ``2 pi`` copy of the path.
    ``line_kwargs`` go to ``ax.plot`` (``linewidth`` defaults to ``LW``,
    ``zorder`` to 3). With ``outline`` a white line below the path keeps it
    visible over the hatching; it sits just below ``zorder``, so it never
    covers another path drawn at the same ``zorder``. A single configuration
    (or a path that stays within 1e-6 rad) is drawn as a marker. Angles
    beyond ``MAX_PATH_ANGLE`` are refused. Returns the list of ``Line2D``
    objects of the path.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    if theta.ndim != 2 or theta.shape[1] != 2 or len(theta) == 0:
        raise ValueError(f"theta must have shape (N, 2) with N >= 1; got {np.shape(theta)}")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must be finite")
    if np.abs(theta).max() > MAX_PATH_ANGLE:
        raise ValueError(f"theta must stay within +-{MAX_PATH_ANGLE:g} rad to be wrapped")
    (x0, x1), (y0, y1) = _check_window(window)

    kwargs = cbook.normalize_kwargs(line_kwargs, Line2D)
    kwargs.setdefault("linewidth", LW)
    kwargs.setdefault("zorder", 3)
    if np.ptp(theta, axis=0).max() <= 1e-6:        # one configuration, or a path that stays put
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 2.4 * SCALE)
        kwargs.setdefault("markeredgecolor", "white")
    x, y = _wrapped_polyline(theta, (x0, y0))
    lines = []
    for i in range(max(1, math.ceil((x1 - x0) / TWO_PI - 1e-12))):
        for j in range(max(1, math.ceil((y1 - y0) / TWO_PI - 1e-12))):
            xs, ys = x + TWO_PI * i, y + TWO_PI * j
            if outline and len(theta) > 1:
                ax.plot(xs, ys, color="white", linewidth=kwargs["linewidth"] + 0.9 * SCALE,
                        solid_capstyle="round", zorder=kwargs["zorder"] - 0.1)
            copy = dict(kwargs)
            if lines:
                copy.pop("label", None)
            lines += ax.plot(xs, ys, **copy)
            if kwargs.get("color") is None:          # every copy keeps the first copy's colour
                kwargs["color"] = lines[0].get_color()
    return lines


def plot_eigenvalues(region: NotPDRegion, ax=None, *, samples: int = 1441, legend: bool = True):
    """Draw both eigenvalues of ``H`` against ``theta2 - theta1`` onto ``ax``.

    The intervals where ``lambda_min(H) <= 0`` are hatched as in
    :func:`plot_joint_space`, with the same dashed and dotted lines. With
    ``legend`` the two curves are named in a legend inside the axes. Returns ``ax``.
    """
    ax = plt.gca() if ax is None else ax
    diff = np.linspace(-np.pi, np.pi, samples)
    diff = np.union1d(diff, region.boundary_theta_diffs)
    lam_min, lam_max = region.eigenvalues(diff)

    if region.singular_only:
        for value in (-np.pi, 0.0, np.pi):
            ax.axvline(value, zorder=2, **DEGENERATE_STYLE)
    else:
        for lo, hi in region.theta_diff_intervals:
            ax.add_patch(Rectangle((lo, 0), hi - lo, 1, transform=ax.get_xaxis_transform(),
                                   zorder=0, **NOT_PD_STYLE))
        for value in (-np.pi, 0.0, np.pi):
            ax.axvline(value, zorder=1, **SINGULAR_STYLE)
        for value in region.boundary_theta_diffs:
            ax.axvline(value, zorder=2, **BOUNDARY_STYLE)
    ax.axhline(0.0, color="k", lw=REF_LW, zorder=1)
    ax.plot(diff, lam_max, color=LAMBDA_MAX_COLOR, lw=LW, zorder=3,
            label=r"$\lambda_{\max}(\mathbf{H})$")
    ax.plot(diff, lam_min, color=LAMBDA_MIN_COLOR, lw=LW, zorder=3,
            label=r"$\lambda_{\min}(\mathbf{H})$")
    if legend:
        ax.legend(loc="best", fontsize=LEGEND_FS, framealpha=0.9, handlelength=1.5,
                  borderpad=0.3, labelspacing=0.25)

    ax.set_xlim(-np.pi, np.pi)
    _pi_axis(ax.xaxis, -np.pi, np.pi)
    ax.set_xlabel(r"$\theta_2-\theta_1$", fontsize=LABEL_FS)
    ax.set_ylabel(r"eigenvalues of $\mathbf{H}$", fontsize=LABEL_FS)
    ax.grid(alpha=0.3, linewidth=GRID_LW)
    ax.tick_params(labelsize=TICK_FS, width=0.5 * SCALE, length=1.5 * SCALE)
    return ax


def plot_not_pd_regions(
    kinematics: TwoLinkKinematics,
    sigma_bar: Optional[float] = None,
    *,
    paths: Optional[Mapping[str, object]] = None,
    window=DEFAULT_WINDOW,
    outdir: Optional[Path] = None,
    name: str = "fig_not_pd_regions",
    width: float = 6.9,
):
    """Figure of where ``H`` is not positive definite, for one ``sigma_bar``.

    Left: the ``(theta1, theta2)`` plane (:func:`plot_joint_space`). Right: the
    eigenvalues of ``H`` against ``theta2 - theta1`` (:func:`plot_eigenvalues`).

    ``kinematics`` gives the link lengths. ``sigma_bar`` may be any value
    ``>= 0``; when it is omitted, ``kinematics`` must be a :class:`TwoLinkArm`
    and its own threshold is used.

    ``paths`` maps keys to configurations of shape ``(N, 2)`` or to
    :class:`~odmcbf.simulate.Trace` objects, drawn on the left panel with
    :func:`plot_joint_path`. An array is labelled by its key; a trace by its
    own ``label``. A key or trace name among the controllers of
    :data:`~odmcbf.simulate.CONTROLLERS` gets that controller's colour, so the
    dict returned by :func:`~odmcbf.simulate.run_comparison` can be passed as
    it is; other paths get colours no controller uses. Later paths are drawn
    thinner, on top of earlier ones, and the last one dash-dotted, so that
    paths which run together all stay visible.

    ``width`` is in inches at ``figures.SCALE`` times the printed size, like
    the other figures; the fonts do not shrink with it, so below about 5 the
    labels crowd. The figure is written to ``outdir/name.png`` when ``outdir``
    is given, and returned.
    """
    if sigma_bar is None:
        if not isinstance(kinematics, TwoLinkArm):
            raise ValueError("pass sigma_bar, or a TwoLinkArm whose sigma_bar is used")
        sigma_bar = kinematics.sigma_bar
    region = NotPDRegion(kinematics, sigma_bar)
    window = _check_window(window)
    overlays = []
    for key, path in (paths or {}).items():
        theta = np.atleast_2d(np.asarray(getattr(path, "theta", path), dtype=float))
        if theta.ndim != 2 or theta.shape[1] != 2 or len(theta) == 0 or not np.all(np.isfinite(theta)):
            raise ValueError(f"path {key!r} must be finite with shape (N, 2); got {np.shape(theta)}")
        name_ = getattr(path, "name", None)
        color = COLORS.get(key, COLORS.get(name_) if isinstance(name_, str) else None)
        if color is None:
            fallback = sum(1 for *_, c, own in overlays if not own)
            color = PATH_FALLBACK_COLORS[fallback % len(PATH_FALLBACK_COLORS)]
            overlays.append((getattr(path, "label", key), theta, color, False))
        else:
            overlays.append((getattr(path, "label", key), theta, color, True))

    apply_style()
    fig, (ax_joint, ax_eig) = plt.subplots(
        1, 2, figsize=(width, 0.6 * width), layout="constrained",
        gridspec_kw=dict(width_ratios=(1.0, 1.15)))
    try:
        plot_joint_space(region, ax_joint, window=window)
        plot_eigenvalues(region, ax_eig, legend=False)   # the curves go in the figure legend

        handles, labels = _legend_entries(region)
        for curve in ax_eig.get_lines():
            if not curve.get_label().startswith("_"):
                handles.append(Line2D([], [], color=curve.get_color(), lw=LW))
                labels.append(curve.get_label())
        wide, thin = PATH_WIDTHS
        for i, (label, theta, color, _) in enumerate(overlays):
            lw = LW * (wide - (wide - thin) * i / max(len(overlays) - 1, 1))
            ls = TOP_PATH_DASHES if len(overlays) > 1 and i == len(overlays) - 1 else "-"
            (line, *_) = plot_joint_path(ax_joint, theta, window=window, color=color,
                                         linewidth=lw, linestyle=ls)
            marker = line.get_marker()
            if marker not in (None, "None", ""):
                handles.append(Line2D([], [], color=line.get_color(), linestyle="none",
                                      marker=marker, markersize=line.get_markersize(),
                                      markeredgecolor=line.get_markeredgecolor()))
            else:
                handles.append(Line2D([], [], color=line.get_color(), lw=lw, linestyle=ls))
            labels.append(str(label))

        fig.legend(handles, labels, loc="outside lower center",
                   ncols=min(len(labels), 4), fontsize=LEGEND_FS, frameon=False,
                   handlelength=1.8, columnspacing=1.2)
        fig.suptitle(rf"$l_1={region.l1:g}$,  $l_2={region.l2:g}$,  "
                     rf"$\bar\sigma={region.sigma_bar:.4g}$", fontsize=LABEL_FS)
        save(fig, outdir, name)
    finally:
        plt.close(fig)      # pyplot lets go of it; the caller gets the figure
    return fig


# ------------------------------------------------------------------ helpers
def _wrap(angle):
    """Wrap to ``(-pi, pi]``, leaving values already there untouched (and unrounded)."""
    angle = np.asarray(angle, dtype=float)
    inside = (angle > -np.pi) & (angle <= np.pi)
    return np.where(inside, angle, np.pi - np.mod(np.pi - angle, TWO_PI))


def _check_window(window):
    (x0, x1), (y0, y1) = window
    x0, x1, y0, y1 = (float(v) for v in (x0, x1, y0, y1))
    if not all(np.isfinite((x0, x1, y0, y1))) or x0 >= x1 or y0 >= y1:
        raise ValueError(f"window must be ((theta1_lo, theta1_hi), (theta2_lo, theta2_hi)) "
                         f"with lo < hi; got {window}")
    return (x0, x1), (y0, y1)


def _wrapped_polyline(theta: np.ndarray, origin) -> tuple[np.ndarray, np.ndarray]:
    """Wrap a path into ``[x0, x0 + 2 pi) x [y0, y0 + 2 pi)``, breaking it at each crossing.

    Each crossing of a cell edge ``x0 + 2 pi k`` or ``y0 + 2 pi k`` is located
    on the straight segment between samples; the path is drawn up to the edge,
    broken with ``NaN``, and resumed from the opposite edge. Between samples
    more than one period apart the path is only broken.
    """
    origin = np.asarray(origin, dtype=float)
    cells = np.floor((theta - origin) / TWO_PI).astype(int)
    points = [theta[0] - TWO_PI * cells[0]]
    for n in range(1, len(theta)):
        p, q = theta[n - 1], theta[n]
        if np.array_equal(cells[n], cells[n - 1]):
            points.append(q - TWO_PI * cells[n])
            continue
        if np.abs(cells[n] - cells[n - 1]).max() > 1:     # jumps whole periods: just break
            points.append(np.full(2, np.nan))
            points.append(q - TWO_PI * cells[n])
            continue
        crossings = []
        for axis in (0, 1):
            c0, c1 = cells[n - 1, axis], cells[n, axis]
            step = 1 if c1 > c0 else -1
            for c in range(c0, c1, step):
                edge = origin[axis] + TWO_PI * (c + (step > 0))
                crossings.append(((edge - p[axis]) / (q[axis] - p[axis]), axis, step))
        cell = cells[n - 1].copy()
        for t, axis, step in sorted(crossings):
            point = p + t * (q - p)
            points.append(point - TWO_PI * cell)
            points.append(np.full(2, np.nan))
            cell[axis] += step
            points.append(point - TWO_PI * cell)
        points.append(q - TWO_PI * cells[n])
    points = np.array(points)
    return points[:, 0], points[:, 1]


def _shifts(lo: float, hi: float, window) -> range:
    """Integers ``k`` for which ``[lo, hi] + 2 pi k`` meets ``theta2 - theta1`` on the window."""
    (x0, x1), (y0, y1) = window
    dmin, dmax = y0 - x1, y1 - x0
    return range(int(np.floor((dmin - hi) / TWO_PI)), int(np.ceil((dmax - lo) / TWO_PI)) + 1)


def _strip_polygons(lo: float, hi: float, window) -> list[np.ndarray]:
    """``{lo <= theta2 - theta1 <= hi}`` mod ``2 pi``, clipped to the window, as polygons."""
    (x0, x1), (y0, y1) = window
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    polys = []
    for k in _shifts(lo, hi, window):
        a, b = lo + TWO_PI * k, hi + TWO_PI * k
        poly = _clip(rect, lambda p, a=a: (p[1] - p[0]) - a)     # theta2 - theta1 >= a
        poly = _clip(poly, lambda p, b=b: b - (p[1] - p[0]))     # theta2 - theta1 <= b
        if len(poly) >= 3 and _area(poly) > 0.0:
            polys.append(poly)
    return polys


def _diagonal_segments(value: float, window) -> list[tuple[np.ndarray, np.ndarray]]:
    """The lines ``theta2 - theta1 = value`` mod ``2 pi`` inside the window, as ``(x, y)`` pairs."""
    (x0, x1), (y0, y1) = window
    segments = []
    for k in _shifts(value, value, window):
        c = value + TWO_PI * k
        start, stop = max(x0, y0 - c), min(x1, y1 - c)
        if start < stop:
            xs = np.array([start, stop])
            segments.append((xs, xs + c))
    return segments


def _clip(poly: np.ndarray, inside) -> np.ndarray:
    """Sutherland-Hodgman: keep the part of a convex polygon where ``inside(p) >= 0``."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = inside(p), inside(q)
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            out.append(p + (q - p) * (fp / (fp - fq)))
    return np.array(out).reshape(-1, 2)


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _pi_step(span: float) -> float:
    """Tick step for an axis ``span`` long: ``pi/2``, coarser by ``pi/2`` per extra period."""
    return np.pi / 2 * max(1, math.ceil(span / TWO_PI - 1e-9))


def _pi_ticks(lo: float, hi: float, step: float) -> int:
    return math.floor(hi / step + 1e-9) - math.ceil(lo / step - 1e-9) + 1


def _pi_axis(axis, lo: float, hi: float, step: Optional[float] = None) -> None:
    """Ticks at multiples of ``step`` (a multiple of ``pi/2``), labelled as fractions of ``pi``.

    The step shrinks by ``pi/2`` until the axis holds two ticks; an axis too
    short to hold two multiples of ``pi/2`` keeps plain decimal ticks.
    """
    step = _pi_step(hi - lo) if step is None else step
    while step > np.pi / 2 + 1e-12 and _pi_ticks(lo, hi, step) < 2:
        step -= np.pi / 2
    if _pi_ticks(lo, hi, step) >= 2:
        axis.set_major_locator(MultipleLocator(step))
        axis.set_major_formatter(FuncFormatter(_pi_label))
    else:
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())


def _pi_label(value, _pos=None) -> str:
    halves = int(round(value / (np.pi / 2)))
    if halves == 0:
        return "$0$"
    sign = "-" if halves < 0 else ""
    n = abs(halves)
    if n % 2 == 0:
        whole = n // 2
        return f"${sign}\\pi$" if whole == 1 else f"${sign}{whole}\\pi$"
    return f"${sign}\\pi/2$" if n == 1 else f"${sign}{n}\\pi/2$"


def _legend_entries(region: NotPDRegion):
    handles, labels = [], []
    if region.singular_only:
        handles.append(Line2D([], [], **DEGENERATE_STYLE))
        labels.append(r"$\lambda_{\min}(\mathbf{H})\leq 0$ ($J$ singular)")
        return handles, labels
    handles.append(Patch(**NOT_PD_STYLE))
    labels.append(r"$\lambda_{\min}(\mathbf{H})\leq 0$")
    if region.boundary_theta_diffs:
        handles.append(Line2D([], [], **BOUNDARY_STYLE))
        labels.append(r"$\lambda_{\min}(\mathbf{H})=0$")
    handles.append(Line2D([], [], **SINGULAR_STYLE))
    labels.append(r"$J$ singular")
    return handles, labels
