"""Unit tests of odmcbf.regions: where H = J J^T - sigma_bar^2 I is not positive definite."""

from __future__ import annotations

import matplotlib.pyplot as plt
import mpmath
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, Rectangle

from odmcbf import DEFAULT_BOUNDARY, TwoLinkArm, TwoLinkKinematics
from odmcbf import regions
from odmcbf.figures import COLORS
from odmcbf.regions import NotPDRegion
from odmcbf.two_link import band_half_width

#: ``(l1, l2, sigma_bar)`` covering every regime: zero threshold, a tiny one,
#: interior thresholds with equal and unequal links, exactly ``min(l1, l2)``,
#: and above it.
REGIMES = [
    (1.0, 1.0, 0.0),
    (1.3, 0.8, 0.0),
    (1.0, 1.0, 1e-9),
    (1.0, 1.0, 0.3),
    (1.0, 0.6, 0.3),
    (0.5, 1.2, 0.35),
    (1.5, 0.8, 0.78),
    (1.0, 1.0, 1.0),
    (1.3, 0.8, 0.8),
    (1.3, 0.8, 1.0),
    (0.7, 1.9, 5.0),
]


def _id(params) -> str:
    return "l1={:g}-l2={:g}-sb={:g}".format(*params)


@pytest.fixture(params=REGIMES, ids=_id)
def region(request) -> NotPDRegion:
    l1, l2, sigma_bar = request.param
    return NotPDRegion(TwoLinkKinematics(l1, l2), sigma_bar)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _brute_min_eigenvalue(region: NotPDRegion, theta) -> np.ndarray:
    """``lambda_min(H)`` from ``eigvalsh`` of ``M(theta) - sigma_bar^2 I``, one configuration at a time."""
    kin = region.kinematics
    shift = region.sigma_bar ** 2 * np.eye(2)
    return np.array([np.linalg.eigvalsh(kin.M(t) - shift)[0] for t in theta])


def _render(fig) -> np.ndarray:
    """Draw ``fig`` with Agg (it is closed, so it has no canvas of its own) and return RGB in [0, 1]."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].astype(float) / 255


def _exact_half_width(l1: float, l2: float, sigma_bar: float) -> float:
    """``asin`` of ``sin(w)**2 = sigma_bar**2 (l1**2 + l2**2 - sigma_bar**2) / (l1**2 l2**2)``, at 50 digits."""
    with mpmath.workdps(50):
        l1, l2, s = mpmath.mpf(l1), mpmath.mpf(l2), mpmath.mpf(sigma_bar)
        return float(mpmath.asin(mpmath.sqrt(s ** 2 * (l1 ** 2 + l2 ** 2 - s ** 2) / (l1 ** 2 * l2 ** 2))))


# ----------------------------------------------------------------- the set
def test_contains_matches_brute_force_eigenvalues(region, rng):
    theta = rng.uniform(-3 * np.pi, 3 * np.pi, size=(4000, 2))
    lam = _brute_min_eigenvalue(region, theta)
    clear = np.abs(lam) > 1e-9                     # rounding decides points on the boundary
    np.testing.assert_array_equal(region.contains(theta)[clear], (lam <= 0)[clear])


def test_eigenvalues_match_brute_force(region, rng):
    theta = rng.uniform(-np.pi, np.pi, size=(200, 2))
    lam_min, lam_max = region.eigenvalues(theta[:, 1] - theta[:, 0])
    shift = region.sigma_bar ** 2 * np.eye(2)
    brute = np.array([np.linalg.eigvalsh(region.kinematics.M(t) - shift) for t in theta])
    np.testing.assert_allclose(lam_min, brute[:, 0], atol=1e-12)
    np.testing.assert_allclose(lam_max, brute[:, 1], atol=1e-12)


def test_boundary_values_have_zero_min_eigenvalue(region):
    for value in region.boundary_theta_diffs:
        assert -np.pi < value <= np.pi
        assert abs(region.min_eigenvalue(value)) <= 1e-12 * max(1.0, region.sigma_bar ** 2)
    low = min(region.l1, region.l2)
    expected = 0 if region.sigma_bar > low else 2 if region.sigma_bar in (0.0, low) else 4
    assert len(region.boundary_theta_diffs) == expected


def test_boundary_configurations_belong_to_the_set(region):
    """The set is closed: every boundary value, and every singular line, is inside."""
    for value in region.boundary_theta_diffs:
        assert region.contains([0.0, value])
    for value in (0.0, np.pi, -np.pi, 2 * np.pi):
        assert region.contains([0.0, value])       # J is singular there


def test_half_width_and_intervals(region):
    w = region.half_width
    assert 0.0 <= w <= np.pi / 2
    lengths = sum(hi - lo for lo, hi in region.theta_diff_intervals)
    if region.everywhere:
        assert w == np.pi / 2
        assert lengths == pytest.approx(2 * np.pi)
    else:
        assert lengths == pytest.approx(4 * w)
        if region.singular_only:
            assert w == 0.0
        else:
            assert 0.0 < w < np.pi / 2
            assert abs(region.min_eigenvalue(w)) <= 1e-12 * max(1.0, region.sigma_bar ** 2)
            assert region.min_eigenvalue(np.pi / 2) > 0      # the middle of the safe band


@pytest.mark.parametrize("l1, l2, sigma_bar", [
    (1.0, 1.0, 1e-300),
    (1.0, 1.0, 1e-170),
    (1.0, 1.0, 1e-9),
    (1e3, 1.0, 1e-8),
    (1e6, 1e6, 1e-3),
    (1.0, 1.0, 1e-4),
    (1.3, 0.8, 0.5),
    (1.3, 0.8, float(np.nextafter(0.8, 0.0))),
    (0.5, 1.2, 0.35),
])
def test_half_width_has_full_relative_precision(l1, l2, sigma_bar):
    exact = _exact_half_width(l1, l2, sigma_bar)
    assert band_half_width(l1, l2, sigma_bar) == pytest.approx(exact, rel=1e-13)
    assert NotPDRegion(TwoLinkKinematics(l1, l2), sigma_bar).half_width == pytest.approx(exact, rel=1e-13)
    if sigma_bar < min(l1, l2):
        lo, hi = TwoLinkArm(l1, l2, sigma_bar=sigma_bar).safe_theta_diff_interval
        assert lo == pytest.approx(exact, rel=1e-13)


def test_band_half_width_covers_the_plane_at_and_above_min():
    for sigma_bar in (0.8, 1.0, 1.3, 5.0):
        assert band_half_width(1.3, 0.8, sigma_bar) == np.pi / 2


def test_tiny_sigma_bar_keeps_its_bands():
    """A tiny threshold gives thin bands, not the sigma_bar = 0 lines."""
    region = NotPDRegion(TwoLinkKinematics(1e6, 1e6), 1e-3)
    assert not region.singular_only and len(region.boundary_theta_diffs) == 4
    assert region.contains([0.0, 7.07e-10])             # lambda_min = -0.75 sigma_bar**2 there
    assert region.min_eigenvalue(7.07e-10) < 0


def test_regime_flags():
    kin = TwoLinkKinematics(1.3, 0.8)
    assert NotPDRegion(kin, 0.0).singular_only
    assert not NotPDRegion(kin, 1e-12).singular_only
    assert not NotPDRegion(kin, 0.0).everywhere
    assert not NotPDRegion(kin, 0.7999).everywhere
    assert NotPDRegion(kin, 0.8).everywhere
    assert NotPDRegion(kin, 0.8).boundary_theta_diffs == [-np.pi / 2, np.pi / 2]
    assert NotPDRegion(kin, 0.9).boundary_theta_diffs == []


def test_huge_sigma_bar_does_not_overflow():
    region = NotPDRegion(TwoLinkKinematics(), 1e200)
    assert region.everywhere and region.contains([[0.0, 1.0], [2.0, -3.0]]).all()
    lam_min, lam_max = region.eigenvalues(np.array([0.0, 1.0]))
    assert np.all(lam_min < 0) and np.all(lam_max < 0)
    _render(regions.plot_not_pd_regions(TwoLinkKinematics(), 1e200))


def test_arm_default_region_matches_safe_interval(arm):
    region = NotPDRegion.from_arm(arm)
    lo, hi = arm.safe_theta_diff_interval
    assert region.sigma_bar == arm.sigma_bar
    assert region.half_width == pytest.approx(lo, abs=1e-12)
    assert np.pi - region.half_width == pytest.approx(hi, abs=1e-12)
    if arm.sigma_bar == TwoLinkArm(arm.l1, arm.l2).sigma_bar:
        assert region.half_width == pytest.approx(DEFAULT_BOUNDARY)


# ------------------------------------------------------------ the geometry
@pytest.mark.parametrize("window", [
    regions.DEFAULT_WINDOW,
    ((0.3, 0.3 + 2 * np.pi), (-1.0, -1.0 + 2 * np.pi)),
])
def test_polygons_have_the_exact_area_over_one_period(region, window):
    if region.everywhere or region.singular_only:
        pytest.skip("drawn as a rectangle or as lines, not as bands")
    total = sum(regions._area(poly)
                for lo, hi in region._bands()
                for poly in regions._strip_polygons(lo, hi, window))
    # Over one period in each angle the set covers the fraction 4 w / (2 pi).
    assert total == pytest.approx(4 * region.half_width / (2 * np.pi) * (2 * np.pi) ** 2, rel=1e-9)


def test_polygons_agree_with_contains_on_any_window(region):
    if region.everywhere or region.singular_only or region.half_width < 1e-6:
        pytest.skip("drawn as a rectangle, as lines, or too thin for a grid")
    window = ((-0.4, 5.1), (1.0, 3.3))                       # not a whole period
    (x0, x1), (y0, y1) = window
    polys = [poly for lo, hi in region._bands() for poly in regions._strip_polygons(lo, hi, window)]
    for poly in polys:
        assert np.all(poly[:, 0] >= x0 - 1e-12) and np.all(poly[:, 0] <= x1 + 1e-12)
        assert np.all(poly[:, 1] >= y0 - 1e-12) and np.all(poly[:, 1] <= y1 + 1e-12)
        assert region.contains(poly.mean(axis=0))            # convex: the centroid is inside
    xs, ys = np.meshgrid(np.linspace(x0, x1, 400), np.linspace(y0, y1, 300))
    grid_fraction = region.contains(np.stack([xs, ys], axis=-1)).mean()
    area_fraction = sum(regions._area(p) for p in polys) / ((x1 - x0) * (y1 - y0))
    assert area_fraction == pytest.approx(grid_fraction, abs=0.01)


def test_diagonal_segments_lie_on_their_line():
    window = ((-np.pi, np.pi), (0.0, 2 * np.pi))
    for value in (0.0, np.pi, 1.1, -2.9):
        segments = regions._diagonal_segments(value, window)
        assert segments
        for xs, ys in segments:
            assert xs[0] < xs[1]
            assert np.all((xs >= -np.pi - 1e-12) & (xs <= np.pi + 1e-12))
            assert np.all((ys >= -1e-12) & (ys <= 2 * np.pi + 1e-12))
            offset = np.mod(ys - xs - value + np.pi, 2 * np.pi) - np.pi
            np.testing.assert_allclose(offset, 0.0, atol=1e-12)


# ---------------------------------------------------------------- drawing
def test_drawn_patches_are_the_set(region):
    """The joint panel's polygons lie in the set and the window; the eigenvalue panel's
    rectangles span exactly the intervals of theta2 - theta1."""
    fig = regions.plot_not_pd_regions(region.kinematics, region.sigma_bar)
    joint, eig = fig.axes[:2]
    (x0, x1), (y0, y1) = regions.DEFAULT_WINDOW
    polys = [p for p in joint.patches if isinstance(p, Polygon)]
    tol = 1e-9 * max(1.0, region.sigma_bar ** 2)
    if region.everywhere:
        assert any(isinstance(p, Rectangle) for p in joint.patches)
    elif region.singular_only:
        assert not polys and joint.lines
    else:
        assert polys
        for patch in polys:
            xy = patch.get_xy()
            assert np.all((xy[:, 0] >= x0 - 1e-9) & (xy[:, 0] <= x1 + 1e-9))
            assert np.all((xy[:, 1] >= y0 - 1e-9) & (xy[:, 1] <= y1 + 1e-9))
            # Every vertex lies on a band edge or on the window inside a band.
            assert np.all(region.min_eigenvalue(xy[:, 1] - xy[:, 0]) <= tol)
            if region.half_width > 1e-6:
                assert region.contains(xy[:-1].mean(axis=0))
    # A dashed line along every boundary value, in the joint panel.
    dashed = [line for line in joint.lines if line.get_linestyle() == "--"]
    for value in (region.boundary_theta_diffs if not region.singular_only else []):
        offsets = [np.mod(line.get_ydata()[0] - line.get_xdata()[0] - value + np.pi, 2 * np.pi) - np.pi
                   for line in dashed]
        assert min(abs(o) for o in offsets) < 1e-12
    if not region.singular_only:
        rects = [r for r in eig.patches if isinstance(r, Rectangle)]
        assert all(r.get_height() == 1 for r in rects)              # full height, in axes units
        spans = sorted((r.get_x(), r.get_x() + r.get_width()) for r in rects)
        expected = sorted(region.theta_diff_intervals)
        np.testing.assert_allclose(spans, expected, atol=1e-12)


@pytest.mark.parametrize("params", REGIMES, ids=_id)
def test_figure_renders_every_regime(params):
    l1, l2, sigma_bar = params
    fig = regions.plot_not_pd_regions(TwoLinkKinematics(l1, l2), sigma_bar)
    joint, eig = fig.axes[:2]
    assert joint.get_xlabel() == r"$\theta_1$" and eig.get_xlabel() == r"$\theta_2-\theta_1$"
    _render(fig)


def test_figure_with_closed_loops(comparison, tmp_path):
    arm = next(iter(comparison.values())).arm
    fig = regions.plot_not_pd_regions(arm, paths=comparison, outdir=tmp_path)
    assert (tmp_path / "fig_not_pd_regions.png").is_file()
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    for trace in comparison.values():
        assert trace.label in labels
    assert r"$\lambda_{\min}(\mathbf{H})$" in labels and r"$\lambda_{\max}(\mathbf{H})$" in labels
    assert fig.axes[1].get_legend() is None          # the curves are named in the figure legend
    joint = fig.axes[0]
    colored = {to_rgb(line.get_color()) for line in joint.lines}
    for key in comparison:
        assert to_rgb(COLORS[key]) in colored


def test_single_configuration_is_a_marker_in_the_legend():
    """A path of one configuration is drawn as a marker, and its legend entry shows that marker."""
    fig = regions.plot_not_pd_regions(TwoLinkArm(), paths={
        "start": [(0.3, 1.9)],
        "path": np.array([[0.0, 1.0], [0.5, 2.0]]),
    })
    legend = fig.legends[0]
    entries = dict(zip([t.get_text() for t in legend.get_texts()], legend.legend_handles))
    assert entries["start"].get_marker() == "o"
    assert entries["start"].get_linestyle() == "None"
    assert entries["path"].get_marker() in (None, "None", "")
    assert entries["path"].get_linestyle() != "None"


def test_eigenvalue_panel_legend():
    """On its own, the eigenvalue panel names both curves in a legend, unless ``legend=False``."""
    region = NotPDRegion(TwoLinkKinematics(), 0.5)
    _, ax = plt.subplots()
    regions.plot_eigenvalues(region, ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        r"$\lambda_{\max}(\mathbf{H})$", r"$\lambda_{\min}(\mathbf{H})$"]
    _, ax = plt.subplots()
    regions.plot_eigenvalues(region, ax, legend=False)
    assert ax.get_legend() is None


def test_repr():
    assert repr(NotPDRegion(TwoLinkKinematics(1, 0.6), 0.3)) == "NotPDRegion(l1=1, l2=0.6, sigma_bar=0.3)"
    assert repr(NotPDRegion.from_arm(TwoLinkArm())) == "NotPDRegion(l1=1, l2=1, sigma_bar=0.707107)"


def test_every_closed_loop_is_visible(comparison):
    """Each controller's colour shows in the rendered joint panel, including the scalar
    OD path, which runs within a few hundredths of a radian of the matrix OD path."""
    arm = next(iter(comparison.values())).arm
    fig = regions.plot_not_pd_regions(arm, paths=comparison)
    fig.set_dpi(100)
    image = _render(fig)
    joint = fig.axes[0]
    bbox = joint.get_window_extent()
    height = image.shape[0]
    crop = image[int(height - bbox.y1):int(height - bbox.y0), int(bbox.x0):int(bbox.x1)]
    for key in ("nominal", "scalar", "matrix"):       # standard shares the region's red
        target = np.array(to_rgb(COLORS[key]))
        close = np.linalg.norm(crop - target, axis=-1) < 0.12
        assert close.sum() >= 30, f"{key} path hardly visible: {close.sum()} pixels"


def test_matrix_od_path_runs_along_the_boundary(comparison):
    """The matrix OD closed loop ends on a boundary line of the region, the others do not cross it."""
    arm = next(iter(comparison.values())).arm
    region = NotPDRegion.from_arm(arm)
    final = comparison["matrix"].theta[-1]
    assert abs(region.min_eigenvalue(final[1] - final[0])) < 1e-3
    for key in ("standard", "scalar", "matrix"):
        inside = region.contains(comparison[key].theta)
        margins = np.array([arm.margin(t) for t in comparison[key].theta[inside]])
        assert np.all(margins > -1e-6)               # only boundary contact, never deep inside
    assert region.contains(comparison["nominal"].theta).any()


# ------------------------------------------------------------------ paths
def test_joint_path_runs_to_the_window_edge_where_it_wraps():
    fig, ax = plt.subplots()
    theta = np.column_stack([np.linspace(0.1, 2.4 * np.pi, 97), np.full(97, 0.5)])
    (line,) = regions.plot_joint_path(ax, theta)
    x = line.get_xdata()
    breaks = np.flatnonzero(np.isnan(x))
    assert len(breaks) == 1                          # crosses theta1 = pi once
    assert x[breaks[0] - 1] == pytest.approx(np.pi, abs=1e-12)
    assert x[breaks[0] + 1] == pytest.approx(-np.pi, abs=1e-12)
    finite = x[np.isfinite(x)]
    assert finite.min() >= -np.pi and finite.max() <= np.pi


def test_joint_path_crossing_two_edges_in_one_step():
    fig, ax = plt.subplots()
    (line,) = regions.plot_joint_path(ax, [[3.0, 3.1], [3.3, 3.2]], outline=False)
    x, y = line.get_xdata(), line.get_ydata()
    assert np.isnan(x).sum() == 2                    # through theta2 = pi, then theta1 = pi
    finite = np.isfinite(x)
    assert np.all(np.abs(x[finite]) <= np.pi + 1e-12) and np.all(np.abs(y[finite]) <= np.pi + 1e-12)


@pytest.mark.parametrize("axis, window", [
    (0, ((-2 * np.pi, 2 * np.pi), (-np.pi, np.pi))),
    (1, ((-np.pi, np.pi), (-2 * np.pi, 2 * np.pi))),
])
def test_joint_path_on_a_window_of_several_periods(axis, window):
    fig, ax = plt.subplots()
    theta = np.column_stack([np.linspace(0, 1, 5), np.linspace(0, 1, 5)])
    lines = regions.plot_joint_path(ax, theta, window=window)      # no colour given
    assert len(lines) == 2                           # one copy per period along that axis
    data = [line.get_xdata() if axis == 0 else line.get_ydata() for line in lines]
    offsets = sorted(np.nanmin(d) for d in data)
    assert offsets[1] - offsets[0] == pytest.approx(2 * np.pi)
    assert len({line.get_color() for line in lines}) == 1        # the copies share one colour


def test_joint_path_that_jumps_periods_is_only_broken():
    fig, ax = plt.subplots()
    (line,) = regions.plot_joint_path(ax, [[0.0, 0.5], [20.0, 20.5]], outline=False)
    x = line.get_xdata()
    assert np.isnan(x).sum() == 1 and np.isfinite(x).sum() == 2


def test_joint_path_refuses_angles_too_large_to_wrap():
    with pytest.raises(ValueError):
        regions.plot_joint_path(plt.gca(), [[0.0, 0.0], [1e13, 1e13]])


def test_joint_path_accepts_matplotlib_keywords():
    fig, ax = plt.subplots()
    theta = [[0.0, 1.0], [0.5, 1.2]]
    (line,) = regions.plot_joint_path(ax, theta, linewidth=5, zorder=7, color="tab:green")
    assert line.get_linewidth() == 5 and line.get_zorder() == 7
    (line,) = regions.plot_joint_path(ax, theta, lw=2, ls="--")
    assert line.get_linewidth() == 2 and line.get_linestyle() == "--"
    (point,) = regions.plot_joint_path(ax, [[0.3, 0.4]])
    assert point.get_marker() == "o"                 # a single configuration is still drawn
    (still,) = regions.plot_joint_path(ax, [[0.3, 0.4]] * 5)
    assert still.get_marker() == "o"                 # so is a path that does not move


def test_outline_stays_below_every_path():
    fig, ax = plt.subplots()
    regions.plot_joint_path(ax, [[0.0, 0.0], [1.0, 1.0]], color="tab:blue")
    regions.plot_joint_path(ax, [[0.0, 0.1], [1.0, 1.1]], color="tab:orange")
    white = [line.get_zorder() for line in ax.lines if line.get_color() == "white"]
    colored = [line.get_zorder() for line in ax.lines if line.get_color() != "white"]
    assert max(white) < min(colored)


# ------------------------------------------------------------------ axes
@pytest.mark.parametrize("window, step", [
    (((-np.pi, np.pi), (-np.pi, np.pi)), np.pi / 2),
    (((-2 * np.pi, 2 * np.pi), (-np.pi, np.pi)), np.pi),
    (((0.0, 6 * np.pi), (0.0, 2 * np.pi)), 3 * np.pi / 2),
])
def test_tick_spacing_follows_the_window(window, step):
    fig, ax = plt.subplots()
    regions.plot_joint_space(NotPDRegion(TwoLinkKinematics(), 0.5), ax, window=window)
    fig.canvas.draw()
    ticks = ax.xaxis.get_majorticklocs()
    assert np.allclose(np.diff(ticks), step)
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels and all(r"\pi" in s or s == "$0$" for s in labels)


def test_elongated_window_uses_one_tick_step():
    """The panel has equal aspect, so both axes share the step of the longer one."""
    fig, ax = plt.subplots()
    regions.plot_joint_space(NotPDRegion(TwoLinkKinematics(), 0.5), ax,
                             window=((0.0, 6 * np.pi), (0.0, 2 * np.pi)))
    x_step = np.diff(ax.xaxis.get_majorticklocs())
    y_step = np.diff(ax.yaxis.get_majorticklocs())
    assert np.allclose(x_step, 3 * np.pi / 2) and np.allclose(y_step, 3 * np.pi / 2)


def test_short_axis_steps_down_to_two_ticks():
    """Where the shared step would leave the short axis with one tick, it shrinks by ``pi/2`` at a time.

    The window is ten periods tall and one wide: the shared step ``5 pi/2``
    leaves one tick across ``(-pi, pi)``, and so do ``2 pi`` and ``3 pi/2``;
    ``pi`` gives three.
    """
    fig, ax = plt.subplots()
    regions.plot_joint_space(NotPDRegion(TwoLinkKinematics(), 0.5), ax,
                             window=((-np.pi, np.pi), (-5 * np.pi, 5 * np.pi)))
    fig.canvas.draw()
    assert np.allclose(np.diff(ax.xaxis.get_majorticklocs()), np.pi)
    assert np.allclose(np.diff(ax.yaxis.get_majorticklocs()), 5 * np.pi / 2)
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert {"$-\\pi$", "$0$", "$\\pi$"} <= set(labels)


def test_small_window_keeps_decimal_ticks():
    fig, ax = plt.subplots()
    regions.plot_joint_space(NotPDRegion(TwoLinkKinematics(), 0.5), ax,
                             window=((0.5, 1.5), (2.0, 2.4)))
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert len(labels) >= 2 and not any("pi" in s for s in labels)


# ------------------------------------------------------------- validation
def test_validation():
    kin = TwoLinkKinematics()
    for bad in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            NotPDRegion(kin, bad)
    with pytest.raises(ValueError):
        regions.plot_not_pd_regions(kin)             # no sigma_bar and no arm to take it from
    with pytest.raises(ValueError):
        NotPDRegion(kin, 0.5).contains([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        regions.plot_joint_path(plt.gca(), np.zeros((0, 2)))
    with pytest.raises(ValueError):
        regions.plot_joint_path(plt.gca(), [[0.0, np.nan]])


def test_overlay_styles_stay_apart_from_the_region(tmp_path):
    """Paths that do not name a controller get colours no controller uses, and the
    topmost path is dash-dotted, never dashed like the lambda_min(H) = 0 lines."""
    paths = {"a": [[0.0, 0.5], [1.0, 1.5]], "b": [[0.0, 0.6], [1.0, 1.6]]}
    fig = regions.plot_not_pd_regions(TwoLinkKinematics(), 0.5, paths=paths)
    joint = fig.axes[0]
    drawn = [line for line in joint.lines if line.get_zorder() == 3]        # the two paths
    boundary = [line for line in joint.lines if line.get_zorder() == 2]     # lambda_min(H) = 0
    assert len(drawn) == 2 and boundary
    colors = {to_rgb(line.get_color()) for line in drawn}
    assert not colors & {to_rgb(c) for c in COLORS.values()}
    # get_linestyle() calls every custom pattern '--'; compare the dash sequences.
    top_dashes = drawn[-1]._unscaled_dash_pattern[1]
    assert top_dashes is not None and len(top_dashes) == 4                   # dash-dot
    assert all(line._unscaled_dash_pattern[1] != top_dashes for line in boundary)


def test_figure_is_closed_when_saving_fails(tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("")
    plt.close("all")
    with pytest.raises(OSError):
        regions.plot_not_pd_regions(TwoLinkKinematics(), 0.5, outdir=blocker)
    assert plt.get_fignums() == []


def test_no_figure_is_left_open_on_bad_input():
    kin = TwoLinkKinematics()
    plt.close("all")
    with pytest.raises(ValueError):
        regions.plot_not_pd_regions(kin, 0.5, window=((1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError):
        regions.plot_not_pd_regions(kin, 0.5, paths={"bad": [[0.0, 1.0, 2.0]]})
    assert plt.get_fignums() == []


@pytest.mark.parametrize("halves, text", [
    (0, "$0$"), (1, r"$\pi/2$"), (-1, r"$-\pi/2$"), (2, r"$\pi$"),
    (-2, r"$-\pi$"), (3, r"$3\pi/2$"), (4, r"$2\pi$"),
])
def test_pi_tick_labels(halves, text):
    assert regions._pi_label(halves * np.pi / 2) == text
