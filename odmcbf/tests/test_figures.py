"""Unit tests of ``odmcbf.figures``: the workspace figure, ``margin_status``,
the closed-loop figures and the animation.

Angles are absolute, ``theta = (theta1, theta2)``; joint 1 is at the origin
and joint 2 at ``l1 (cos theta1, sin theta1)``. Every static figure is closed
before it is returned, which leaves it with a plain ``FigureCanvasBase``; the
pixel checks attach a ``FigureCanvasAgg`` to render it. The animation is
inspected through its public ``save`` method, with a writer that records the
state of every frame instead of encoding it.
"""

from __future__ import annotations

import dataclasses
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch, Wedge
from PIL import Image

from odmcbf import TwoLinkArm, rollout
from odmcbf import figures
from odmcbf.figures import (
    ANIMATE_MARGIN,
    CMAPS,
    COLORS,
    DRAW_SCALE,
    ORDER,
    SNAPSHOT_MARGIN,
    SNAPSHOT_SPACING,
    TITLES,
    WORKSPACE_COLUMNS,
    margin_status,
)

#: The closed-loop setup of the ``comparison`` fixture.
U_NOM = np.array([1.0, 0.4])
THETA0 = np.array([0.0, np.pi / 2])
DT = 0.04
HORIZON = 8.0

GREY = to_rgba("0.82")
RED = COLORS["standard"]
#: Arm pose for the pixel checks: both links in the upper half plane, so
#: nothing but the shading is drawn in the lower half.
UPPER_POSE = (np.pi / 2, np.pi / 2 + 1.0)
#: Directions, in the lower half plane, along which the shading is sampled.
LOWER_ANGLES = (-0.15, -np.pi / 4, -np.pi / 2, -3 * np.pi / 4, -np.pi + 0.15)
#: ``(l1, l2, sigma_bar)`` whose bands are at least 8 pixels wide at
#: ``panel_width=7.2``, including unequal links and very large and very small arms.
PIXEL_ARMS = [
    (1.0, 1.0, None),
    (1.0, 0.6, 0.3),
    (0.5, 1.2, 0.35),
    (1.5, 0.8, 0.78),
    (1000.0, 600.0, 300.0),
    (1e-6, 0.6e-6, 0.3e-6),
]


# ----------------------------------------------------------------- helpers
def _open_figures() -> set[int]:
    return set(plt.get_fignums())


def _suptitle(fig) -> str:
    """``fig.get_suptitle()``, which matplotlib added in 3.8; requirements.txt allows 3.7.3."""
    if hasattr(fig, "get_suptitle"):
        return fig.get_suptitle()
    return fig._suptitle.get_text()


def _render(fig) -> np.ndarray:
    """RGB pixels of ``fig``, drawn on a fresh Agg canvas."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].astype(int)


def _pixel(image: np.ndarray, ax, point) -> np.ndarray:
    """RGB value of the pixel under the data point ``point`` of ``ax``."""
    x, y = ax.transData.transform(point)
    return image[int(image.shape[0] - y), int(x)]


def _pixels_per_unit(ax, arm: TwoLinkArm) -> float:
    return (ax.transData.transform((arm.reach, 0.0))[0]
            - ax.transData.transform((0.0, 0.0))[0]) / arm.reach


def _is_red_tint(rgb) -> bool:
    red, green, blue = rgb
    return red - green >= 15 and red - blue >= 15


def _is_white(rgb) -> bool:
    return bool(np.all(np.asarray(rgb) == 255))


def _is_grey(rgb) -> bool:
    return bool(np.all(np.abs(np.asarray(rgb) - 209) <= 1))      # 0.82 * 255


def _arm_with_end_effector(arm: TwoLinkArm, point) -> np.ndarray:
    """A configuration ``theta`` whose end effector is at ``point``."""
    radius = np.hypot(*point)
    cos_diff = (radius ** 2 - arm.l1 ** 2 - arm.l2 ** 2) / (2 * arm.l1 * arm.l2)
    diff = np.arccos(np.clip(cos_diff, -1.0, 1.0))
    theta1 = (np.arctan2(point[1], point[0])
              - np.arctan2(arm.l2 * np.sin(diff), arm.l1 + arm.l2 * np.cos(diff)))
    theta = np.array([theta1, theta1 + diff])
    np.testing.assert_allclose(arm.end_effector(theta), point, atol=1e-9 * arm.reach)
    return theta


def _band_patches(ax):
    """Hatched wedges, as sorted ``(inner, outer)`` radii."""
    return sorted((w.r - w.width, w.r) for w in ax.patches
                  if isinstance(w, Wedge) and w.get_hatch() == "////")


def _circles(ax, linestyle: str):
    return [c for c in ax.patches
            if isinstance(c, Circle) and not c.get_fill() and c.get_linestyle() == linestyle]


def _patch_signature(patch) -> tuple:
    """Everything a workspace patch draws: shape, radii, colours, hatch, style, zorder."""
    radii = (patch.r, patch.width) if isinstance(patch, Wedge) else (patch.radius,)
    return (type(patch).__name__, tuple(patch.center), radii, patch.get_facecolor(),
            patch.get_edgecolor(), patch.get_hatch(), patch.get_linestyle(),
            patch.get_zorder())


def _boundary_configurations(arm: TwoLinkArm):
    """Safe, on-boundary and violated configurations, with their status."""
    lo, hi = arm.safe_theta_diff_interval
    return [
        ((0.3, 0.3 + np.pi / 2), "safe"),
        ((0.3, 0.3 + lo), "on boundary"),
        ((-1.0, -1.0 - hi), "on boundary"),
        ((0.3, 0.3 + lo / 2), "violated"),
        ((2.0, 2.0 + (hi + np.pi) / 2), "violated"),
    ]


def _nominal_traces(arm: TwoLinkArm, dt: float = 0.25, horizon: float = 2.0):
    """Four copies of an unfiltered run, standing in for the four controllers."""
    trace = rollout(arm, U_NOM, THETA0, dt=dt, horizon=horizon)
    return {key: trace for key in ORDER}


class _FrameRecorder(animation.AbstractMovieWriter):
    """Movie writer that records what every frame shows instead of encoding it."""

    def __init__(self):
        super().__init__(fps=5)
        self.frames = []

    def setup(self, fig, outfile, dpi=None):
        self.fig, self.outfile, self.dpi = fig, outfile, dpi

    def grab_frame(self, **savefig_kwargs):
        panels = []
        for ax in self.fig.axes:
            (ellipse,) = [p for p in ax.patches if type(p) is Ellipse]
            (circle,) = [p for p in ax.patches if isinstance(p, Circle)]
            panels.append(dict(
                arm=ax.lines[0].get_xydata().copy(),
                ellipse=(np.array(ellipse.get_center()), ellipse.width,
                         ellipse.height, ellipse.angle),
                circle=(np.array(circle.center), circle.radius),
                title=ax.get_title(),
                xlim=ax.get_xlim(),
                ylim=ax.get_ylim(),
            ))
        self.frames.append(panels)

    def finish(self):
        pass


def _record(ani, tmp_path) -> list:
    recorder = _FrameRecorder()
    ani.save(str(tmp_path / "unused"), writer=recorder)
    return recorder.frames


# ------------------------------------------------------ workspace: layout
def test_workspace_single_configuration(arm):
    """A ``(2,)`` theta gives one closed figure with one panel."""
    before = _open_figures()
    fig = figures.workspace(arm, (0.3, 1.7))
    assert isinstance(fig, Figure)
    assert not plt.fignum_exists(fig.number)
    assert _open_figures() == before
    assert len(fig.axes) == 1
    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (1, 1)


@pytest.mark.parametrize("count, nrows, ncols", [
    (1, 1, 1), (3, 1, 3), (4, 1, 4), (5, 2, 4), (6, 2, 4), (9, 3, 4), (12, 3, 4),
])
def test_workspace_grid(count, nrows, ncols):
    """``k`` configurations fill rows of ``WORKSPACE_COLUMNS``; unused axes are removed."""
    theta1 = np.linspace(-1.0, 1.0, count)
    thetas = np.column_stack([theta1, theta1 + 1.2])
    before = _open_figures()
    fig = figures.workspace(TwoLinkArm(), thetas, panel_width=2.0)
    assert not plt.fignum_exists(fig.number)
    assert _open_figures() == before
    assert len(fig.axes) == count
    assert fig.get_size_inches() == pytest.approx((2.0 * ncols, 2.0 * 1.3 * nrows))
    for index, (ax, (theta1, theta2)) in enumerate(zip(fig.axes, thetas)):
        spec = ax.get_subplotspec()
        assert spec.get_gridspec().get_geometry() == (nrows, ncols)
        row, col = divmod(index, WORKSPACE_COLUMNS)
        assert (spec.rowspan.start, spec.colspan.start) == (row, col)
        assert ax.get_xlabel() == "$x$"
        assert ax.get_ylabel() == ("$y$" if col == 0 else "")
        assert ax.get_title().startswith(f"$\\theta=({theta1:.2f},\\,{theta2:.2f})$\n")


def test_workspace_accepts_lists():
    fig = figures.workspace(TwoLinkArm(), [[0, 1], [1, 2]])
    assert len(fig.axes) == 2


@pytest.mark.parametrize("shape", [(3,), (1,), (4, 3), (1, 1), (2, 2, 2), (0, 2), (0,)])
def test_workspace_rejects_bad_shapes(shape):
    before = _open_figures()
    with pytest.raises(ValueError, match="shape"):
        figures.workspace(TwoLinkArm(), np.ones(shape))
    assert _open_figures() == before


@pytest.mark.parametrize("theta", [
    (np.nan, 0.0),
    (0.0, np.inf),
    (-np.inf, 1.0),
    [[0.0, 1.0], [1.0, np.nan]],
])
def test_workspace_rejects_non_finite_angles(theta):
    before = _open_figures()
    with pytest.raises(ValueError, match="finite"):
        figures.workspace(TwoLinkArm(), theta)
    assert _open_figures() == before


# ---------------------------------------------------- workspace: geometry
def test_workspace_outside_is_grey(arm):
    """A grey wedge covers ``r_out`` to past the axes corners, plus a disc of radius ``r_in``."""
    r_in, r_out = arm.workspace_radii
    lim = SNAPSHOT_MARGIN * arm.reach
    ax = figures.workspace(arm, (0.3, 1.7)).axes[0]

    (wedge,) = [p for p in ax.patches if isinstance(p, Wedge) and p.get_facecolor() == GREY]
    assert tuple(wedge.center) == (0.0, 0.0)
    assert (wedge.theta1, wedge.theta2) == (0, 360)
    assert wedge.r > math.hypot(lim, lim)
    assert wedge.r - wedge.width == pytest.approx(r_out, rel=1e-12)
    assert wedge.get_zorder() == 0

    discs = [p for p in ax.patches if isinstance(p, Circle) and p.get_fill()]
    if arm.l1 == arm.l2:
        assert r_in == 0.0 and discs == []
    else:
        (disc,) = discs
        assert tuple(disc.center) == (0.0, 0.0)
        assert disc.radius == r_in
        assert disc.get_facecolor() == GREY
        assert disc.get_zorder() == 0


def test_workspace_violated_bands(arm):
    """The hatched bands are ``r_hi..r_out`` and ``r_in..r_lo``, tinted red."""
    r_in, r_out = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    ax = figures.workspace(arm, (0.3, 1.7)).axes[0]

    bands = _band_patches(ax)
    assert len(bands) == 2
    np.testing.assert_allclose(bands, [(r_in, r_lo), (r_hi, r_out)], rtol=0, atol=1e-12 * r_out)
    for wedge in ax.patches:
        if isinstance(wedge, Wedge) and wedge.get_hatch() == "////":
            assert tuple(wedge.center) == (0.0, 0.0)
            assert (wedge.theta1, wedge.theta2) == (0, 360)
            assert wedge.get_facecolor() == pytest.approx(to_rgba(RED, 0.12))
            assert wedge.get_zorder() == 0


def test_workspace_circles(arm):
    """Dashed safe-set circles at ``r_lo, r_hi`` lie above the solid workspace circles."""
    r_in, r_out = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    ax = figures.workspace(arm, (0.3, 1.7)).axes[0]

    dashed, solid = _circles(ax, "--"), _circles(ax, "solid")
    assert sorted(c.radius for c in dashed) == [r_lo, r_hi]
    assert sorted(c.radius for c in solid) == ([r_out] if arm.l1 == arm.l2 else [r_in, r_out])
    for circle in dashed + solid:
        assert tuple(circle.center) == (0.0, 0.0)
    assert all(c.get_edgecolor() == to_rgba(RED) for c in dashed)
    assert all(c.get_edgecolor() == to_rgba("0.2") for c in solid)
    # Everything the circles are drawn over sits below them.
    assert min(c.get_zorder() for c in dashed) > max(c.get_zorder() for c in solid)
    filled = [p for p in ax.patches if p not in dashed + solid]
    assert max(p.get_zorder() for p in filled) < min(c.get_zorder() for c in solid)
    assert len(filled) == (3 if arm.l1 == arm.l2 else 4)


def test_workspace_skips_zero_width_bands():
    """A band whose radii coincide in floating point is not drawn."""
    arm = TwoLinkArm(1.0, 1.0, sigma_bar=1e-9)
    r_lo, r_hi = arm.safe_radius_interval
    assert r_hi == arm.workspace_radii[1] and r_lo > 0.0     # outer band rounds away
    ax = figures.workspace(arm, (0.3, 1.7)).axes[0]
    assert _band_patches(ax) == [(0.0, r_lo)]

    arm = TwoLinkArm(1.0, 0.5, sigma_bar=1e-9)
    (r_in, r_out), (r_lo, r_hi) = arm.workspace_radii, arm.safe_radius_interval
    assert (r_lo, r_hi) == (r_in, r_out)                        # both bands round away
    ax = figures.workspace(arm, (0.3, 1.7)).axes[0]
    assert _band_patches(ax) == []
    assert sorted(c.radius for c in _circles(ax, "--")) == [r_lo, r_hi]


def test_workspace_shades_every_panel_alike(arm):
    """Whether ``H >= 0`` depends on ``|p|`` alone, so every panel, on every row, has the
    same shading and circles as the first, which the tests above check."""
    thetas = [(0.3, 1.7), (0.0, 0.05), (2.0, 2.0 + np.pi), (-1.0, 1.0), (0.5, 0.5 + np.pi / 2)]
    fig = figures.workspace(arm, thetas)
    assert len(fig.axes) == len(thetas)
    first = [_patch_signature(p) for p in fig.axes[0].patches]
    assert len(first) == (6 if arm.l1 == arm.l2 else 8)
    for ax in fig.axes[1:]:
        assert [_patch_signature(p) for p in ax.patches] == first


def test_workspace_arm_and_limits(arm, rng):
    """Each panel draws ``arm.arm_points(theta)``; limits are ``SNAPSHOT_MARGIN * reach``."""
    thetas = rng.uniform(-3 * np.pi, 3 * np.pi, size=(3, 2))
    lim = SNAPSHOT_MARGIN * arm.reach
    fig = figures.workspace(arm, thetas)
    for ax, theta in zip(fig.axes, thetas):
        points = arm.arm_points(theta)
        link, joints, joint1 = ax.lines
        np.testing.assert_array_equal(link.get_xydata(), points)
        np.testing.assert_array_equal(joints.get_xydata(), points[1:])
        np.testing.assert_array_equal(joint1.get_xydata(), [[0.0, 0.0]])
        assert ax.get_xlim() == pytest.approx((-lim, lim), rel=1e-15)
        assert ax.get_ylim() == pytest.approx((-lim, lim), rel=1e-15)
        assert ax.get_aspect() == 1.0


# ------------------------------------------------------ workspace: pixels
@pytest.mark.parametrize("l1, l2, sigma_bar", PIXEL_ARMS)
def test_workspace_rendered_regions(l1, l2, sigma_bar):
    """Bands render red-tinted, the safe annulus white, the rest of the plane grey."""
    arm = TwoLinkArm(l1, l2, sigma_bar=sigma_bar)
    (r_in, r_out), (r_lo, r_hi) = arm.workspace_radii, arm.safe_radius_interval
    lim = SNAPSHOT_MARGIN * arm.reach
    fig = figures.workspace(arm, UPPER_POSE, panel_width=7.2)
    image = _render(fig)
    ax = fig.axes[0]
    scale = _pixels_per_unit(ax, arm)
    assert min(r_out - r_hi, r_lo - r_in, r_hi - r_lo) * scale >= 8

    for angle in LOWER_ANGLES:
        direction = np.array([np.cos(angle), np.sin(angle)])
        assert _is_red_tint(_pixel(image, ax, (r_hi + r_out) / 2 * direction))
        assert _is_red_tint(_pixel(image, ax, (r_in + r_lo) / 2 * direction))
        assert _is_white(_pixel(image, ax, (r_lo + r_hi) / 2 * direction))
        assert _is_grey(_pixel(image, ax, (r_out + lim) / 2 * direction))
        if r_in > 0:
            assert _is_grey(_pixel(image, ax, r_in / 2 * direction))
    for corner in ((-0.98 * lim, -0.98 * lim), (0.98 * lim, -0.98 * lim)):
        assert _is_grey(_pixel(image, ax, corner))


@pytest.mark.parametrize("l1, l2, sigma_bar", PIXEL_ARMS)
def test_workspace_shading_matches_margin_status(l1, l2, sigma_bar):
    """On a grid, a point is grey outside the workspace, else red iff ``margin_status`` says violated.

    Points within 4 pixels of a circle are left out; inside the workspace the
    status is that of a configuration reaching the point.
    """
    arm = TwoLinkArm(l1, l2, sigma_bar=sigma_bar)
    r_in, r_out = arm.workspace_radii
    radii = np.array([r_in, *arm.safe_radius_interval, r_out])
    lim = SNAPSHOT_MARGIN * arm.reach
    fig = figures.workspace(arm, UPPER_POSE, panel_width=7.2)
    image = _render(fig)
    ax = fig.axes[0]
    scale = _pixels_per_unit(ax, arm)

    counts = {"grey": 0, "violated": 0, "safe": 0}
    for x in np.linspace(-0.95 * lim, 0.95 * lim, 31):
        for y in np.linspace(-0.95 * lim, -0.1 * lim, 12):
            radius = math.hypot(x, y)
            if np.min(np.abs(radii - radius)) * scale < 4:
                continue
            rgb = _pixel(image, ax, (x, y))
            if radius > r_out or radius < r_in:
                assert _is_grey(rgb), (x, y, rgb)
                counts["grey"] += 1
                continue
            status = margin_status(arm, _arm_with_end_effector(arm, (x, y)))[1]
            if status == "violated":
                assert _is_red_tint(rgb), (x, y, rgb)
            else:
                assert status == "safe" and _is_white(rgb), (x, y, status, rgb)
            counts[status] += 1
    assert min(counts.values()) >= 10, counts


def test_workspace_dashed_boundary_shows_over_workspace_circle():
    """With ``sigma_bar`` small, the dashed ``r_hi`` circle is visible on top of ``r_out``."""
    arm = TwoLinkArm(1.0, 1.0, sigma_bar=0.1)
    r_hi, r_out = arm.safe_radius_interval[1], arm.workspace_radii[1]
    fig = figures.workspace(arm, UPPER_POSE, panel_width=7.2)
    image = _render(fig)
    ax = fig.axes[0]
    assert (r_out - r_hi) * _pixels_per_unit(ax, arm) < 1       # the lines overlap
    radius = (r_hi + r_out) / 2
    pixels = np.array([_pixel(image, ax, (radius * np.cos(a), radius * np.sin(a)))
                       for a in np.linspace(-np.pi + 0.2, -0.2, 200)])
    dash = pixels[:, 0] - pixels[:, 1] > 60
    assert 0.5 < dash.mean() < 0.95          # red dashes, dark gaps between them


# ---------------------------------------------------------- margin_status
def test_margin_status_away_from_boundary(arm, rng):
    """Off the boundary the margin is ``arm.margin(theta)`` itself, labelled by its sign."""
    tolerance = 1e-12 * (arm.l1 ** 2 + arm.l2 ** 2)
    seen = set()
    for theta in rng.uniform(-3 * np.pi, 3 * np.pi, size=(200, 2)):
        margin, status = margin_status(arm, theta)
        expected = arm.margin(theta)
        assert abs(expected) > tolerance
        assert margin == pytest.approx(expected, rel=0, abs=1e-2 * tolerance)   # rounding only
        assert status == ("safe" if expected > 0 else "violated")
        seen.add(status)
    assert seen == {"safe", "violated"}


def test_margin_status_on_boundary(arm, rng):
    """At ``|theta2 - theta1| = lo, hi`` the margin is exactly ``0.0`` and on the boundary."""
    lo, hi = arm.safe_theta_diff_interval
    for theta1 in rng.uniform(-10, 10, size=25):
        for diff in (lo, hi, -lo, -hi, lo - 2 * np.pi):
            margin, status = margin_status(arm, (theta1, theta1 + diff))
            assert status == "on boundary"
            assert type(margin) is float
            assert margin == 0.0 and math.copysign(1.0, margin) == 1.0


@pytest.mark.parametrize("factor, status", [
    (0.0, "on boundary"),
    (-0.0, "on boundary"),
    (0.999, "on boundary"),
    (-0.999, "on boundary"),
    (1.0, "on boundary"),
    (-1.0, "on boundary"),
    (1.001, "safe"),
    (-1.001, "violated"),
    (1e3, "safe"),
])
def test_margin_status_tolerance(arm, monkeypatch, factor, status):
    """The boundary tolerance is ``1e-12 (l1**2 + l2**2)``, inclusive."""
    value = factor * 1e-12 * (arm.l1 ** 2 + arm.l2 ** 2)
    monkeypatch.setattr(arm, "margin", lambda theta: value)
    margin, label = margin_status(arm, (0.0, 1.0))
    assert label == status
    if status == "on boundary":
        assert margin == 0.0 and math.copysign(1.0, margin) == 1.0
    else:
        assert margin == value


@pytest.mark.parametrize("length", [1000.0, 1.0, 1e-6])
@pytest.mark.parametrize("theta, status", [
    ((0.0, np.pi / 2), "safe"),
    ((0.0, np.pi / 3), "on boundary"),
    ((0.0, 2 * np.pi / 3), "on boundary"),
    ((0.3, 0.3 - np.pi / 3), "on boundary"),
    ((5.0, 5.0 + np.pi / 3), "on boundary"),
    ((0.0, np.pi / 3 + 1e-13), "on boundary"),
    ((0.0, np.pi / 3 + 1e-6), "safe"),
    ((0.0, np.pi / 3 - 1e-6), "violated"),
    ((0.0, 0.2), "violated"),
    ((0.0, np.pi - 0.2), "violated"),
])
def test_margin_status_scales_with_the_arm(length, theta, status):
    """The tolerance is relative: at ``pi/3 + 1e-6`` the tiny arm is safe with margin ~1e-18,
    and at ``pi/3 + 1e-13`` the large arm is on the boundary with margin ~1e-7."""
    arm = TwoLinkArm(length, length)
    margin, label = margin_status(arm, theta)
    assert label == status
    if status == "on boundary":
        assert margin == 0.0
    else:
        assert margin == pytest.approx(arm.margin(theta), rel=0, abs=1e-14 * 2 * length ** 2)


# ------------------------------------------------ workspace: text, legend
def test_workspace_titles(arm):
    """Titles show theta, the margin with ``+.3g`` and the status from ``margin_status``."""
    cases = _boundary_configurations(arm)
    fig = figures.workspace(arm, [theta for theta, _ in cases])
    for ax, (theta, status) in zip(fig.axes, cases):
        margin, label = margin_status(arm, theta)
        assert label == status
        title = ax.get_title()
        first, second = title.split("\n")
        assert first == f"$\\theta=({theta[0]:.2f},\\,{theta[1]:.2f})$"
        assert second == f"$\\lambda_{{\\min}}(\\mathbf{{H}})={margin:+.3g}$  ({status})"
        if status == "on boundary":
            assert "=+0$" in second


@pytest.mark.parametrize("length", [1000.0, 1e-6])
def test_workspace_titles_at_any_scale(length):
    arm = TwoLinkArm(length, length)
    fig = figures.workspace(arm, [(0.0, np.pi / 2), (0.0, np.pi / 3), (0.0, 0.2)])
    endings = [ax.get_title().rsplit("  ", 1)[1] for ax in fig.axes]
    assert endings == ["(safe)", "(on boundary)", "(violated)"]
    assert "=+0$" in fig.axes[1].get_title()


def test_workspace_suptitle(arm):
    """The suptitle gives ``l1``, ``l2`` and ``sigma_bar`` (four significant digits)."""
    text = _suptitle(figures.workspace(arm, (0.3, 1.7)))
    found = re.fullmatch(r"\$l_1=(.+)\$,  \$l_2=(.+)\$,  \$\\bar\\sigma=(.+)\$", text)
    assert found is not None, text
    l1, l2, sigma_bar = map(float, found.groups())
    assert (l1, l2) == (arm.l1, arm.l2)
    assert sigma_bar == pytest.approx(arm.sigma_bar, rel=5e-4)
    assert found.group(3) == f"{arm.sigma_bar:.4g}"


@pytest.mark.parametrize("length", [1000.0, 1e-6])
def test_workspace_suptitle_at_any_scale(length):
    text = _suptitle(figures.workspace(TwoLinkArm(length, length), (0.3, 1.7)))
    assert f"$l_1={length:g}$" in text and f"$l_2={length:g}$" in text


def test_workspace_legend():
    fig = figures.workspace(TwoLinkArm(), [(0.3, 1.7), (0.0, 0.5)])
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "workspace boundary", "outside the workspace",
        r"$\lambda_{\min}(\mathbf{H})=0$", r"$\lambda_{\min}(\mathbf{H})<0$",
    ]
    reach, outside, boundary, violated = legend.legend_handles
    assert isinstance(reach, Line2D) and reach.get_linestyle() == "-"
    assert to_rgba(reach.get_color()) == to_rgba("0.2")
    assert isinstance(outside, Patch) and outside.get_facecolor() == GREY
    assert isinstance(boundary, Line2D) and boundary.get_linestyle() == "--"
    assert to_rgba(boundary.get_color()) == to_rgba(RED)
    assert isinstance(violated, Patch) and violated.get_hatch() == "////"
    assert violated.get_facecolor() == pytest.approx(to_rgba(RED, 0.12))


# ----------------------------------------------------------------- saving
@pytest.mark.parametrize("as_str", [False, True])
def test_workspace_writes_png(tmp_path, as_str):
    """``outdir`` receives ``name.png``, the directory being created."""
    outdir = tmp_path / "new" / "dir"
    before = _open_figures()
    figures.workspace(TwoLinkArm(), (0.3, 1.7), outdir=str(outdir) if as_str else outdir,
                      name="panel", panel_width=1.8)
    assert _open_figures() == before
    assert sorted(outdir.iterdir()) == [outdir / "panel.png"]
    assert (outdir / "panel.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_workspace_without_outdir_writes_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: calls.append(a))
    figures.workspace(TwoLinkArm(), (0.3, 1.7))
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_save(tmp_path, monkeypatch):
    """``save`` writes ``outdir/name.png``, creating ``outdir``; ``None`` writes nothing,
    not even into the working directory."""
    monkeypatch.chdir(tmp_path)
    fig = Figure(figsize=(0.5, 0.5))
    figures.save(fig, None, "tiny")
    assert list(tmp_path.iterdir()) == []
    figures.save(fig, tmp_path / "a" / "b", "tiny")
    assert sorted(tmp_path.rglob("*.*")) == [tmp_path / "a" / "b" / "tiny.png"]
    assert (tmp_path / "a" / "b" / "tiny.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("draw", ["snapshots", "closed_loop", "workspace"])
def test_static_figures_apply_the_style(comparison, draw):
    """Every static figure applies ``apply_style``: serif text, STIX math, TrueType fonts in PDFs."""
    make = {
        "snapshots": lambda: figures.snapshots(comparison),
        "closed_loop": lambda: figures.closed_loop(comparison, U_NOM),
        "workspace": lambda: figures.workspace(TwoLinkArm(), (0.3, 1.7)),
    }[draw]
    with plt.rc_context({"font.family": ["sans-serif"], "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 3}):
        fig = make()
        assert plt.rcParams["pdf.fonttype"] == 42
    for ax in fig.axes:
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            assert text.get_fontfamily() == ["serif"]
            assert text.get_math_fontfamily() == "stix"


def test_figures_pass_outdir_and_name_to_save(comparison, monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(figures, "save", lambda fig, outdir, name: saved.append((fig, outdir, name)))
    fig_a = figures.snapshots(comparison, outdir=tmp_path, name="a")
    fig_b = figures.closed_loop(comparison, U_NOM, outdir=tmp_path, name="b")
    fig_c = figures.workspace(TwoLinkArm(), (0.3, 1.7), outdir=tmp_path, name="c")
    fig_d = figures.snapshots(comparison)
    fig_e = figures.closed_loop(comparison, U_NOM)
    fig_f = figures.workspace(TwoLinkArm(), (0.3, 1.7))
    assert saved == [
        (fig_a, tmp_path, "a"), (fig_b, tmp_path, "b"), (fig_c, tmp_path, "c"),
        (fig_d, None, "fig_snapshots"), (fig_e, None, "fig_closed_loop"),
        (fig_f, None, "fig_workspace"),
    ]


# -------------------------------------------------------------- snapshots
def test_snapshot_spacing():
    """Every panel samples each second, which ``dt = 0.04`` hits exactly."""
    assert SNAPSHOT_SPACING == {key: 1.0 for key in ORDER}
    for spacing in SNAPSHOT_SPACING.values():
        assert round(spacing / DT) * DT == pytest.approx(spacing, rel=1e-12)


def test_snapshots(comparison):
    """One panel per controller, each with the arm, ellipse and circle every second."""
    arm = comparison["nominal"].arm
    lim = SNAPSHOT_MARGIN * arm.reach
    times = np.arange(0.0, HORIZON + 0.5, 1.0)
    before = _open_figures()
    fig = figures.snapshots(comparison)
    assert isinstance(fig, Figure)
    assert not plt.fignum_exists(fig.number)
    assert _open_figures() == before

    panels, bars = fig.axes[:4], fig.axes[4:]
    assert len(bars) == 4
    contains = {False: 0, True: 0}          # snapshots with the circle outside / inside
    for ax, key in zip(panels, ORDER):
        trace, cmap = comparison[key], plt.get_cmap(CMAPS[key])
        assert ax.get_title() == TITLES[key]
        assert ax.get_xlim() == pytest.approx((-lim, lim))
        assert ax.get_ylim() == pytest.approx((-lim, lim))
        ellipses = [p for p in ax.patches if type(p) is Ellipse]   # Circle subclasses Ellipse
        circles = [p for p in ax.patches if isinstance(p, Circle)]
        assert len(ellipses) == len(circles) == len(times)
        assert len(ax.lines) == len(times) + 1                   # and the joint-1 marker
        for line, ellipse, circle, t in zip(ax.lines, ellipses, circles, times):
            idx = round(t / DT)
            assert trace.t[idx] == pytest.approx(t)
            points = arm.arm_points(trace.theta[idx])
            np.testing.assert_array_equal(line.get_xydata(), points)
            assert to_rgba(line.get_color()) == cmap(0.30 + 0.62 * t / HORIZON)
            np.testing.assert_array_equal(ellipse.get_center(), points[-1])
            width, height, angle = figures._ellipse_axes(arm, trace.theta[idx])
            assert (ellipse.width, ellipse.height, ellipse.angle) == (width, height, angle)
            np.testing.assert_array_equal(circle.center, points[-1])
            assert circle.radius == DRAW_SCALE * arm.sigma_bar
            assert circle.get_linestyle() == "--"
            # The constraint holds exactly when the (concentric) ellipse contains the circle.
            assert abs(trace.margin[idx]) > 1e-6
            assert (ellipse.height / 2 >= circle.radius) == (trace.margin[idx] >= 0)
            contains[trace.margin[idx] >= 0] += 1
        np.testing.assert_array_equal(ax.lines[-1].get_xydata(), [[0.0, 0.0]])
    assert min(contains.values()) >= 3, contains
    for bar in bars:
        assert bar.get_xlim() == pytest.approx((0.0, HORIZON))
        np.testing.assert_array_equal(bar.get_xticks(), [0.0, HORIZON])


def test_snapshots_reject_dt_that_misses_the_spacing():
    traces = _nominal_traces(TwoLinkArm(), dt=0.3, horizon=3.0)
    before = _open_figures()
    try:
        with pytest.raises(ValueError, match="no sample at t = 1"):
            figures.snapshots(traces)
    finally:
        for number in _open_figures() - before:
            plt.close(number)


@pytest.mark.parametrize("trace_arm, explicit, drawn", [
    ((1.5, 0.8, 0.78), None, (1.5, 0.8, 0.78)),
    ((1.5, 0.8, 0.78), (0.5, 1.2, 0.35), (0.5, 1.2, 0.35)),
    (None, None, (1.0, 1.0, None)),
])
def test_snapshots_draw_the_traces_arm(trace_arm, explicit, drawn):
    """The drawn arm is the explicit one, else the traces' arm, else ``TwoLinkArm()``."""
    arms = {params: TwoLinkArm(params[0], params[1], sigma_bar=params[2])
            for params in {trace_arm, explicit, drawn} if params is not None}
    run_with = arms[trace_arm] if trace_arm else TwoLinkArm()
    traces = _nominal_traces(run_with)
    if trace_arm is None:
        traces = {key: dataclasses.replace(trace, arm=None) for key, trace in traces.items()}
    arm = arms[drawn]
    fig = figures.snapshots(traces, arm=arms[explicit] if explicit else None)
    for ax in fig.axes[:4]:
        assert ax.get_xlim() == pytest.approx((-SNAPSHOT_MARGIN * arm.reach,
                                               SNAPSHOT_MARGIN * arm.reach))
        circles = [p for p in ax.patches if isinstance(p, Circle)]
        assert {c.radius for c in circles} == {DRAW_SCALE * arm.sigma_bar}
        np.testing.assert_array_equal(ax.lines[1].get_xydata(),
                                      arm.arm_points(traces["nominal"].theta[4]))


# ------------------------------------------------------------ closed_loop
@pytest.mark.parametrize("omega_d", [1.0, 0.5])
def test_closed_loop(comparison, omega_d):
    """Margins, control deviations, decay deviations and the scalar lower bound."""
    before = _open_figures()
    fig = figures.closed_loop(comparison, list(U_NOM), omega_d=omega_d)
    assert isinstance(fig, Figure)
    assert not plt.fignum_exists(fig.number)
    assert _open_figures() == before

    margins, controls, decays = fig.axes
    for ax in fig.axes:
        assert ax.get_xlim() == (0.0, HORIZON)
    for line, key in zip(margins.lines, ORDER):
        trace = comparison[key]
        np.testing.assert_array_equal(line.get_xydata(), np.column_stack([trace.t, trace.margin]))
        assert line.get_label() == trace.label
        assert line.get_color() == COLORS[key]
    np.testing.assert_array_equal(margins.lines[4].get_ydata(), [0, 0])
    assert [t.get_text() for t in margins.get_legend().get_texts()] == [
        "nominal", "scalar OD", "standard", "matrix OD"]

    assert len(controls.lines) == 4
    for line, key in zip(controls.lines, ORDER):
        trace = comparison[key]
        np.testing.assert_array_equal(
            line.get_xydata(), np.column_stack([trace.t[:-1], trace.control_deviation(U_NOM)]))

    scalar, matrix = comparison["scalar"], comparison["matrix"]
    scalar_line, matrix_line, bound_line = decays.lines
    np.testing.assert_array_equal(scalar_line.get_ydata(), scalar.decay_deviation)
    np.testing.assert_array_equal(matrix_line.get_ydata(), matrix.decay_deviation)
    np.testing.assert_array_equal(bound_line.get_xdata(), scalar.t[:-1])
    np.testing.assert_array_equal(
        bound_line.get_ydata(), np.maximum(omega_d, scalar.scalar_certificate) - omega_d)
    assert bound_line.get_label() == "Prop. 2 lower bound"
    assert bound_line.get_zorder() > max(scalar_line.get_zorder(), matrix_line.get_zorder())


# ---------------------------------------------------------------- animate
@pytest.mark.parametrize("skip", [1, 5, 7, 200, 500])
def test_animate_frame_count(comparison, tmp_path, skip):
    """``skip`` keeps every ``skip``-th state, starting from the first."""
    before = _open_figures()
    ani = figures.animate(comparison, skip=skip)
    assert isinstance(ani, animation.FuncAnimation)
    assert _open_figures() == before
    frames = _record(ani, tmp_path)
    assert len(frames) == len(range(0, len(comparison["nominal"].t), skip))


def test_animate_frames(comparison, tmp_path):
    """Every frame shows each loop's arm, ellipse, threshold circle, time and margin."""
    arm, skip = comparison["nominal"].arm, 5
    lim = ANIMATE_MARGIN * arm.reach
    frames = _record(figures.animate(comparison, skip=skip), tmp_path)
    for index, panels in enumerate(frames):
        k = index * skip
        assert len(panels) == len(ORDER)
        for panel, key in zip(panels, ORDER):
            trace = comparison[key]
            points = arm.arm_points(trace.theta[k])
            np.testing.assert_array_equal(panel["arm"], points)
            center, width, height, angle = panel["ellipse"]
            np.testing.assert_array_equal(center, points[-1])
            assert (width, height, angle) == figures._ellipse_axes(arm, trace.theta[k])
            np.testing.assert_array_equal(panel["circle"][0], points[-1])
            assert panel["circle"][1] == DRAW_SCALE * arm.sigma_bar
            assert panel["title"] == (
                f"{TITLES[key]}\n$t$ = {trace.t[k]:.1f} s,  "
                f"$\\lambda_{{\\min}}(\\mathbf{{H}})$ = {trace.margin[k]:+.3f}")
            assert panel["xlim"] == pytest.approx((-lim, lim))
            assert panel["ylim"] == pytest.approx((-lim, lim))


@pytest.mark.parametrize("explicit", [None, (0.5, 1.2, 0.35)])
def test_animate_draws_the_traces_arm(tmp_path, explicit):
    trace_arm = TwoLinkArm(1.5, 0.8, sigma_bar=0.78)
    arm = trace_arm if explicit is None else TwoLinkArm(*explicit[:2], sigma_bar=explicit[2])
    traces = _nominal_traces(trace_arm)
    frames = _record(figures.animate(traces, arm=None if explicit is None else arm, skip=4),
                     tmp_path)
    for index, panels in enumerate(frames):
        for panel in panels:
            np.testing.assert_array_equal(panel["arm"],
                                          arm.arm_points(traces["nominal"].theta[4 * index]))
            assert panel["circle"][1] == DRAW_SCALE * arm.sigma_bar
            assert panel["xlim"] == pytest.approx((-ANIMATE_MARGIN * arm.reach,
                                                   ANIMATE_MARGIN * arm.reach))


def test_animate_renders(comparison, tmp_path):
    """The animation renders; at ``skip=100`` it has the states 0, 100 and 200."""
    path = tmp_path / "loops.gif"
    figures.animate(comparison, skip=100).save(
        str(path), writer=animation.PillowWriter(fps=5), dpi=20)
    with Image.open(path) as image:
        assert image.n_frames == 3
        assert image.size == (round(3.4 * len(ORDER) * 20), 4 * 20)


# ---------------------------------------------------------- _ellipse_axes
def test_ellipse_axes_are_the_singular_values(arm, rng):
    """Full axes ``2 DRAW_SCALE sigma(J)``, i.e. ``2 DRAW_SCALE sqrt(eig M)``, major first."""
    for theta in rng.uniform(-3 * np.pi, 3 * np.pi, size=(200, 2)):
        width, height, _ = figures._ellipse_axes(arm, theta)
        sigma = np.linalg.svd(arm.jacobian(theta), compute_uv=False)
        assert width == pytest.approx(2 * DRAW_SCALE * sigma[0], rel=1e-13)
        assert height == pytest.approx(2 * DRAW_SCALE * sigma[1], abs=1e-11 * arm.reach)
        sigma_min_sq, sigma_max_sq = arm.singular_values_sq(theta[1] - theta[0])
        assert width == pytest.approx(2 * DRAW_SCALE * np.sqrt(sigma_max_sq), rel=1e-13)
        assert height == pytest.approx(2 * DRAW_SCALE * np.sqrt(sigma_min_sq),
                                       abs=1e-11 * arm.reach)


def test_ellipse_axes_orientation(arm, rng):
    """The ellipse is the image of the disc of radius ``DRAW_SCALE`` under ``J``."""
    s = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    unit_circle = np.vstack([np.cos(s), np.sin(s)])
    checked = 0
    for theta in rng.uniform(-3 * np.pi, 3 * np.pi, size=(200, 2)):
        J = arm.jacobian(theta)
        U, sigma, _ = np.linalg.svd(J)
        width, height, angle = figures._ellipse_axes(arm, theta)
        if sigma[0] - sigma[1] > 1e-3 * arm.reach:
            major = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            assert abs(major[0] * U[1, 0] - major[1] * U[0, 0]) < 1e-10
        if sigma[1] > 1e-3 * arm.reach:
            ellipse = Ellipse((0.0, 0.0), width, height, angle=angle)
            image = DRAW_SCALE * (J @ unit_circle).T
            unit = ellipse.get_patch_transform().inverted().transform(image)
            np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0, rtol=0, atol=1e-10)
            checked += 1
    assert checked >= 150


@pytest.mark.parametrize("theta1, diff", [(0.4, 0.0), (0.4, np.pi), (-2.0, 0.0), (1.1, -np.pi)])
def test_ellipse_axes_at_singular_configurations(arm, theta1, diff):
    """Stretched or folded, the ellipse is a segment of length ``2 DRAW_SCALE sqrt(l1**2 + l2**2)``
    along ``(-sin theta1, cos theta1)``; no NaN."""
    width, height, angle = figures._ellipse_axes(arm, (theta1, theta1 + diff))
    assert width == pytest.approx(2 * DRAW_SCALE * math.hypot(arm.l1, arm.l2), rel=1e-13)
    assert 0.0 <= height <= 1e-7 * arm.reach
    offset = (angle - np.degrees(theta1) - 90.0) % 180.0
    assert min(offset, 180.0 - offset) < 1e-9
