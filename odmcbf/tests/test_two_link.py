"""Unit tests of ``odmcbf.two_link``: kinematics, the barrier ``H`` and its safe set.

Angles are absolute, ``theta = (theta1, theta2)``; joint 1 is at the origin
and joint 2 at ``l1 (cos theta1, sin theta1)``. Rotation of the whole arm
about joint 1 advances ``theta1`` and ``theta2`` together. Finite differences
use a five-point central stencil, and the high-precision references use
mpmath evaluated at the exact floating-point inputs.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest

from odmcbf import DEFAULT_BOUNDARY, TwoLinkArm, TwoLinkKinematics

N_SAMPLES = 400
#: Samples closer than this to a boundary are left out of the sign checks.
EDGE = 1e-9


# ----------------------------------------------------------------- helpers
def _central_difference(f, theta, direction, h=1e-3):
    """Five-point central difference of ``f`` at ``theta`` along ``direction``."""
    theta = np.asarray(theta, dtype=float)
    direction = np.asarray(direction, dtype=float)
    return (-f(theta + 2 * h * direction) + 8 * f(theta + h * direction)
            - 8 * f(theta - h * direction) + f(theta - 2 * h * direction)) / (12 * h)


def _rotation(c: float) -> np.ndarray:
    return np.array([[np.cos(c), -np.sin(c)], [np.sin(c), np.cos(c)]])


def _wrap(angle):
    """``angle`` wrapped to ``[-pi, pi)``."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _random_thetas(rng, n=N_SAMPLES):
    return rng.uniform(-3 * np.pi, 3 * np.pi, size=(n, 2))


def _mp_sigma_min_sq(l1: float, l2: float, theta_diff: float) -> mpmath.mpf:
    """``(tr M - sqrt(tr M**2 - 4 det M)) / 2`` at 400 digits, for the float inputs."""
    with mpmath.workdps(400):
        l1, l2, d = mpmath.mpf(l1), mpmath.mpf(l2), mpmath.mpf(theta_diff)
        trace = l1 ** 2 + l2 ** 2
        det = (l1 * l2 * mpmath.sin(d)) ** 2
        return (trace - mpmath.sqrt(trace ** 2 - 4 * det)) / 2


def _mp_safe_radii(l1: float, l2: float, sigma_bar: float) -> tuple[mpmath.mpf, mpmath.mpf]:
    """``r_lo, r_hi`` from ``l1**2 + l2**2 -+ 2 sqrt(...)`` at 400 digits."""
    with mpmath.workdps(400):
        l1, l2, sb = mpmath.mpf(l1), mpmath.mpf(l2), mpmath.mpf(sigma_bar)
        root = mpmath.sqrt((l1 ** 2 - sb ** 2) * (l2 ** 2 - sb ** 2))
        return (mpmath.sqrt(l1 ** 2 + l2 ** 2 - 2 * root),
                mpmath.sqrt(l1 ** 2 + l2 ** 2 + 2 * root))


def _mp_lo(l1: float, l2: float, sigma_bar: float) -> mpmath.mpf:
    """``lo = arccos(sqrt((l1**2 - sigma_bar**2) (l2**2 - sigma_bar**2)) / (l1 l2))`` at 400 digits."""
    with mpmath.workdps(400):
        l1, l2, sb = mpmath.mpf(l1), mpmath.mpf(l2), mpmath.mpf(sigma_bar)
        return mpmath.acos(mpmath.sqrt((l1 ** 2 - sb ** 2) * (l2 ** 2 - sb ** 2)) / (l1 * l2))


def _rel_err(value: float, reference: mpmath.mpf) -> float:
    with mpmath.workdps(400):
        return float(abs((mpmath.mpf(float(value)) - reference) / reference))


def _v_min(arm: TwoLinkArm, theta) -> np.ndarray:
    """Unit eigenvector of ``H(theta)`` for ``lambda_min``."""
    return np.linalg.eigh(arm.H(theta))[1][:, 0]


# ------------------------------------------------------------ construction
def test_default_boundary_is_pi_over_3():
    assert DEFAULT_BOUNDARY == np.pi / 3


@pytest.mark.parametrize("l1, l2", [
    (0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -0.5),
    (np.nan, 1.0), (1.0, np.nan), (np.inf, 1.0), (1.0, -np.inf),
])
@pytest.mark.parametrize("cls", [TwoLinkKinematics, TwoLinkArm])
def test_rejects_invalid_link_lengths(cls, l1, l2):
    with pytest.raises(ValueError, match="link lengths"):
        cls(l1, l2)


@pytest.mark.parametrize("gamma", [0.0, -1.0, np.nan, np.inf])
def test_arm_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        TwoLinkArm(1.0, 0.6, gamma=gamma, sigma_bar=0.3)


@pytest.mark.parametrize("l1, l2, sigma_bar", [
    (1.0, 1.0, 0.0), (1.0, 1.0, -0.2), (1.0, 1.0, 1.0), (1.0, 0.6, 0.6),
    (0.5, 1.2, 0.5), (0.5, 1.2, 0.8), (1.0, 1.0, np.nan), (1.0, 1.0, np.inf),
])
def test_arm_rejects_sigma_bar_outside_open_interval(l1, l2, sigma_bar):
    """``sigma_bar`` must lie strictly inside ``(0, min(l1, l2))``."""
    with pytest.raises(ValueError, match="sigma_bar"):
        TwoLinkArm(l1, l2, sigma_bar=sigma_bar)


@pytest.mark.parametrize("l1, l2", [(1.0, 1.0), (1.0, 0.6), (0.5, 1.2)])
def test_arm_accepts_sigma_bar_just_inside_interval(l1, l2):
    upper = min(l1, l2)
    for sigma_bar in (np.nextafter(upper, 0.0), 1e-12):
        arm = TwoLinkArm(l1, l2, gamma=3, sigma_bar=sigma_bar)
        assert arm.sigma_bar == sigma_bar
        assert arm.sigma_bar_sq == sigma_bar ** 2
        assert arm.gamma == 3.0
        assert isinstance(arm.gamma, float) and isinstance(arm.sigma_bar, float)
        assert (arm.l1, arm.l2) == (l1, l2)


def test_parameters_are_stored_as_python_floats():
    """Integer and ``float32`` inputs are converted, so all arithmetic is in double precision."""
    l1 = np.float32(0.6)
    kinematics = TwoLinkKinematics(l1, 2)
    arm = TwoLinkArm(l1, 2, gamma=np.float32(1.5), sigma_bar=np.float32(0.3))
    for obj in (kinematics, arm):
        assert type(obj.l1) is float and type(obj.l2) is float
        assert (obj.l1, obj.l2) == (float(l1), 2.0)
        assert obj.reach == float(l1) + 2.0
    assert type(arm.gamma) is float and type(arm.sigma_bar) is float
    assert arm.sigma_bar_sq == float(np.float32(0.3)) ** 2


def test_default_sigma_bar_for_unit_links():
    """For unit links the default threshold is ``sigma_bar**2 = 1/2``, gamma is 1."""
    arm = TwoLinkArm()
    assert (arm.l1, arm.l2, arm.gamma) == (1.0, 1.0, 1.0)
    assert arm.sigma_bar_sq == pytest.approx(0.5, rel=1e-15)
    assert arm.sigma_bar == pytest.approx(math.sqrt(0.5), rel=1e-15)
    assert (arm.n_controls, arm.dim) == (2, 2)


@pytest.mark.parametrize("l1, l2", [(1.0, 1.0), (1.0, 0.6), (0.5, 1.2), (3.0, 0.01)])
def test_default_sigma_bar_puts_boundary_at_default_boundary(l1, l2):
    """By default ``lo = DEFAULT_BOUNDARY`` and ``hi = 2 pi/3`` for any link lengths."""
    arm = TwoLinkArm(l1, l2)
    assert arm.sigma_bar_sq == pytest.approx(arm.sigma_min_sq(DEFAULT_BOUNDARY), rel=1e-15)
    lo, hi = arm.safe_theta_diff_interval
    assert lo == pytest.approx(DEFAULT_BOUNDARY, abs=1e-14)
    assert hi == pytest.approx(2 * np.pi / 3, abs=1e-14)
    scale = max(l1, l2) ** 2
    for theta1 in (0.0, 1.3, -2.4):
        for d in (DEFAULT_BOUNDARY, -DEFAULT_BOUNDARY, 2 * np.pi / 3, -2 * np.pi / 3):
            assert arm.margin((theta1, theta1 + d)) == pytest.approx(0.0, abs=1e-14 * scale)


def test_repr():
    assert repr(TwoLinkKinematics(1, 0.6)) == "TwoLinkKinematics(l1=1, l2=0.6)"
    assert (repr(TwoLinkArm(1, 0.6, gamma=2, sigma_bar=0.3))
            == "TwoLinkArm(l1=1, l2=0.6, gamma=2, sigma_bar=0.3)")
    assert repr(TwoLinkArm()) == "TwoLinkArm(l1=1, l2=1, gamma=1, sigma_bar=0.707107)"


# --------------------------------------------------------------- positions
def test_reach_and_workspace_radii(arm):
    """The outer circle is reached stretched (``theta2 = theta1``), the inner one folded."""
    l1, l2 = arm.l1, arm.l2
    assert arm.reach == l1 + l2
    assert arm.workspace_radii == (abs(l1 - l2), l1 + l2)
    for theta1 in (0.0, 0.7, -2.9):
        stretched = np.linalg.norm(arm.end_effector((theta1, theta1)))
        folded = np.linalg.norm(arm.end_effector((theta1, theta1 + np.pi)))
        assert stretched == pytest.approx(arm.reach, rel=1e-15)
        assert folded == pytest.approx(abs(l1 - l2), abs=1e-15)


def test_joint2_end_effector_and_arm_points(arm, rng):
    l1, l2 = arm.l1, arm.l2
    for theta1, theta2 in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        theta = np.array([theta1, theta2])
        joint2 = arm.joint2(theta)
        np.testing.assert_allclose(joint2, l1 * np.array([np.cos(theta1), np.sin(theta1)]),
                                   rtol=0, atol=1e-15)
        np.testing.assert_array_equal(joint2, arm.joint2((theta1, theta2 + 1.0)))
        p = arm.end_effector(theta)
        np.testing.assert_allclose(
            p, joint2 + l2 * np.array([np.cos(theta2), np.sin(theta2)]), rtol=0, atol=1e-15)
        points = arm.arm_points([theta1, theta2])      # a list works as well
        assert points.shape == (3, 2)
        np.testing.assert_array_equal(points[0], np.zeros(2))
        np.testing.assert_array_equal(points[1], joint2)
        np.testing.assert_array_equal(points[2], p)
        links = np.linalg.norm(np.diff(points, axis=0), axis=1)
        np.testing.assert_allclose(links, [l1, l2], rtol=1e-14)


def test_end_effector_distance_identity(arm, rng):
    """``|p|**2 = l1**2 + l2**2 + 2 l1 l2 cos(theta2 - theta1)``."""
    l1, l2 = arm.l1, arm.l2
    for theta in _random_thetas(rng, 100):
        p_sq = float(np.sum(arm.end_effector(theta) ** 2))
        expected = l1 ** 2 + l2 ** 2 + 2 * l1 * l2 * np.cos(theta[1] - theta[0])
        assert p_sq == pytest.approx(expected, abs=1e-14 * (l1 + l2) ** 2)


# -------------------------------------------------- Jacobian and M = J J.T
def test_jacobian_matches_finite_differences(arm, rng):
    for theta in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        J = arm.jacobian(theta)
        for i, direction in enumerate(np.eye(2)):
            fd = _central_difference(arm.end_effector, theta, direction)
            np.testing.assert_allclose(J[:, i], fd, rtol=0, atol=1e-11)


def test_M_is_J_JT_with_closed_form_trace_and_determinant(arm, rng):
    """``tr M = l1**2 + l2**2`` and ``det M = (l1 l2 sin(theta2 - theta1))**2``."""
    l1, l2 = arm.l1, arm.l2
    for theta in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        J = arm.jacobian(theta)
        M = arm.M(theta)
        np.testing.assert_array_equal(M, J @ J.T)
        np.testing.assert_array_equal(M, M.T)
        assert np.trace(M) == pytest.approx(l1 ** 2 + l2 ** 2, rel=1e-15)
        assert np.linalg.det(M) == pytest.approx(
            (l1 * l2 * np.sin(theta[1] - theta[0])) ** 2, abs=1e-14 * (l1 * l2) ** 2)


def test_dM_matches_finite_differences(arm, rng):
    for theta in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        for dM, direction in ((arm.dM_dtheta1, (1.0, 0.0)), (arm.dM_dtheta2, (0.0, 1.0))):
            fd = _central_difference(arm.M, theta, direction)
            np.testing.assert_allclose(dM(theta), fd, rtol=0, atol=1e-10)


def test_dM_depends_on_its_own_angle_only(arm, rng):
    """``dM/dtheta_i`` is symmetric and traceless, and depends on ``theta_i`` alone."""
    for theta1, theta2 in rng.uniform(-np.pi, np.pi, size=(10, 2)):
        G1 = arm.dM_dtheta1((theta1, theta2))
        G2 = arm.dM_dtheta2((theta1, theta2))
        np.testing.assert_array_equal(G1, arm.dM_dtheta1((theta1, theta2 + 2.1)))
        np.testing.assert_array_equal(G2, arm.dM_dtheta2((theta1 - 0.8, theta2)))
        for G, length in ((G1, arm.l1), (G2, arm.l2)):
            np.testing.assert_array_equal(G, G.T)
            assert np.trace(G) == pytest.approx(0.0, abs=1e-15 * length ** 2)
            # The eigenvalues of l**2 [[sin 2t, -cos 2t], [-cos 2t, -sin 2t]] are +-l**2.
            np.testing.assert_allclose(np.linalg.eigvalsh(G), [-length ** 2, length ** 2],
                                       rtol=1e-14)


# ------------------------------------------------------------------ barrier
def test_H_is_M_minus_sigma_bar_sq_identity(arm, rng):
    for theta in rng.uniform(-np.pi, np.pi, size=(10, 2)):
        np.testing.assert_array_equal(arm.H(theta), arm.M(theta) - arm.sigma_bar_sq * np.eye(2))


def test_alpha_H_is_gamma_H(rng):
    for gamma in (1.0, 2.5, 0.1):
        arm = TwoLinkArm(1.0, 0.6, gamma=gamma, sigma_bar=0.3)
        for theta in rng.uniform(-np.pi, np.pi, size=(5, 2)):
            np.testing.assert_array_equal(arm.alpha_H(theta), gamma * arm.H(theta))


def test_LgH_is_joint_derivatives_of_M(arm, rng):
    theta = rng.uniform(-np.pi, np.pi, 2)
    LgH = arm.LgH(theta)
    assert isinstance(LgH, list) and len(LgH) == arm.n_controls == 2
    np.testing.assert_array_equal(LgH[0], arm.dM_dtheta1(theta))
    np.testing.assert_array_equal(LgH[1], arm.dM_dtheta2(theta))


def test_Hdot_is_time_derivative_of_H(arm, rng):
    """``Hdot(theta, u) = d/dt H(theta + t u)`` at ``t = 0``: the barrier is driftless."""
    for theta in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        u = rng.uniform(-1.0, 1.0, 2)
        fd = _central_difference(arm.H, theta, u)
        np.testing.assert_allclose(arm.Hdot(theta, u), fd, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(arm.Hdot(theta, (0.0, 0.0)), np.zeros((2, 2)))


def test_Hdot_is_linear_in_u(arm, rng):
    theta = rng.uniform(-np.pi, np.pi, 2)
    u, w = rng.normal(size=2), rng.normal(size=2)
    np.testing.assert_allclose(arm.Hdot(theta, 2.0 * u - 3.0 * w),
                               2.0 * arm.Hdot(theta, u) - 3.0 * arm.Hdot(theta, w),
                               rtol=0, atol=1e-14 * (arm.l1 + arm.l2) ** 2 * 10)


# ------------------------------------------------ spectrum of M vs theta2 - theta1
def test_singular_values_sq_match_svd_of_J(arm, rng):
    for theta in _random_thetas(rng, 50):
        s = np.linalg.svd(arm.jacobian(theta), compute_uv=False)
        mu_min, mu_max = arm.singular_values_sq(theta[1] - theta[0])
        assert mu_min == pytest.approx(s[1] ** 2, rel=1e-12, abs=1e-15)
        assert mu_max == pytest.approx(s[0] ** 2, rel=1e-14)
        assert arm.sigma_min_sq(theta[1] - theta[0]) == mu_min


def test_singular_values_sq_accept_arrays(arm):
    theta_diff = np.linspace(-7.0, 7.0, 24).reshape(4, 6)
    mu_min, mu_max = arm.singular_values_sq(theta_diff)
    assert mu_min.shape == mu_max.shape == theta_diff.shape
    np.testing.assert_array_equal(arm.sigma_min_sq(theta_diff), mu_min)
    for index, d in np.ndenumerate(theta_diff):
        scalar_min, scalar_max = arm.singular_values_sq(float(d))
        assert mu_min[index] == pytest.approx(scalar_min, rel=1e-14)
        assert mu_max[index] == pytest.approx(scalar_max, rel=1e-14)
    # The eigenvalues of M sum to tr M and multiply to det M.
    np.testing.assert_allclose(mu_min + mu_max, arm.l1 ** 2 + arm.l2 ** 2, rtol=1e-15)
    np.testing.assert_allclose(mu_min * mu_max, (arm.l1 * arm.l2 * np.sin(theta_diff)) ** 2,
                               rtol=1e-14)


def test_sigma_min_sq_depends_on_cos_sq_of_theta_diff(arm):
    """The spectrum is even in ``theta2 - theta1`` and has period ``pi``."""
    d = np.linspace(0.05, 3.0, 30)
    reference = arm.sigma_min_sq(d)
    for other in (-d, np.pi - d, d + np.pi, d - 2 * np.pi):
        np.testing.assert_allclose(arm.sigma_min_sq(other), reference, rtol=1e-13)


@pytest.mark.parametrize("l1, l2", [(1.0, 1.0), (1.0, 0.6), (0.5, 1.2), (1.5, 0.8),
                                    (1.0, 1.0 + 1e-9)])
@pytest.mark.parametrize("theta_diff", [
    1e-15, 1e-12, 1e-9, 1e-6, -1e-10, np.pi - 1e-12, np.pi - 1e-9, np.pi - 1e-6,
    -np.pi + 1e-10, 2 * np.pi - 1e-9, np.pi,
], ids=lambda d: f"{d:.17g}")
def test_sigma_min_sq_accurate_near_singular_configurations(l1, l2, theta_diff):
    """``det M / sigma_max**2`` has no cancellation where ``sigma_min`` vanishes."""
    value = TwoLinkKinematics(l1, l2).sigma_min_sq(theta_diff)
    assert value > 0
    assert _rel_err(value, _mp_sigma_min_sq(l1, l2, theta_diff)) < 1e-14


def test_sigma_min_sq_vanishes_stretched(arm):
    assert arm.sigma_min_sq(0.0) == 0.0
    assert arm.singular_values_sq(0.0)[1] == pytest.approx(arm.l1 ** 2 + arm.l2 ** 2, rel=1e-15)


def test_max_sigma_min_sq_attained_at_quarter_turn(arm):
    """``max sigma_min(J)**2 = min(l1, l2)**2``, at ``theta2 - theta1 = +-pi/2``."""
    best = arm.max_sigma_min_sq
    assert best == min(arm.l1, arm.l2) ** 2
    for d in (np.pi / 2, -np.pi / 2, 3 * np.pi / 2):
        assert arm.sigma_min_sq(d) == pytest.approx(best, rel=1e-15)
    grid = np.linspace(-np.pi, np.pi, 20001)
    values = arm.sigma_min_sq(grid)
    assert values.max() <= best * (1 + 1e-15)
    assert abs(abs(grid[np.argmax(values)]) - np.pi / 2) < 1e-3
    # Any sigma_bar below min(l1, l2) leaves a strictly positive margin there.
    assert arm.margin((0.4, 0.4 + np.pi / 2)) == pytest.approx(best - arm.sigma_bar_sq, rel=1e-12)
    assert arm.margin((0.4, 0.4 + np.pi / 2)) > 0


# ------------------------------------------------------------------- margin
def test_margin_is_sigma_min_sq_minus_sigma_bar_sq(arm, rng):
    for theta in _random_thetas(rng, 50):
        margin = arm.margin(theta)
        assert isinstance(margin, float)
        assert margin == pytest.approx(np.linalg.eigvalsh(arm.H(theta)).min(), abs=1e-15)
        assert margin == pytest.approx(
            arm.sigma_min_sq(theta[1] - theta[0]) - arm.sigma_bar_sq,
            abs=1e-14 * (arm.l1 + arm.l2) ** 2)


def test_stretched_and_folded_configurations_violate_the_barrier(arm):
    for theta1 in (0.0, 1.1, -2.5):
        assert arm.margin((theta1, theta1)) == pytest.approx(-arm.sigma_bar_sq, abs=1e-14)
        assert arm.margin((theta1, theta1 + np.pi)) == pytest.approx(-arm.sigma_bar_sq, abs=1e-14)


@pytest.mark.parametrize("c", [0.3, -1.7, np.pi, 5.0])
def test_rotation_about_joint1_conjugates_H(arm, rng, c):
    """``H(theta + (c, c)) = R(c) H(theta) R(c).T``, so the margin is unchanged."""
    R = _rotation(c)
    scale = 1e-14 * (arm.l1 + arm.l2) ** 2
    for theta in rng.uniform(-np.pi, np.pi, size=(10, 2)):
        shifted = theta + c
        np.testing.assert_allclose(arm.H(shifted), R @ arm.H(theta) @ R.T, rtol=0, atol=scale)
        np.testing.assert_allclose(arm.jacobian(shifted), R @ arm.jacobian(theta),
                                   rtol=0, atol=1e-15 * (arm.l1 + arm.l2))
        assert arm.margin(shifted) == pytest.approx(arm.margin(theta), abs=scale)
        assert np.linalg.norm(arm.end_effector(shifted)) == pytest.approx(
            np.linalg.norm(arm.end_effector(theta)), rel=1e-14, abs=1e-15)


def test_margin_rate_along_u(arm, rng):
    """``v_min.T Hdot(theta, u) v_min = (u2 - u1) d sigma_min_sq / d(theta2 - theta1)``.

    Rotation of the whole arm about joint 1, ``u1 = u2``, leaves the margin
    unchanged to first order.
    """
    for theta in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        d = theta[1] - theta[0]
        if arm.l1 == arm.l2 and abs(np.cos(d)) < 1e-2:
            continue                        # repeated eigenvalue of M at +-pi/2
        v = _v_min(arm, theta)
        u = rng.uniform(-1.0, 1.0, 2)
        slope = _central_difference(arm.sigma_min_sq, d, 1.0)
        assert v @ arm.Hdot(theta, u) @ v == pytest.approx((u[1] - u[0]) * slope, abs=1e-10)
        assert v @ arm.Hdot(theta, (u[0], u[0])) @ v == pytest.approx(0.0, abs=1e-14)


def test_margin_rate_sign_at_boundary(arm):
    """On the boundary the margin decreases along ``u`` iff ``(u2 - u1) sin 2(theta2 - theta1) < 0``."""
    lo, hi = arm.safe_theta_diff_interval
    controls = [(1.0, 0.4), (0.4, 1.0), (-1.0, 0.5), (0.3, -0.7), (0.0, 1.0), (1.0, 0.0)]
    for d in (lo, -lo, hi, -hi):
        theta = np.array([0.6, 0.6 + d])
        v = _v_min(arm, theta)
        for u in controls:
            rate = v @ arm.Hdot(theta, u) @ v
            assert abs(rate) > 1e-6
            assert np.sign(rate) == np.sign((u[1] - u[0]) * np.sin(2 * d))


# ----------------------------------------------------------- the safe set
def test_safe_theta_diff_interval_closed_form(arm):
    """``cos(lo)**2 = (l1**2 - sigma_bar**2) (l2**2 - sigma_bar**2) / (l1**2 l2**2)``, ``hi = pi - lo``."""
    l1, l2, r = arm.l1, arm.l2, arm.sigma_bar_sq
    lo, hi = arm.safe_theta_diff_interval
    assert isinstance(lo, float) and isinstance(hi, float)
    assert 0.0 < lo < np.pi / 2 < hi < np.pi
    assert hi == np.pi - lo
    assert np.cos(lo) ** 2 == pytest.approx((l1 ** 2 - r) * (l2 ** 2 - r) / (l1 ** 2 * l2 ** 2),
                                            rel=1e-14)
    assert _rel_err(lo, _mp_lo(l1, l2, arm.sigma_bar)) < 1e-14


def test_margin_zero_at_safe_theta_diff_bounds(arm):
    lo, hi = arm.safe_theta_diff_interval
    scale = 1e-14 * (arm.l1 + arm.l2) ** 2
    for theta1 in (0.0, 0.9, -2.2, 7.0):
        for d in (lo, -lo, hi, -hi, lo + 2 * np.pi, -hi - 2 * np.pi):
            assert arm.margin((theta1, theta1 + d)) == pytest.approx(0.0, abs=scale)
        # Inside and outside the interval, the margin has the matching sign.
        assert arm.margin((theta1, theta1 + lo + 1e-3)) > 0
        assert arm.margin((theta1, theta1 + lo - 1e-3)) < 0
        assert arm.margin((theta1, theta1 + hi - 1e-3)) > 0
        assert arm.margin((theta1, theta1 + hi + 1e-3)) < 0


def test_safe_theta_diff_interval_matches_margin_sign(arm, rng):
    """``H >= 0`` iff ``lo <= |theta2 - theta1| <= hi`` (mod ``2 pi``), whatever ``theta1``."""
    lo, hi = arm.safe_theta_diff_interval
    thetas = _random_thetas(rng)
    wrapped = np.abs(_wrap(thetas[:, 1] - thetas[:, 0]))
    keep = (np.abs(wrapped - lo) > EDGE) & (np.abs(wrapped - hi) > EDGE)
    inside = (lo <= wrapped) & (wrapped <= hi)
    safe = np.array([arm.margin(theta) >= 0 for theta in thetas])
    np.testing.assert_array_equal(safe[keep], inside[keep])
    assert keep.sum() > 0.99 * len(thetas)
    assert 0 < safe.sum() < len(thetas)


@pytest.mark.parametrize("l1, l2, sigma_bar", [
    (1.0, 1.0, 1e-9), (1.0, 0.6, 1e-12), (0.5, 1.2, 1e-6),
])
def test_safe_theta_diff_interval_accurate_for_small_sigma_bar(l1, l2, sigma_bar):
    """``(lo, hi)`` lies in ``(0, pi)`` and ``lo`` keeps full relative accuracy as ``sigma_bar -> 0``.

    Exactly, ``lo`` is about ``sigma_bar sqrt(1/l1**2 + 1/l2**2)``. The
    cancellation that ``safe_radius_interval`` avoids for ``r_lo`` also hits
    ``arccos`` of ``sqrt(cos_sq)``, with ``cos_sq`` rounded close to 1.
    """
    lo, hi = TwoLinkArm(l1, l2, sigma_bar=sigma_bar).safe_theta_diff_interval
    assert 0.0 < lo and hi < np.pi
    assert _rel_err(lo, _mp_lo(l1, l2, sigma_bar)) < 1e-14


@pytest.mark.parametrize("l1, l2", [(1.0, 1.0), (1.0, 0.6), (0.5, 1.2)])
@pytest.mark.parametrize("boundary", [0.3, 1.0, 2.0, -0.7, 0.3 + 2 * np.pi, 4.0, 1.5])
def test_from_boundary_puts_margin_zero_at_boundary(l1, l2, boundary):
    """The boundary is at ``b``, ``pi - b`` and their negatives."""
    arm = TwoLinkArm.from_boundary(boundary, l1, l2, gamma=2.0)
    assert type(arm) is TwoLinkArm
    assert (arm.l1, arm.l2, arm.gamma) == (l1, l2, 2.0)
    assert arm.sigma_bar_sq == pytest.approx(arm.sigma_min_sq(boundary), rel=1e-15)
    for d in (boundary, np.pi - boundary, -boundary, boundary - np.pi):
        for theta1 in (0.0, -1.2):
            assert arm.margin((theta1, theta1 + d)) == pytest.approx(0.0, abs=1e-14)
    folded = abs(_wrap(boundary))
    lo = min(folded, np.pi - folded)
    assert arm.safe_theta_diff_interval[0] == pytest.approx(lo, abs=1e-12)


@pytest.mark.parametrize("boundary", [1e-6, 1e-9])
def test_from_boundary_recovers_small_boundary(boundary):
    """``from_boundary(b).safe_theta_diff_interval[0] = b`` also for a boundary close to 0."""
    arm = TwoLinkArm.from_boundary(boundary, 1.0, 0.6)
    assert arm.safe_theta_diff_interval[0] == pytest.approx(boundary, rel=1e-13)


@pytest.mark.parametrize("l1, l2", [(1.0, 1.0), (1.0, 0.6), (0.5, 1.2), (3.0, 0.01)])
def test_from_boundary_at_default_boundary_is_default_arm(l1, l2):
    assert (TwoLinkArm.from_boundary(DEFAULT_BOUNDARY, l1, l2).sigma_bar
            == TwoLinkArm(l1, l2).sigma_bar)


def test_from_boundary_and_repr_follow_subclasses():
    """``from_boundary`` is a classmethod: a subclass gets an instance of itself."""
    class Arm(TwoLinkArm):
        pass

    arm = Arm.from_boundary(1.0, 1.0, 0.6, gamma=2.0)
    assert type(arm) is Arm
    assert repr(arm) == f"Arm(l1=1, l2=0.6, gamma=2, sigma_bar={arm.sigma_bar:.6g})"


@pytest.mark.parametrize("boundary", [0.0, np.pi, np.pi / 2, -np.pi / 2, 3 * np.pi / 2,
                                      2 * np.pi, -np.pi, np.nan, np.inf])
def test_from_boundary_rejects_multiples_of_quarter_turn(boundary):
    """At ``0`` or ``pi`` the threshold would be ``sigma_bar = 0``; at ``+-pi/2`` there is no interior."""
    with pytest.raises(ValueError, match="multiple of pi/2"):
        TwoLinkArm.from_boundary(boundary, 1.0, 0.6)


# --------------------------------------------------------- safe radii
def test_safe_radius_interval_lies_inside_workspace(arm):
    inner, outer = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    assert isinstance(r_lo, float) and isinstance(r_hi, float)
    assert inner < r_lo < r_hi < outer


def test_safe_radius_interval_closed_form(arm):
    """``r_lo**2, r_hi**2 = l1**2 + l2**2 -+ 2 sqrt((l1**2 - sigma_bar**2) (l2**2 - sigma_bar**2))``."""
    l1, l2, r = arm.l1, arm.l2, arm.sigma_bar_sq
    root = math.sqrt((l1 ** 2 - r) * (l2 ** 2 - r))
    r_lo, r_hi = arm.safe_radius_interval
    assert r_lo ** 2 == pytest.approx(l1 ** 2 + l2 ** 2 - 2 * root, rel=1e-12)
    assert r_hi ** 2 == pytest.approx(l1 ** 2 + l2 ** 2 + 2 * root, rel=1e-15)


def test_safe_radius_interval_consistent_with_theta_diff_interval(arm):
    """Via ``|p|**2``, ``r_hi`` corresponds to ``lo`` and ``r_lo`` to ``hi``."""
    l1, l2 = arm.l1, arm.l2
    lo, hi = arm.safe_theta_diff_interval
    r_lo, r_hi = arm.safe_radius_interval
    assert r_hi ** 2 == pytest.approx(l1 ** 2 + l2 ** 2 + 2 * l1 * l2 * np.cos(lo), rel=1e-14)
    assert r_lo ** 2 == pytest.approx(l1 ** 2 + l2 ** 2 + 2 * l1 * l2 * np.cos(hi), rel=1e-12)
    assert np.linalg.norm(arm.end_effector((0.2, 0.2 + lo))) == pytest.approx(r_hi, rel=1e-14)
    assert np.linalg.norm(arm.end_effector((0.2, 0.2 + hi))) == pytest.approx(r_lo, rel=1e-14)


def test_margin_zero_on_safe_radius_circles(arm):
    """Both configurations reaching each boundary circle lie on the boundary of the safe set."""
    l1, l2 = arm.l1, arm.l2
    scale = 1e-14 * (l1 + l2) ** 2
    for radius in arm.safe_radius_interval:
        d = np.arccos((radius ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
        for sign in (1.0, -1.0):
            # Rotation of the whole arm about joint 1 moves p along the circle.
            for theta1 in np.linspace(-np.pi, np.pi, 7):
                theta = (theta1, theta1 + sign * d)
                assert np.linalg.norm(arm.end_effector(theta)) == pytest.approx(radius, rel=1e-14)
                assert arm.margin(theta) == pytest.approx(0.0, abs=scale)


def test_safe_radius_interval_matches_margin_sign(arm, rng):
    """``H >= 0`` iff ``r_lo <= |p| <= r_hi``; the rest of the workspace is two bands."""
    r_lo, r_hi = arm.safe_radius_interval
    inner, outer = arm.workspace_radii
    thetas = _random_thetas(rng)
    radii = np.array([np.linalg.norm(arm.end_effector(theta)) for theta in thetas])
    assert np.all((radii >= inner - 1e-15) & (radii <= outer + 1e-15))
    keep = (np.abs(radii - r_lo) > EDGE) & (np.abs(radii - r_hi) > EDGE)
    inside = (r_lo <= radii) & (radii <= r_hi)
    safe = np.array([arm.margin(theta) >= 0 for theta in thetas])
    np.testing.assert_array_equal(safe[keep], inside[keep])
    assert keep.sum() > 0.99 * len(thetas)
    # Both violating bands, next to the inner and to the outer circle, are sampled.
    assert np.any(radii[keep] < r_lo) and np.any(radii[keep] > r_hi) and np.any(inside)


def test_mirror_configurations_share_the_margin(arm, rng):
    """The two configurations reaching ``p`` have opposite ``theta2 - theta1`` and one margin."""
    for theta in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        p = arm.end_effector(theta)
        angle = np.arctan2(p[1], p[0])
        mirror = 2 * angle - theta              # reflection across the line through p
        np.testing.assert_allclose(arm.end_effector(mirror), p, rtol=0, atol=1e-14)
        assert _wrap(mirror[1] - mirror[0]) == pytest.approx(-_wrap(theta[1] - theta[0]),
                                                              abs=1e-12)
        assert arm.margin(mirror) == pytest.approx(arm.margin(theta), abs=1e-14)


@pytest.mark.parametrize("l1, l2, sigma_bar", [
    (1.0, 1.0, 1e-9), (1.0, 1.0, 1e-12), (2.0, 2.0, 1e-9),
    (1.0, 1.0 + 1e-9, 1e-9), (1.0, 1.0 - 1e-12, 1e-12), (1.0, 1.0 + 1e-6, 1e-9),
    (1.0, 0.6, 1e-9), (1.0, 0.6, 1e-12), (1.5, 0.8, 1e-9), (1.0, 0.9, 1e-12),
    (1.0, 1.0, 0.1), (1.0, 0.6, 0.3), (0.5, 1.2, 0.35), (1.5, 0.8, 0.78),
])
def test_safe_radius_interval_accurate_for_small_sigma_bar(l1, l2, sigma_bar):
    """``r_lo`` avoids the cancellation in ``l1**2 + l2**2 - 2 sqrt(...)``."""
    r_lo, r_hi = TwoLinkArm(l1, l2, sigma_bar=sigma_bar).safe_radius_interval
    ref_lo, ref_hi = _mp_safe_radii(l1, l2, sigma_bar)
    assert _rel_err(r_lo, ref_lo) < 1e-14
    assert _rel_err(r_hi, ref_hi) < 1e-14


@pytest.mark.parametrize("l1, l2, sigma_bar", [
    (1.0, 0.6, 1e-9), (1.5, 0.8, 1e-9), (1.0, 0.9, 1e-12), (0.5, 0.9, 1e-9),
    (3.0, 0.8, 1e-15), (1.0, 0.6, 0.3),
])
def test_r_lo_never_below_inner_workspace_radius(l1, l2, sigma_bar):
    arm = TwoLinkArm(l1, l2, sigma_bar=sigma_bar)
    assert arm.safe_radius_interval[0] >= arm.workspace_radii[0] == abs(l1 - l2)


def test_r_lo_never_below_inner_workspace_radius_random_lengths(rng):
    for l1, l2 in rng.uniform(0.1, 3.0, size=(1000, 2)):
        for sigma_bar in (1e-12, 1e-9):
            r_lo = TwoLinkArm(l1, l2, sigma_bar=sigma_bar).safe_radius_interval[0]
            assert r_lo >= abs(l1 - l2)


@pytest.mark.parametrize("l1, l2, sigma_bar", [
    (1.5, 0.8, 1e-9), (1.0, 0.9, 1e-12), (0.7, 0.2, 1e-9),
])
def test_safe_radius_interval_never_leaves_workspace(l1, l2, sigma_bar):
    """Exactly ``|l1 - l2| < r_lo < r_hi < l1 + l2``; rounding may close a band, not invert it."""
    arm = TwoLinkArm(l1, l2, sigma_bar=sigma_bar)
    inner, outer = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    assert inner <= r_lo <= r_hi <= outer == arm.reach


@pytest.mark.parametrize("length", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("sigma_bar", [1e-12, 1e-9, 0.1, 0.4])
def test_r_lo_for_equal_links(length, sigma_bar):
    """For ``l1 = l2 = l``, ``r_lo = sqrt(2) sigma_bar > 0`` and ``r_hi**2 = 4 l**2 - 2 sigma_bar**2``."""
    r_lo, r_hi = TwoLinkArm(length, length, sigma_bar=sigma_bar).safe_radius_interval
    assert r_lo > 0
    assert r_lo == pytest.approx(math.sqrt(2) * sigma_bar, rel=1e-14)
    assert r_hi ** 2 == pytest.approx(4 * length ** 2 - 2 * sigma_bar ** 2, rel=1e-15)


# ------------------------------------------------------------- margin rate
def _eigenvalue_gap(arm: TwoLinkArm, theta) -> float:
    lam = np.linalg.eigvalsh(arm.H(theta))
    return float(lam[1] - lam[0])


def test_margin_rate_is_the_derivative_of_the_margin(arm, rng):
    """``v0' Hdot v0`` against a five-point difference of ``lambda_min(H)`` along ``u``.

    Configurations within 1e-2 of a repeated eigenvalue, where ``lambda_min`` has a
    kink, are left out; they exist only for ``l1 = l2``.
    """
    checked = 0
    for theta, u in zip(_random_thetas(rng, 100), rng.normal(size=(100, 2))):
        if _eigenvalue_gap(arm, theta) < 1e-2:
            continue
        expected = _central_difference(arm.margin, theta, u)
        assert arm.margin_rate(theta, u) == pytest.approx(expected, rel=1e-6, abs=1e-9)
        checked += 1
    assert checked >= 50


def test_margin_rate_vanishes_under_rotation_about_joint_1(arm, rng):
    """Advancing both joints together rotates ``H`` and keeps its spectrum."""
    scale = arm.l1 ** 2 + arm.l2 ** 2
    for theta, w in zip(_random_thetas(rng, 100), rng.normal(size=100)):
        if _eigenvalue_gap(arm, theta) < 1e-2:
            continue
        assert arm.margin_rate(theta, (w, w)) == pytest.approx(0.0, abs=1e-12 * scale * (1 + abs(w)))


def test_margin_rate_raises_at_a_repeated_eigenvalue():
    """``l1 = l2`` at ``theta2 - theta1 = pi/2``: ``M = l**2 I``, so ``lambda_min(H)`` is repeated."""
    with pytest.raises(ValueError, match="repeated"):
        TwoLinkArm().margin_rate((0.3, 0.3 + np.pi / 2), (1.0, 0.0))
