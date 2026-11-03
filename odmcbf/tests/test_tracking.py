"""Unit tests of :mod:`odmcbf.tracking`: the turning target, inverse kinematics and the tracking controller."""

from __future__ import annotations

import math

import numpy as np
import pytest

from odmcbf import TurningTarget, TwoLinkArm, inverse_kinematics, rollout, tracking_controller
from odmcbf.tracking import _smoothstep

#: A target that dips from 1.4 to 0.8 over [0.5, 2] s and comes back over [4, 5.5] s.
TARGET = TurningTarget(1.4, 0.8, phi0=0.3, rate=1.2, t_in=(0.5, 2.0), t_out=(4.0, 5.5))


def _finite_difference(f, t, h=1e-6):
    return (np.asarray(f(t + h)) - np.asarray(f(t - h))) / (2 * h)


def test_smoothstep():
    window = (1.0, 3.0)
    assert _smoothstep(0.5, window) == (0.0, 0.0)
    assert _smoothstep(1.0, window) == (0.0, 0.0)
    assert _smoothstep(3.0, window) == (1.0, 0.0)
    assert _smoothstep(4.0, window) == (1.0, 0.0)
    s, ds = _smoothstep(2.0, window)
    assert s == pytest.approx(0.5) and ds == pytest.approx(1.5 / 2.0)
    for t in (1.3, 2.0, 2.9):
        assert _smoothstep(t, window)[1] == pytest.approx(
            _finite_difference(lambda x: _smoothstep(x, window)[0], t), rel=1e-6)


# ---------------------------------------------------------------- target
def test_target_radius_profile():
    assert TARGET.radius(0.0) == (1.4, 0.0)
    assert TARGET.radius(0.5) == (1.4, 0.0)
    assert TARGET.radius(2.0)[0] == pytest.approx(0.8, abs=1e-15)
    assert TARGET.radius(3.0) == pytest.approx((0.8, 0.0), abs=1e-15)
    assert TARGET.radius(5.5)[0] == pytest.approx(1.4, abs=1e-15)
    assert TARGET.radius(9.0) == pytest.approx((1.4, 0.0), abs=1e-15)
    assert TARGET.radius(1.25)[0] == pytest.approx(1.1)          # half way in
    assert TARGET.radius(4.75)[0] == pytest.approx(1.1)          # half way back


def test_target_without_return_stays():
    target = TurningTarget(1.4, 0.8, t_in=(0.0, 1.0))
    for t in (1.0, 5.0, 100.0):
        assert target.radius(t) == pytest.approx((0.8, 0.0), abs=1e-15)


def test_target_position():
    for t in (0.0, 1.0, 2.5, 4.6):
        phi = 0.3 + 1.2 * t
        np.testing.assert_allclose(TARGET.position(t), TARGET.radius(t)[0] * np.array([math.cos(phi), math.sin(phi)]),
                                   rtol=0, atol=1e-15)


@pytest.mark.parametrize("t", [0.3, 0.9, 1.25, 1.99, 3.0, 4.2, 5.0, 6.0])
def test_target_derivatives(t):
    assert TARGET.radius(t)[1] == pytest.approx(_finite_difference(lambda x: TARGET.radius(x)[0], t),
                                                rel=1e-6, abs=1e-8)
    np.testing.assert_allclose(TARGET.velocity(t), _finite_difference(TARGET.position, t),
                               rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("kwargs, match", [
    (dict(r0=0.0), "positive"),
    (dict(r1=-1.0), "positive"),
    (dict(r0=math.inf), "finite"),
    (dict(phi0=math.nan), "finite"),
    (dict(rate=math.inf), "finite"),
    (dict(t_in=(1.0, 1.0)), "finite a < b"),
    (dict(t_in=(0.0, math.inf)), "finite a < b"),
    (dict(t_out=(3.0, 2.0)), "finite a < b"),
    (dict(t_in=(0.0, 2.0), t_out=(1.5, 3.0)), "after t_in"),
])
def test_target_validation(kwargs, match):
    params = dict(r0=1.4, r1=0.8) | kwargs
    with pytest.raises(ValueError, match=match):
        TurningTarget(**params)


# ------------------------------------------------------ inverse kinematics
def test_inverse_kinematics_round_trip(arm, rng):
    r_in, r_out = arm.workspace_radii
    for _ in range(50):
        r = rng.uniform(r_in + 1e-3 * (r_out - r_in), r_out - 1e-3 * (r_out - r_in))
        phi = rng.uniform(-math.pi, math.pi)
        p = r * np.array([math.cos(phi), math.sin(phi)])
        theta = inverse_kinematics(arm, p)
        np.testing.assert_allclose(arm.end_effector(theta), p, rtol=0, atol=1e-12)
        assert 0 < theta[1] - theta[0] < math.pi


def test_inverse_kinematics_on_the_safe_boundary(arm):
    """The boundary circles are reached at ``theta2 - theta1 = hi`` (inner) and ``lo`` (outer)."""
    lo, hi = arm.safe_theta_diff_interval
    r_lo, r_hi = arm.safe_radius_interval
    for r, diff in ((r_lo, hi), (r_hi, lo)):
        theta = inverse_kinematics(arm, (0.0, r))
        assert theta[1] - theta[0] == pytest.approx(diff, abs=1e-9)
        assert abs(arm.margin(theta)) < 1e-9


@pytest.mark.parametrize("r", [0.4, 1.6, 1.7, 0.1])
def test_inverse_kinematics_outside_the_open_workspace(r):
    arm = TwoLinkArm(1.0, 0.6, sigma_bar=0.3)            # workspace 0.4 < |p| < 1.6
    with pytest.raises(ValueError, match="strictly between"):
        inverse_kinematics(arm, (r, 0.0))


# ----------------------------------------------------- tracking controller
@pytest.mark.parametrize("gain", [-1.0, math.nan, math.inf])
def test_controller_rejects_bad_gain(gain):
    with pytest.raises(ValueError, match="gain"):
        tracking_controller(TwoLinkArm(), TARGET, gain)


def test_controller_is_feedforward_plus_position_error(arm, rng):
    """``J u_nom = pdot* + k (p* - p)``."""
    gain = 2.5
    controller = tracking_controller(arm, TARGET, gain)
    lo, hi = arm.safe_theta_diff_interval
    for _ in range(40):
        t = rng.uniform(0, 6)
        theta1 = rng.uniform(-math.pi, math.pi)
        theta = np.array([theta1, theta1 + rng.uniform(lo, hi)])
        expected = TARGET.velocity(t) + gain * (TARGET.position(t) - arm.end_effector(theta))
        np.testing.assert_allclose(arm.jacobian(theta) @ controller(t, theta), expected, rtol=1e-12, atol=1e-12)


def test_on_a_turning_target_the_command_is_rotation_about_joint1(arm):
    """On a target that only turns, at the arm's own distance, the command is ``(w, w)``."""
    theta = np.array([0.2, 0.2 + np.pi / 2])
    x, y = arm.end_effector(theta)
    target = TurningTarget(math.hypot(x, y), math.hypot(x, y), phi0=math.atan2(y, x), rate=0.7)
    u = tracking_controller(arm, target, gain=3.0)(0.0, theta)
    np.testing.assert_allclose(u, [0.7, 0.7], rtol=0, atol=1e-12)


def test_unfiltered_rollout_tracks_the_target(arm):
    """Unfiltered, the end effector stays on a target it starts on, through a violated band."""
    r_in, _ = arm.workspace_radii
    r_lo, _ = arm.safe_radius_interval
    theta0 = np.array([0.0, np.pi / 2])
    x0, y0 = arm.end_effector(theta0)
    r1 = r_lo - 0.3 * (r_lo - r_in)
    target = TurningTarget(math.hypot(x0, y0), r1, phi0=math.atan2(y0, x0), rate=1.0,
                           t_in=(0.0, 1.5), t_out=(3.0, 4.5))
    trace = rollout(arm, tracking_controller(arm, target, 5.0), theta0, dt=0.005, horizon=5.0)
    error = [np.linalg.norm(arm.end_effector(th) - target.position(t)) for t, th in zip(trace.t, trace.theta)]
    assert max(error) < 5e-3
    assert trace.margin.min() < 0                            # the target is in the band
    np.testing.assert_array_equal(trace.u, trace.u_nom)


def test_position_error_decays_at_the_gain():
    """From off a fixed target, ``p* - p`` shrinks like ``exp(-k t)`` along the straight line."""
    arm = TwoLinkArm()
    gain = 2.0
    theta0 = np.array([0.0, np.pi / 2])
    target = TurningTarget(1.2, 1.2, phi0=0.5, rate=0.0)
    trace = rollout(arm, tracking_controller(arm, target, gain), theta0, dt=0.001, horizon=2.0)
    error0 = target.position(0.0) - arm.end_effector(theta0)
    for k in (500, 1000, 2000):
        error = target.position(trace.t[k]) - arm.end_effector(trace.theta[k])
        np.testing.assert_allclose(error, error0 * math.exp(-gain * trace.t[k]), rtol=0,
                                   atol=5e-3 * np.linalg.norm(error0) * math.exp(-gain * trace.t[k]))
