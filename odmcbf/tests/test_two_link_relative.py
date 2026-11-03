"""The two-link arm in relative joint angles, :mod:`odmcbf.two_link_relative` (not in the paper).

``q = (q1, q2)`` holds the base angle and the elbow angle measured from link 1, with unit
links and ``sigma_bar**2 = (3 - sqrt 5) / 2``, which puts the boundary at ``q2 = pi/2``.
Checked here: the Jacobian and the control directions against finite differences, that
``q1`` only rotates ``H``, the boundary, agreement with the barrier of
``test_manuscript_example.RelativeArm``, and the README's closed loop, ``u = (1, -0.6)``
rad/s for 8 s from ``q = (0, 2 pi/3)``, in which the standard MCBF, scalar OD and matrix OD
complete 1.02, 3.12 and 7.91 rad of the commanded 8 rad of base rotation.
"""

from __future__ import annotations

import numpy as np
import pytest

from odmcbf import run_comparison
from odmcbf.two_link_relative import (
    Q2_BOUNDARY,
    S,
    SIGMA_BAR_SQ,
    RelativeTwoLinkArm,
    arm_points,
    jacobian,
)

from .test_manuscript_example import RelativeArm

#: The README's closed loop and the base rotation it reports for each filter, in rad.
Q0 = np.array([0.0, 2 * np.pi / 3])
U_NOM = np.array([1.0, -0.6])
REPORTED_ROTATION = {"standard": 1.02, "scalar": 3.12, "matrix": 7.91}
#: As in ``test_manuscript_example``: half a unit in the second decimal, plus 1e-3 for the time step.
ROTATION_TOL = 0.006


def _random_q(rng, n=20):
    return rng.uniform(-np.pi, np.pi, size=(n, 2))


def _central(f, q, i, h=1e-6):
    e = np.zeros(2)
    e[i] = h
    return (f(q + e) - f(q - e)) / (2 * h)


@pytest.fixture
def arm():
    return RelativeTwoLinkArm()


def test_constants():
    assert SIGMA_BAR_SQ == pytest.approx((3 - np.sqrt(5)) / 2, rel=1e-15)
    assert Q2_BOUNDARY == np.pi / 2
    np.testing.assert_array_equal(S, [[0.0, -1.0], [1.0, 0.0]])


def test_rejects_nonpositive_gamma():
    for gamma in (0.0, -1.0):
        with pytest.raises(ValueError, match="gamma must be positive"):
            RelativeTwoLinkArm(gamma=gamma)


def test_attributes_and_repr():
    arm = RelativeTwoLinkArm(gamma=2.5)
    assert (arm.n_controls, arm.dim, arm.reach, arm.l1, arm.l2) == (2, 2, 2.0, 1.0, 1.0)
    assert arm.gamma == 2.5
    assert arm.sigma_bar == pytest.approx(np.sqrt(SIGMA_BAR_SQ), rel=1e-15)
    assert repr(arm) == f"RelativeTwoLinkArm(gamma=2.5, sigma_bar_sq={SIGMA_BAR_SQ:.6g})"


def test_arm_points_have_unit_links_at_the_link_angles(rng):
    for q in _random_q(rng):
        p = arm_points(q)
        assert p.shape == (3, 2)
        np.testing.assert_array_equal(p[0], [0.0, 0.0])
        np.testing.assert_allclose(p[1], [np.cos(q[0]), np.sin(q[0])], atol=1e-15)
        np.testing.assert_allclose(p[2] - p[1], [np.cos(q.sum()), np.sin(q.sum())], atol=1e-15)


def test_jacobian_is_the_derivative_of_the_end_effector(arm, rng):
    end = lambda q: arm.arm_points(q)[-1]
    for q in _random_q(rng):
        J = arm.jacobian(q)
        np.testing.assert_array_equal(J, jacobian(q))
        for i in range(2):
            np.testing.assert_allclose(J[:, i], _central(end, q, i), atol=1e-8)


def test_barrier_and_class_k_term(rng):
    arm = RelativeTwoLinkArm(gamma=2.5)
    for q in _random_q(rng):
        J = arm.jacobian(q)
        np.testing.assert_allclose(arm.M(q), J @ J.T, atol=1e-15)
        np.testing.assert_allclose(arm.H(q), J @ J.T - SIGMA_BAR_SQ * np.eye(2), atol=1e-15)
        np.testing.assert_allclose(arm.alpha_H(q), 2.5 * arm.H(q), atol=1e-15)
        assert arm.margin(q) == pytest.approx(np.linalg.eigvalsh(arm.H(q))[0], abs=1e-15)


def test_control_directions_are_the_partials_of_H(arm, rng):
    for q in _random_q(rng):
        G = arm.LgH(q)
        for i in range(2):
            np.testing.assert_allclose(G[i], _central(arm.H, q, i), atol=1e-8)
        Hq = arm.H(q)
        np.testing.assert_allclose(G[0], S @ Hq - Hq @ S, atol=1e-12)
        u = rng.normal(size=2)
        np.testing.assert_allclose(arm.Hdot(q, u), u[0] * G[0] + u[1] * G[1], atol=1e-15)


def test_base_angle_only_rotates_H(arm, rng):
    for q in _random_q(rng):
        t = rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        np.testing.assert_allclose(arm.H(q + [t, 0.0]), R @ arm.H(q) @ R.T, atol=1e-12)
        assert arm.margin(q + [t, 0.0]) == pytest.approx(arm.margin(q), abs=1e-12)


def test_boundary_at_q2_pi_over_2(arm, rng):
    for q1 in rng.uniform(-np.pi, np.pi, size=5):
        for q2 in (Q2_BOUNDARY, -Q2_BOUNDARY):
            assert arm.margin([q1, q2]) == pytest.approx(0.0, abs=1e-12)
        assert arm.margin([q1, Q2_BOUNDARY + 0.1]) > 0      # the side the closed loop starts on
        assert arm.margin([q1, Q0[1]]) > 0
        assert arm.margin([q1, Q2_BOUNDARY - 0.1]) < 0      # toward the stretched-out singularity
        assert arm.margin([q1, 0.0]) < 0


def test_agrees_with_the_relative_arm_of_the_regression_test(rng):
    """``RelativeArm`` in ``test_manuscript_example`` is an independent copy of this barrier."""
    ours, theirs = RelativeTwoLinkArm(gamma=1.7), RelativeArm(gamma=1.7)
    for q in _random_q(rng):
        np.testing.assert_allclose(ours.H(q), theirs.H(q), atol=1e-14)
        np.testing.assert_allclose(ours.alpha_H(q), theirs.alpha_H(q), atol=1e-14)
        for a, b in zip(ours.LgH(q), theirs.LgH(q)):
            np.testing.assert_allclose(a, b, atol=1e-14)


@pytest.fixture(scope="module")
def runs():
    return run_comparison(RelativeTwoLinkArm(), u_nom=U_NOM, theta0=Q0, dt=0.01, horizon=8.0,
                          omega_d=1.0, rho_omega=5e-3, recovery=True)


def test_closed_loop_reported_rotation(runs):
    """The README's numbers: 1.02, 3.12 and 7.91 rad of the commanded 8 rad of base rotation."""
    rotation = {name: trace.theta[-1, 0] - trace.theta[0, 0] for name, trace in runs.items()}
    assert rotation["nominal"] == pytest.approx(8.0, rel=1e-12)
    for name, reported in REPORTED_ROTATION.items():
        assert rotation[name] == pytest.approx(reported, abs=ROTATION_TOL), name


def test_closed_loop_filters_stay_safe(runs, arm):
    for name in REPORTED_ROTATION:
        margins = [arm.margin(q) for q in runs[name].theta]
        assert min(margins) > -1e-6, name
