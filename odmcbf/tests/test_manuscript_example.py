"""Section VIII of the paper (singularity avoidance) on other arms, and the relative-angle example.

The paper's own experiments (Figs. 1 to 3 and the numbers of Sec. VIII-C) and the
closed forms of Sec. VIII-A and VIII-B are checked in ``test_paper.py``. Two barriers
are used here.

:class:`~odmcbf.TwoLinkArm`, the package's arm in absolute joint angles, for every
``(l1, l2, sigma_bar, gamma)`` of the ``manuscript_arm`` fixture. On it are checked the
statements of Section VIII that hold for any arm: rotation of the whole arm about
joint 1 acts on ``H`` by a commutator with the rotation generator ``S``, grazes the
boundary with ``Z00 = Z11 = 0`` and ``|Z01| = |v| lambda_H``, is certified by the
multiplier of eq. (33) at a bounded cost, and needs the unbounded scalar decay of
eq. (36). The notebook's closed loop with a constant command (the ``comparison``
fixture of ``conftest.py``) is checked against the qualitative claims.

``RelativeArm`` is the barrier of the relative-angle example, which is not in the
paper (see the README and ``main_relative.ipynb``): unit links, relative joint angles
``q = (q1, q2)``, ``sigma_bar**2 = (3 - sqrt 5) / 2``, and a constant command
``(1, -0.6)`` rad/s for 8 s. It is written out here independently of
:mod:`odmcbf.two_link_relative` and serves as a regression test of the filters on a
second barrier. The package's filters and ``run_comparison`` run on it unchanged and
give the reported numbers: 1.02, 3.12 and 7.91 rad of rotation about joint 1, and
0.987 rad/s for matrix OD at the end.
"""

from __future__ import annotations

import numpy as np
import pytest

from odmcbf import (
    FilterInfeasible,
    SafetyFilter,
    TwoLinkArm,
    TwoLinkKinematics,
    min_scalar_decay,
    run_comparison,
    summarize,
)
from odmcbf.simulate import SETTLE_WINDOW

from .eigenblocks import (
    blocks,
    certify,
    od_lmi,
    scalar_never_certifies,
    sym,
)

#: Feasibility slack allowed on a returned LMI.
LMI_TOL = 1e-7
#: The rotation generator, ``R(t) = expm(t S)``.
S = np.array([[0.0, -1.0], [1.0, 0.0]])
#: The default matrix penalty, ``rho_Omega = (4 / p) rho_omega`` with ``rho_omega = 5e-3``, ``p = 2``.
RHO_OMEGA_MATRIX = 4 * 5e-3 / 2


@pytest.fixture
def arm(manuscript_arm):
    """Here ``arm`` also varies ``gamma``: conftest's arms, and two of them with ``gamma = 2.5``."""
    return manuscript_arm


def _rotation(t: float) -> np.ndarray:
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _boundary_thetas(arm: TwoLinkArm, theta1: float = 0.3):
    lo, hi = arm.safe_theta_diff_interval
    for theta_diff in (lo, -lo, hi, -hi):
        yield np.array([theta1, theta1 + theta_diff])


def _safe_thetas(arm: TwoLinkArm, rng, n: int = 6):
    """Random configurations in the safe set, and some within ``1e-6`` of its boundary."""
    lo, hi = arm.safe_theta_diff_interval
    for _ in range(n):
        theta1 = rng.uniform(-np.pi, np.pi)
        yield np.array([theta1, theta1 + rng.choice([-1, 1]) * rng.uniform(lo, hi)])
    for theta_diff in (lo + 1e-6, hi - 1e-6, -(lo + 1e-3), -(hi - 1e-1)):
        yield np.array([0.7, 0.7 + theta_diff])


# ================================================= the package's arm, absolute angles
@pytest.mark.parametrize("v", [1.0, -0.6])
def test_rotation_about_joint1_is_a_commutator(arm, rng, v):
    """Section VIII: rotating the whole arm about joint 1 at rate ``v`` gives ``Hdot = v (S H - H S)``.

    In absolute angles that rotation is ``u = (v, v)``. It holds at every
    configuration, safe or not.
    """
    for theta in rng.uniform(-np.pi, np.pi, size=(10, 2)):
        H = arm.H(theta)
        np.testing.assert_allclose(arm.Hdot(theta, (v, v)), v * (S @ H - H @ S),
                                   rtol=0, atol=1e-14 * (arm.l1 + arm.l2) ** 2)


@pytest.mark.parametrize("v", [1.0, -0.6])
def test_rotation_about_joint1_grazes_the_boundary(arm, v):
    """Section VIII-A: at a boundary configuration the rotation has ``Z00 = Z11 = 0`` and ``|Z01| = |v| lambda_H``.

    ``lambda_H`` is the nonzero eigenvalue of ``H`` there, ``tr H = l1**2 +
    l2**2 - 2 sigma_bar**2``. Proposition 1 then rules out every scalar
    certificate, while Theorem 1 admits a matrix one.
    """
    lambda_H = arm.l1 ** 2 + arm.l2 ** 2 - 2 * arm.sigma_bar_sq
    scale = 1e-13 * (arm.l1 + arm.l2) ** 2
    for theta in _boundary_thetas(arm):
        b = blocks(arm.H(theta), arm.alpha_H(theta), arm.Hdot(theta, (v, v)))
        assert np.linalg.eigvalsh(arm.H(theta))[1] == pytest.approx(lambda_H, rel=1e-13)
        assert b.Z00.shape == b.Z11.shape == (1, 1)
        assert abs(b.Z00[0, 0]) <= scale and abs(b.Z11[0, 0]) <= scale
        assert abs(b.Z01[0, 0]) == pytest.approx(abs(v) * lambda_H, rel=1e-12)
        assert certify("matrix", arm.alpha_H(theta), arm.Hdot(theta, (v, v))).safety_residual >= -LMI_TOL


@pytest.mark.parametrize("v", [1.0, -0.6, 3.0])
def test_rotation_certificate_eq33(arm, rng, v):
    """Eq. (33): ``Omega = I/2 + (v / gamma) S`` cancels the rotation's ``Hdot`` exactly.

    The LMI of eq. (16) is then ``gamma H``, at every configuration; ``sym(Omega)
    = I/2`` meets the recovery constraint with ``beta = 1/2``; and
    ``||Omega - I/2||_F = sqrt(2) |v| / gamma``.
    """
    gamma = arm.gamma
    Omega = 0.5 * np.eye(2) + (v / gamma) * S
    np.testing.assert_array_equal(sym(Omega), 0.5 * np.eye(2))
    assert np.linalg.norm(Omega - 0.5 * np.eye(2)) == pytest.approx(np.sqrt(2) * abs(v) / gamma, rel=1e-15)
    for theta in [*rng.uniform(-np.pi, np.pi, size=(10, 2)), *_boundary_thetas(arm)]:
        X = od_lmi(arm.Hdot(theta, (v, v)), arm.alpha_H(theta), Omega)
        np.testing.assert_allclose(X, gamma * arm.H(theta), rtol=0,
                                   atol=1e-13 * (1 + abs(v)) * (arm.l1 + arm.l2) ** 2)


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("v", [1.0, -0.6])
def test_rotation_costs_matrix_od_little_and_scalar_od_without_bound(arm, rng, v, recovery):
    """Eqs. (33) and (36) through the filters, over the safe set.

    Because eq. (33) certifies the rotation, the matrix filter's cost of
    certifying it is at most ``rho_Omega ||Omega - I/2||_F**2 = 2 rho_Omega
    (v / gamma)**2``, up to the boundary. In the interior the scalar filter
    needs ``omega = max(omega_d, |v| (lambda1 - lambda0) / (gamma
    sqrt(lambda0 lambda1)))``, with ``lambda0 <= lambda1`` the eigenvalues of
    ``H``, which grows without bound as ``lambda0 -> 0``.
    """
    largest = 0.0
    for theta in _safe_thetas(arm, rng):
        A, Hdot = arm.alpha_H(theta), arm.Hdot(theta, (v, v))
        cert = certify("matrix", A, Hdot, recovery=recovery)
        assert cert.safety_residual >= -LMI_TOL
        assert cert.objective <= 2 * RHO_OMEGA_MATRIX * (v / arm.gamma) ** 2 + 1e-9

        lam0, lam1 = np.linalg.eigvalsh(arm.H(theta))
        bound = abs(v) * (lam1 - lam0) / (arm.gamma * np.sqrt(lam0 * lam1))
        assert min_scalar_decay(A, Hdot) == pytest.approx(bound, rel=1e-6)
        omega = certify("scalar", A, Hdot, recovery=recovery).omega
        assert omega == pytest.approx(max(1.0, bound), rel=1e-5, abs=1e-3)
        largest = max(largest, omega)
    assert largest > 1e2


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("v", [1.0, -0.6])
@pytest.mark.parametrize("boundary", range(4), ids=["lo", "-lo", "hi", "-hi"])
def test_scalar_od_never_certifies_the_rotation_at_the_boundary(arm, boundary, v, recovery):
    """Section VIII-B with Proposition 1: at the boundary no scalar decay certifies the rotation.

    Where CLARABEL fails outright instead of reporting infeasibility, SCS
    decides instead (see ``eigenblocks.scalar_never_certifies``).
    """
    theta = list(_boundary_thetas(arm))[boundary]
    scalar_never_certifies(arm.alpha_H(theta), arm.Hdot(theta, (v, v)), recovery=recovery)


# ------------------------------- the notebook's closed loop, with a constant command
def test_matrix_filter_deviates_least(comparison):
    """The matrix filter has the smallest control deviation of the three filters."""
    deviation = {name: summarize(trace, (1.0, 0.4))["control_deviation_l2"]
                 for name, trace in comparison.items()}
    assert deviation["nominal"] == 0.0
    assert deviation["matrix"] < min(deviation["scalar"], deviation["standard"])


def test_scalar_decay_is_the_proposition2_bound(comparison):
    """Panel (c) of the notebook's closed-loop figure: at every step the scalar filter's decay is ``max(omega_d, omega*)``.

    ``omega*`` is the smallest scalar decay certifying the applied control,
    Proposition 2, recorded in ``scalar_certificate``. Where it exceeds
    ``omega_d`` the two agree to the solver's precision; elsewhere the
    penalty pins ``omega`` at ``omega_d`` only to about ``5e-4``.
    """
    trace = comparison["scalar"]
    omega = 2 * trace.Omega[:, 0, 0]
    bound = np.maximum(1.0, trace.scalar_certificate)
    assert np.all(np.isfinite(bound))
    active = trace.scalar_certificate > 1.0 + 1e-3
    assert active.sum() > 0.5 * len(active)
    np.testing.assert_allclose(omega[active], bound[active], rtol=1e-5)
    np.testing.assert_allclose(omega[~active], bound[~active], rtol=0, atol=1e-3)


def test_matrix_filter_settles_at_the_certified_rotation(comparison):
    """Settled on the boundary, the matrix filter rotates the whole arm about joint 1
    at the rate that balances the command against the cost of certifying it.

    At ``theta2 - theta1 = pi/3`` a boundary control must have ``u2 >= u1``;
    for a rotation ``(w, w)`` the certificate costs ``(4/3) rho_Omega w**2``
    with recovery, so ``w`` minimises ``2 (w - 0.7)**2 + (4/3) rho_Omega w**2``,
    ``w = 0.7 / (1 + (2/3) rho_Omega)``. The multiplier then stays below the
    ``(2 / sqrt 3) w`` of that certificate, and below the ``sqrt(2) w`` of
    eq. (33). The loop settles just inside the boundary, so these hold to
    about ``5e-4``.
    """
    trace = comparison["matrix"]
    window = trace.t[:-1] > trace.t[-1] - SETTLE_WINDOW
    w = 0.7 / (1 + 2 / 3 * RHO_OMEGA_MATRIX)
    np.testing.assert_allclose(trace.u[window], w, rtol=0, atol=5e-4)
    settled = trace.decay_deviation[window]
    assert np.all(settled <= 2 / np.sqrt(3) * w * (1 + 1e-3))
    assert np.all(settled < np.sqrt(2) * w)


# ======================================== the relative-angle arm (not in the paper)
class RelativeArm:
    """The barrier of the relative-angle example, in its own coordinates.

    Unit links and joint configuration ``q = (q1, q2)``, where ``q2`` is the
    angle of link 2 relative to link 1, with ``qdot = u`` and

        J(q) = [[-s1 - s12, -s12], [c1 + c12, c12]],   H(q) = J J.T - sigma_bar**2 I,

    ``s1 = sin q1``, ``s12 = sin(q1 + q2)``, ``alpha(H) = gamma H``. By default
    ``sigma_bar**2 = (3 - sqrt 5) / 2``, which puts the boundary at
    ``q2 = pi/2``. Joint 1 acts by rigid rotation, so ``dM/dq1 = S M - M S``.
    The class has the interface :func:`odmcbf.rollout` uses.
    """

    n_controls = 2
    dim = 2

    def __init__(self, gamma: float = 1.0, sigma_bar_sq: float = (3 - np.sqrt(5)) / 2) -> None:
        self.gamma = gamma
        self.sigma_bar_sq = sigma_bar_sq

    def jacobian(self, q) -> np.ndarray:
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        return np.array([[-s1 - s12, -s12], [c1 + c12, c12]])

    def M(self, q) -> np.ndarray:
        J = self.jacobian(q)
        return J @ J.T

    def H(self, q) -> np.ndarray:
        return self.M(q) - self.sigma_bar_sq * np.eye(2)

    def alpha_H(self, q) -> np.ndarray:
        return self.gamma * self.H(q)

    def LgH(self, q) -> list[np.ndarray]:
        M = self.M(q)
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        X = np.array([[-c12, -c12], [-s12, -s12]]) @ self.jacobian(q).T     # (dJ/dq2) J.T
        return [S @ M - M @ S, X + X.T]

    def Hdot(self, q, u) -> np.ndarray:
        G1, G2 = self.LgH(q)
        return u[0] * G1 + u[1] * G2

    def margin(self, q) -> float:
        return float(np.linalg.eigvalsh(self.H(q))[0])


#: The relative-angle closed loop: start, command, and the reported joint-1 rotation for each filter, in rad.
Q0 = np.array([0.0, 2 * np.pi / 3])
K_NOM = np.array([1.0, -0.6])
REPORTED_ROTATION = {"standard": 1.02, "scalar": 3.12, "matrix": 7.91}
#: Half a unit in the second decimal of the reported numbers, plus 1e-3 for the time
#: step: halving it moves the standard filter's rotation by 7e-4. The scalar filter's 3.12497 rad is 0.00497 from the reported 3.12.
ROTATION_TOL = 0.006


@pytest.fixture(scope="module")
def relative_runs():
    """The relative-angle closed loop: ``gamma = omega_d = 1``, ``rho_omega = 0.005``, recovery on, ``T = 8`` s.

    ``dt = 0.01`` and ``0.005`` give the same numbers to the reported digits.
    """
    return run_comparison(RelativeArm(), u_nom=K_NOM, theta0=Q0, dt=0.01, horizon=8.0,
                          omega_d=1.0, rho_omega=5e-3, recovery=True)


def _five_point(f, x, h=1e-4):
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


def test_relative_arm_is_the_package_arm_in_relative_angles(rng):
    """This arm's Jacobian is the package arm's, ``dp/dtheta``, times ``dtheta/dq = [[1, 0], [1, 1]]``.

    With ``theta = (q1, q1 + q2)`` the end effector is the same, and the
    control directions of ``RelativeArm`` are the derivatives of its ``H``.
    """
    arm, kinematics = RelativeArm(), TwoLinkKinematics(1.0, 1.0)
    T = np.array([[1.0, 0.0], [1.0, 1.0]])
    for q in rng.uniform(-np.pi, np.pi, size=(10, 2)):
        theta = T @ q
        np.testing.assert_allclose(arm.jacobian(q), kinematics.jacobian(theta) @ T, rtol=0, atol=1e-15)
        for G, e in zip(arm.LgH(q), np.eye(2)):
            fd = _five_point(lambda t, q=q, e=e: arm.H(q + t * e), 0.0)
            np.testing.assert_allclose(G, fd, rtol=0, atol=1e-10)


def test_joint1_acts_by_rigid_rotation(rng):
    """``H(q1, q2) = R(q1) H(0, q2) R(q1).T``, so the margin depends on ``q2`` alone."""
    arm = RelativeArm()
    for q1, q2 in rng.uniform(-np.pi, np.pi, size=(10, 2)):
        R = _rotation(q1)
        np.testing.assert_allclose(arm.H((q1, q2)), R @ arm.H((0.0, q2)) @ R.T, rtol=0, atol=1e-14)
        assert arm.margin((q1, q2)) == pytest.approx(arm.margin((0.0, q2)), abs=1e-14)


def test_threshold_puts_the_boundary_at_q2_quarter_turn():
    """``sigma_bar**2 = (3 - sqrt 5)/2`` is ``sigma_min(J)**2`` at ``q2 = pi/2``.

    The command closes ``q2`` from ``2 pi/3`` and reaches that boundary before
    the stretched singular configuration ``q2 = 0``.
    """
    arm = RelativeArm()
    for q1 in (0.0, 0.4, -2.0):
        for q2 in (np.pi / 2, -np.pi / 2):
            sigma_min = np.linalg.svd(arm.jacobian((q1, q2)), compute_uv=False)[-1]
            assert sigma_min ** 2 == pytest.approx(arm.sigma_bar_sq, rel=1e-14)
            assert abs(arm.margin((q1, q2))) <= 1e-15
        assert arm.margin((q1, Q0[1])) > 0
        assert arm.margin((q1, np.pi / 2 + 0.2)) > 0 > arm.margin((q1, np.pi / 2 - 0.2))
        assert arm.margin((q1, 0.0)) == pytest.approx(-arm.sigma_bar_sq, abs=1e-15)


def test_margin_slope_at_the_boundary():
    """At ``q2 = pi/2``, ``d lambda_min(H) / d q2 = 3 / sqrt 5 - 1 > 0``, and ``q1`` does not enter."""
    arm = RelativeArm()
    for q1 in (0.0, 0.4, -2.0):
        q = np.array([q1, np.pi / 2])
        slope = _five_point(lambda t, q=q: arm.margin(q + np.array([0.0, t])), 0.0)
        assert slope == pytest.approx(3 / np.sqrt(5) - 1, abs=1e-9)
        v0 = np.linalg.eigh(arm.H(q))[1][:, 0]
        assert v0 @ arm.LgH(q)[1] @ v0 == pytest.approx(3 / np.sqrt(5) - 1, abs=1e-14)
        assert v0 @ arm.LgH(q)[0] @ v0 == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("u2", [-0.5, -1e-2, 0.0, 1e-2, 0.4])
@pytest.mark.parametrize("u1", [-1.0, 0.0, 0.7, 2.0])
def test_tangency_condition_is_u2_nonnegative(u1, u2, recovery):
    """At ``q2 = pi/2`` the tangency condition is ``u2 >= 0``, ``u1`` unrestricted.

    The matrix filter certifies exactly those controls (Theorem 1). The scalar
    filter certifies ``u2 > 0`` and ``u = 0``, but not the grazing controls
    ``u = (v, 0)`` with ``v != 0`` (Proposition 1).
    """
    arm = RelativeArm()
    q = np.array([0.4, np.pi / 2])
    A, Hdot = arm.alpha_H(q), arm.Hdot(q, (u1, u2))
    if u2 >= 0:
        assert certify("matrix", A, Hdot, recovery=recovery).safety_residual >= -LMI_TOL
    else:
        with pytest.raises(FilterInfeasible, match="matrix filter"):
            certify("matrix", A, Hdot, recovery=recovery)
    if u2 > 0 or u1 == 0.0 and u2 == 0.0:
        assert certify("scalar", A, Hdot, recovery=recovery).safety_residual >= -LMI_TOL
    elif u2 == 0.0:
        scalar_never_certifies(A, Hdot, recovery=recovery)


@pytest.mark.parametrize("v", [1.0, -0.7, 2.0])
def test_grazing_control(v):
    """At the boundary ``u = (v, 0)`` gives ``Z00 = Z11 = 0``, ``|Z01| = |v| lambda_H``.

    Here ``lambda_H = sqrt 5``, the eigenvalue gap of ``M`` at ``q2 = pi/2``.
    ``Hdot = v (S H - H S)``, and the paper's eq. (33) turns the LMI into ``gamma H``.
    """
    arm = RelativeArm(gamma=2.0)
    for q1 in (0.0, 0.4, -2.0):
        q = np.array([q1, np.pi / 2])
        H, Hdot = arm.H(q), arm.Hdot(q, (v, 0.0))
        np.testing.assert_allclose(Hdot, v * (S @ H - H @ S), rtol=0, atol=1e-14)
        b = blocks(H, arm.alpha_H(q), Hdot)
        assert abs(b.Z00[0, 0]) <= 1e-14 and abs(b.Z11[0, 0]) <= 1e-14
        assert abs(b.Z01[0, 0]) == pytest.approx(abs(v) * np.sqrt(5), rel=1e-14)
        Omega = 0.5 * np.eye(2) + (v / arm.gamma) * S
        np.testing.assert_allclose(od_lmi(Hdot, arm.alpha_H(q), Omega), arm.gamma * H, rtol=0, atol=1e-14)


def test_filters_at_the_boundary():
    """The control closest to ``(1, -0.6)`` meeting ``u2 >= 0`` is ``(1, 0)``.

    The matrix filter returns it, less the rate that pays for the certificate:
    ``(4/3) rho_Omega u1**2``, so ``u1 = 1 / (1 + (4/3) rho_Omega)``. The
    standard and scalar filters also cut the rotation about joint 1, by more
    than half.
    """
    arm = RelativeArm()
    q = np.array([0.4, np.pi / 2])
    res = {decay: SafetyFilter(2, 2, decay=decay)(arm.alpha_H(q), arm.LgH(q), K_NOM)
           for decay in ("standard", "scalar", "matrix")}
    np.testing.assert_allclose(res["matrix"].u, [1 / (1 + 4 / 3 * RHO_OMEGA_MATRIX), 0.0],
                               rtol=0, atol=1e-4)
    for decay in ("standard", "scalar"):
        assert res[decay].u[0] < 0.5
        assert res[decay].objective > res["matrix"].objective + 0.5


def test_closed_loop_settings():
    """Section VIII-A: ``rho_omega = 0.005`` gives ``rho_Omega = 0.01``; the two penalties agree on ``(omega/2) I``."""
    filt = SafetyFilter(2, 2, decay="matrix", rho_omega=5e-3)
    assert filt.rho_Omega == pytest.approx(0.01, rel=1e-15)
    assert (filt.omega_d, filt.beta, filt.recovery) == (1.0, 0.5, True)


def test_closed_loop_reported_rotation(relative_runs):
    """The reported numbers: the standard, scalar OD and matrix OD filters complete 1.02, 3.12 and
    7.91 rad of rotation about joint 1, against the commanded 8 rad."""
    rotation = {name: trace.theta[-1, 0] - trace.theta[0, 0] for name, trace in relative_runs.items()}
    assert rotation["nominal"] == pytest.approx(8.0, rel=1e-12)
    for name, reported in REPORTED_ROTATION.items():
        assert rotation[name] == pytest.approx(reported, abs=ROTATION_TOL), name


def test_closed_loop_reported_late_rate(relative_runs):
    """The reported number: the matrix filter's mean joint-1 rate over the final 2 s is 0.987 rad/s.

    The remaining deviation is the penalty on the multiplier: on the boundary
    the rate is ``1 / (1 + (4/3) rho_Omega)``, as in ``test_filters_at_the_boundary``.
    """
    late = summarize(relative_runs["matrix"], K_NOM)["late_joint1_rate"]
    assert late == pytest.approx(0.987, abs=6e-4)
    assert late == pytest.approx(1 / (1 + 4 / 3 * RHO_OMEGA_MATRIX), abs=1e-4)


def test_closed_loop_constraint(relative_runs):
    """All three filters maintain the constraint; the nominal command violates it.

    Unfiltered, ``q2`` passes the stretched singular configuration, where the
    margin reaches ``-sigma_bar**2``. The matrix filter runs along the boundary
    ``q2 = pi/2`` from above and ends there, braking joint 2 to a stop.
    """
    sigma_bar_sq = (3 - np.sqrt(5)) / 2
    assert relative_runs["nominal"].margin.min() == pytest.approx(-sigma_bar_sq, abs=1e-4)
    for name in ("standard", "scalar", "matrix"):
        trace = relative_runs[name]
        assert trace.margin.min() >= -1e-9, name
        assert np.nanmin(trace.safety_residual) >= -LMI_TOL, name
    matrix = relative_runs["matrix"]
    assert matrix.theta[:, 1].min() >= np.pi / 2 - 1e-9
    assert matrix.theta[-1, 1] - np.pi / 2 < 1e-4
    window = matrix.t[:-1] > matrix.t[-1] - SETTLE_WINDOW
    assert np.abs(matrix.u[window, 1]).max() < 1e-4


def test_closed_loop_decay_deviations(relative_runs):
    """The scalar decay deviation settles near six, as its certificate bound predicts,
        while the matrix deviation stays near one.

    The scalar filter's ``omega`` is ``max(omega_d, omega*)`` at every step
    (Proposition 2). The matrix multiplier settles at the cheapest certificate
    of the settled rotation, ``(2 / sqrt 3) w``, below the ``sqrt(2) w`` of
    the paper's eq. (33).
    """
    scalar = relative_runs["scalar"]
    window = scalar.t[:-1] > scalar.t[-1] - SETTLE_WINDOW
    settled = summarize(scalar, K_NOM)["decay_deviation_settled"]
    assert 5.5 < settled < 6.5
    omega = 2 * scalar.Omega[:, 0, 0]
    np.testing.assert_allclose(omega[window], scalar.scalar_certificate[window], rtol=1e-5)

    matrix = relative_runs["matrix"]
    w = summarize(matrix, K_NOM)["late_joint1_rate"]
    settled = summarize(matrix, K_NOM)["decay_deviation_settled"]
    assert 0.99 * 2 / np.sqrt(3) * w <= settled <= 2 / np.sqrt(3) * w * (1 + 1e-3)
    assert settled < np.sqrt(2) * w
    assert settled == pytest.approx(1.0, abs=0.2)


def test_closed_loop_matrix_filter_deviates_least(relative_runs):
    """Matrix OD has the smallest control deviation of the three filters."""
    deviation = {name: summarize(trace, K_NOM)["control_deviation_l2"]
                 for name, trace in relative_runs.items()}
    assert deviation["matrix"] < min(deviation["scalar"], deviation["standard"])

