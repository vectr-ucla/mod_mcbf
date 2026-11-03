"""Section VIII of the paper, Singularity Avoidance, checked against the implementation.

The experiments are those of :mod:`odmcbf.paper`; ``scripts/reproduce_paper.py`` draws
the figures from the same runs. Equation numbers are those of the arXiv version
(arXiv:2610.18675), with the LaTeX label of each in brackets.

The closed forms of Sec. VIII-A and VIII-B are checked in sympy on the symbolic arm and
numerically on the hero arm:

====================================  ==========================================================
eq. (27) [eq:arm_jacobian]            ``J``, and ``det J = l1 l2 sin(theta2 - theta1)``
eq. (28) [eq:singularitybarrier]      ``H = J J^T - sigma_bar^2 I``; ``tr H`` is constant
eq. (29) [eq:arm_det]                 ``det H`` depends on ``theta2 - theta1`` alone
eq. (30) [eq:arm_annulus]             the safe set is the annulus ``TwoLinkArm.safe_radius_interval``
eq. (31) [eq:arm_rates]               ``d/dt det H`` and ``|Hdot|_F^2 / 2``
the tangent cone condition            ``(theta2dot - theta1dot) sin 2(theta2 - theta1) >= 0`` at the boundary
the rotation about joint 1            keeps the eigenvalues of ``H``; on the boundary ``|b| = |theta1dot| tr H``
eq. (33) [eq:rotationcertificate]     the matrix multiplier cancels ``Hdot``; ``||Omega - I/2||_F = sqrt 2 |theta1dot|``
eq. (34) [eq:arm_scalar]              ``det(Hdot + omega H)``
eq. (35) [eq:arm_omega_boundary]      the smallest scalar decay at the boundary
eq. (36) [eq:rotation_scalar]         the scalar decay the rotation needs in the interior
eq. (37) [eq:rotation_cap]            the standard MCBF's rotation cap; ``0.8229`` rad/s on the hero arm
====================================  ==========================================================

The settings of Sec. VIII-A are ``l1 = 1``, ``sigma_bar = 0.6``, ``kappa = 5``, ``omega_d = 1``,
``rho_omega = 0.005``, ``rho_Omega = 0.01``, ``beta = 1/2``, ``dt = 0.01``, solver tolerances ``1e-9``.

The closed loops are checked against the numbers Sec. VIII-C prints, and against regression
values that the paper's text does not print:

- Fig. 1, the hero run (regression values): the held distances to the target, margins, scalar
  OD's ``omega`` and the rotations about joint 1;
- the six tracking cases: every filter keeps ``lambda_min(H) > 0``; the standard MCBF is farthest
  from the target, then scalar OD, then matrix OD; the smallest scalar decay certifying matrix
  OD's inputs has a median of 24 to 36, a penalty distance of 17 to 25, and scalar OD's mean
  ``omega`` is 6.3 to 8.3;
- Fig. 3, recovery: from ``lambda_min(H) = -0.098`` the margin rises at exactly ``2 beta`` with
  the recovery constraint, and the arm stays unsafe without it.

The solve times of Sec. VIII-C depend on the machine and are not checked.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from odmcbf import (
    NO_RECOVERY,
    SafetyFilter,
    band_start,
    min_scalar_decay,
    paper,
    recovery_rate,
)
from odmcbf.filters import CLARABEL_PARAMS

#: The hero arm, and its numbers. The paper's text does not print these; they pin down Fig. 1's run.
HERO_L2 = paper.HERO_ARM["l2"]
ROTATION_CAP = 0.8229                      # eq. (37) on the hero arm, rad/s
SAFE_RADII = (0.918, 1.511)                # eq. (30) on the hero arm
#: Fig. 1, held over 1.75 <= t < 3.25 s: the distance to the target, lambda_min(H),
#: and the rotation about joint 1 over the run, per filter; scalar OD's omega; the least possible distance.
HERO_HELD_DISTANCE = {"standard": 0.895, "scalar": 0.0935, "matrix": 0.0741}
HERO_HELD_MARGIN = {"standard": 1.31e-1, "scalar": 1.99e-2, "matrix": 1.94e-3}
HERO_ROTATION = {"standard": 2.82, "scalar": 4.50, "matrix": 4.50}
HERO_OMEGA = 6.26
HERO_LEAST_DISTANCE = 0.0718
#: Sec. VIII-C: the ranges over the six cases, as printed.
MEDIAN_DECAY_RANGE = (24, 36)
PENALTY_DISTANCE_RANGE = (17, 25)
MEAN_OMEGA_RANGE = (6.3, 8.3)
#: The recovery paragraph and Fig. 3. The paper prints the starting margin; the rest are regression values.
RECOVERY_MARGIN0 = -0.098
RECOVERY_LEFT = {0.5: 0.00657, 0.25: 0.0816}   # lambda_min(H) at 5 s relative to the start
NO_RECOVERY_LEFT = 0.970


# ------------------------------------------------------------ the symbolic arm
@pytest.fixture(scope="module")
def symbolic():
    l1, l2, sb, w = sp.symbols("l1 l2 sigma omega", positive=True)
    t1, t2, d1, d2, v = sp.symbols("theta1 theta2 theta1dot theta2dot v", real=True)
    J = sp.Matrix([[-l1 * sp.sin(t1), -l2 * sp.sin(t2)], [l1 * sp.cos(t1), l2 * sp.cos(t2)]])
    H = J * J.T - sb ** 2 * sp.eye(2)
    Hdot = sp.diff(H, t1) * d1 + sp.diff(H, t2) * d2
    return dict(l1=l1, l2=l2, sb=sb, w=w, t1=t1, t2=t2, d1=d1, d2=d2, v=v, J=J, H=H, Hdot=Hdot,
                D=t2 - t1, ddet=sp.diff(H.det(), t1) * d1 + sp.diff(H.det(), t2) * d2,
                fro=sum(Hdot[i, j] ** 2 for i in range(2) for j in range(2)))


def _zero(expr) -> None:
    assert sp.simplify(sp.expand_trig(expr)) == 0


def test_eq27_eq28_eq29_jacobian_barrier_trace_and_determinant(symbolic):
    s = symbolic
    _zero(s["J"].det() - s["l1"] * s["l2"] * sp.sin(s["D"]))
    _zero(s["H"].trace() - (s["l1"] ** 2 + s["l2"] ** 2 - 2 * s["sb"] ** 2))
    _zero(s["H"].det() - ((s["l1"] ** 2 - s["sb"] ** 2) * (s["l2"] ** 2 - s["sb"] ** 2)
                          - s["l1"] ** 2 * s["l2"] ** 2 * sp.cos(s["D"]) ** 2))


def test_eq30_safe_set_is_the_annulus_of_safe_radius_interval(symbolic):
    """``|y|^2 = l1^2 + l2^2 + 2 l1 l2 cos(theta2 - theta1)``, so ``det H >= 0`` is eq. (30); on the hero arm
    the radii are 0.918 and 1.511, and the margin changes sign on each circle."""
    s = symbolic
    y = sp.Matrix([s["l1"] * sp.cos(s["t1"]) + s["l2"] * sp.cos(s["t2"]),
                   s["l1"] * sp.sin(s["t1"]) + s["l2"] * sp.sin(s["t2"])])
    _zero((y.T * y)[0] - (s["l1"] ** 2 + s["l2"] ** 2 + 2 * s["l1"] * s["l2"] * sp.cos(s["D"])))
    arm = paper.arm()
    r_lo, r_hi = arm.safe_radius_interval
    root = math.sqrt((arm.l1 ** 2 - arm.sigma_bar_sq) * (arm.l2 ** 2 - arm.sigma_bar_sq))
    assert r_hi ** 2 == pytest.approx(arm.l1 ** 2 + arm.l2 ** 2 + 2 * root, rel=1e-12)
    assert r_lo ** 2 == pytest.approx(arm.l1 ** 2 + arm.l2 ** 2 - 2 * root, rel=1e-12)
    assert (round(r_lo, 3), round(r_hi, 3)) == SAFE_RADII
    for radius, sign in ((r_lo - 1e-3, -1), (r_lo + 1e-3, 1), (r_hi - 1e-3, 1), (r_hi + 1e-3, -1)):
        diff = math.acos((radius ** 2 - arm.l1 ** 2 - arm.l2 ** 2) / (2 * arm.l1 * arm.l2))
        assert math.copysign(1, arm.margin((0.4, 0.4 + diff))) == sign


def test_eq31_rates(symbolic):
    s = symbolic
    _zero(s["ddet"] - s["l1"] ** 2 * s["l2"] ** 2 * sp.sin(2 * s["D"]) * (s["d2"] - s["d1"]))
    _zero(s["fro"] / 2 - (s["l1"] ** 4 * s["d1"] ** 2 + s["l2"] ** 4 * s["d2"] ** 2
                          + 2 * s["l1"] ** 2 * s["l2"] ** 2 * s["d1"] * s["d2"] * sp.cos(2 * s["D"])))
    assert sp.simplify(s["Hdot"].trace()) == 0


def test_tangent_cone_condition_at_the_boundary(rng):
    """``tr H v0' Hdot v0 = d/dt det H`` on the boundary, so eq. (9) is ``(theta2dot - theta1dot) sin 2(theta2 - theta1) >= 0``:
    no straightening at the outer boundary, no folding at the inner one, and the rotation about joint 1 is free."""
    arm = paper.arm()
    lo, hi = arm.safe_theta_diff_interval
    trace = arm.l1 ** 2 + arm.l2 ** 2 - 2 * arm.sigma_bar_sq
    for diff in (lo, hi):
        for theta1 in rng.uniform(-math.pi, math.pi, size=5):
            theta = np.array([theta1, theta1 + diff])
            for u in rng.normal(size=(5, 2)):
                ddet = arm.l1 ** 2 * arm.l2 ** 2 * math.sin(2 * diff) * (u[1] - u[0])
                assert trace * arm.margin_rate(theta, u) == pytest.approx(ddet, abs=1e-12)
            assert arm.margin_rate(theta, (0.3, 0.3)) == pytest.approx(0.0, abs=1e-12)
    assert math.sin(2 * lo) > 0 > math.sin(2 * hi)           # outer: theta2dot >= theta1dot; inner: <=


def test_rotation_keeps_the_eigenvalues_and_couples_kernel_and_range(symbolic, rng):
    """Rotating about joint 1 gives ``Hdot = v (S H - H S)``; on the boundary the eigenbasis blocks are
    ``Z00 = Z11 = 0`` and ``|Z01| = |b| = |v| tr H``."""
    s = symbolic
    S = sp.Matrix([[0, -1], [1, 0]])
    rot = {s["d1"]: s["v"], s["d2"]: s["v"]}
    assert sp.simplify(s["Hdot"].subs(rot) - s["v"] * (S * s["H"] - s["H"] * S)) == sp.zeros(2, 2)
    arm = paper.arm()
    lo, hi = arm.safe_theta_diff_interval
    trace = arm.l1 ** 2 + arm.l2 ** 2 - 2 * arm.sigma_bar_sq
    for diff in (lo, hi):
        theta = np.array([0.7, 0.7 + diff])
        lam, V = np.linalg.eigh(arm.H(theta))
        assert lam[0] == pytest.approx(0.0, abs=1e-12) and lam[1] == pytest.approx(trace, rel=1e-12)
        for v in (1.0, -0.6):
            Z = V.T @ arm.Hdot(theta, (v, v)) @ V
            assert abs(Z[0, 0]) < 1e-12 and abs(Z[1, 1]) < 1e-12
            assert abs(Z[0, 1]) == pytest.approx(abs(v) * trace, rel=1e-12)


def test_eq33_rotation_certificate(symbolic, rng):
    """``Omega = [[1/2, -theta1dot], [theta1dot, 1/2]]`` turns the LMI into ``H`` at every configuration,
    has ``sym(Omega) = I/2`` and ``||Omega - I/2||_F = sqrt 2 |theta1dot|``."""
    s = symbolic
    rot = {s["d1"]: s["v"], s["d2"]: s["v"]}
    Omega = sp.Matrix([[sp.Rational(1, 2), -s["v"]], [s["v"], sp.Rational(1, 2)]])
    lmi = s["Hdot"].subs(rot) + Omega.T * s["H"] + s["H"] * Omega
    assert sp.simplify(lmi - s["H"]) == sp.zeros(2, 2)
    assert (Omega + Omega.T) / 2 == sp.eye(2) / 2
    assert sp.simplify(sum((Omega - sp.eye(2) / 2)[i, j] ** 2 for i in range(2) for j in range(2))) == 2 * s["v"] ** 2
    arm = paper.arm()
    for theta in rng.uniform(-math.pi, math.pi, size=(5, 2)):
        for v in (1.0, -2.5):
            Om = np.array([[0.5, -v], [v, 0.5]])
            np.testing.assert_allclose(arm.Hdot(theta, (v, v)) + Om.T @ arm.H(theta) + arm.H(theta) @ Om,
                                       arm.H(theta), rtol=0, atol=1e-14)


def test_eq34_eq35_scalar_decay_at_the_boundary(symbolic, rng):
    """eq. (34) symbolically, and eq. (35) numerically: on the boundary, ``omega >= |Hdot|_F^2 / (2 d/dt det H)``
    is exactly where ``Hdot + omega H >= 0``, and for the rotation no ``omega`` works."""
    s = symbolic
    lhs = (s["Hdot"] + s["w"] * s["H"]).det()
    _zero(lhs - (s["w"] ** 2 * s["H"].det() + s["w"] * s["ddet"] - s["fro"] / 2))
    arm = paper.arm()
    lo, hi = arm.safe_theta_diff_interval
    for diff in (lo, hi):
        theta = np.array([-0.2, -0.2 + diff])
        H = arm.H(theta)
        checked = 0
        for u in rng.normal(size=(40, 2)):
            ddet = arm.l1 ** 2 * arm.l2 ** 2 * math.sin(2 * diff) * (u[1] - u[0])
            Hdot = arm.Hdot(theta, u)
            if ddet <= 1e-3:
                continue
            bound = np.sum(Hdot ** 2) / (2 * ddet)
            assert np.linalg.eigvalsh(Hdot + bound * (1 + 1e-6) * H)[0] >= -1e-12
            assert np.linalg.eigvalsh(Hdot + bound * (1 - 1e-6) * H)[0] < 0
            checked += 1
        assert checked >= 10
        Hdot = arm.Hdot(theta, (1.0, 1.0))
        for omega in (1.0, 1e2, 1e4, 1e6):                   # det H is zero up to rounding, which omega^2 amplifies
            assert np.linalg.det(Hdot + omega * H) == pytest.approx(-np.sum(Hdot ** 2) / 2,
                                                                    abs=1e-9 + omega ** 2 * 1e-15)
            assert np.linalg.eigvalsh(Hdot + omega * H)[0] < 0


def test_eq36_rotation_needs_scalar_decay_growing_with_the_inverse_root_of_det_H(symbolic, rng):
    s = symbolic
    rot = {s["d1"]: s["v"], s["d2"]: s["v"]}
    _zero((s["fro"] / 2).subs(rot) - s["v"] ** 2 * ((s["l1"] ** 2 - s["l2"] ** 2) ** 2
                                                    + 4 * s["l1"] ** 2 * s["l2"] ** 2 * sp.cos(s["D"]) ** 2))
    assert sp.simplify(s["ddet"].subs(rot)) == 0
    arm = paper.arm()
    lo, hi = arm.safe_theta_diff_interval
    for diff in rng.uniform(lo + 1e-3, hi - 1e-3, size=8):
        theta = np.array([0.3, 0.3 + diff])
        det = np.linalg.det(arm.H(theta))
        for v in (1.0, -0.6):
            needed = abs(v) * math.sqrt(((arm.l1 ** 2 - arm.l2 ** 2) ** 2
                                         + 4 * arm.l1 ** 2 * arm.l2 ** 2 * math.cos(diff) ** 2) / det)
            assert min_scalar_decay(arm.alpha_H(theta), arm.Hdot(theta, (v, v))) == pytest.approx(needed, rel=1e-9)


def test_eq37_rotation_cap_of_the_standard_mcbf(symbolic):
    """``det H / ((l1^2 - l2^2)^2 + 4 l1^2 l2^2 cos^2)`` decreases in ``cos^2``, so the cap is largest at
    ``theta2 - theta1 = pi/2``; on the hero arm it is 0.8229 rad/s, and the standard MCBF's LMI holds for
    the rotation exactly up to it."""
    s = symbolic
    c = sp.symbols("c", positive=True)
    ratio = (((s["l1"] ** 2 - s["sb"] ** 2) * (s["l2"] ** 2 - s["sb"] ** 2) - s["l1"] ** 2 * s["l2"] ** 2 * c)
             / ((s["l1"] ** 2 - s["l2"] ** 2) ** 2 + 4 * s["l1"] ** 2 * s["l2"] ** 2 * c))
    numerator = sp.factor(sp.together(sp.diff(ratio, c)).as_numer_denom()[0])
    assert numerator == -s["l1"] ** 2 * s["l2"] ** 2 * (s["l1"] ** 2 + s["l2"] ** 2 - 2 * s["sb"] ** 2) ** 2
    arm = paper.arm()
    cap = math.sqrt((arm.l1 ** 2 - arm.sigma_bar_sq) * (arm.l2 ** 2 - arm.sigma_bar_sq)) / abs(arm.l1 ** 2 - arm.l2 ** 2)
    assert round(cap, 4) == ROTATION_CAP
    theta = np.array([0.0, math.pi / 2])
    for v, certified in ((0.999 * cap, True), (1.001 * cap, False), (-0.999 * cap, True), (-1.001 * cap, False)):
        lmi = arm.Hdot(theta, (v, v)) + arm.alpha_H(theta)
        assert (np.linalg.eigvalsh(lmi)[0] >= -1e-12) == certified
    lo, _ = arm.safe_theta_diff_interval
    for diff in np.linspace(lo, math.pi - lo, 9):           # nowhere in the safe set does it certify more
        theta = np.array([0.0, diff])
        lmi = arm.Hdot(theta, (1.001 * cap, 1.001 * cap)) + arm.alpha_H(theta)
        assert np.linalg.eigvalsh(lmi)[0] < 0


# ------------------------------------------------------------ Sec. VIII-A, the settings
def test_settings_of_the_section():
    arm = paper.arm()
    assert (arm.l1, arm.l2, arm.sigma_bar, arm.gamma) == (1.0, 0.75, 0.6, 1.0)
    assert paper.TRACKING_L2 == (0.75, 1.0, 1.25) and len(paper.TRACKING_CASES) == 6
    assert (paper.GAIN, paper.DT) == (5.0, 0.01)
    assert paper.TIMELINE["rate"] == 1.0
    filt = SafetyFilter(2, 2, decay="matrix", **paper.FILTER)
    assert (filt.omega_d, filt.rho_omega, filt.rho_Omega, filt.beta, filt.recovery) == (1.0, 0.005, 0.01, 0.5, True)
    filt = SafetyFilter(2, 2, decay="scalar", **paper.FILTER)
    assert filt.recovery and filt.omega_d == 1.0
    assert CLARABEL_PARAMS["tol_gap_abs"] == CLARABEL_PARAMS["tol_gap_rel"] == CLARABEL_PARAMS["tol_feas"] == 1e-9
    theta0 = band_start(paper.arm(), **{k: paper.RECOVERY[k] for k in ("band", "depth")})
    np.testing.assert_allclose(theta0, [0.0, 0.8918], rtol=0, atol=5e-5)
    assert np.linalg.norm(paper.arm().end_effector(theta0)) == pytest.approx(1.5826, abs=5e-5)


# ------------------------------------------------------------ Sec. VIII-C, the closed loops
@pytest.fixture(scope="module")
def tracking():
    """The six tracking cases: ``(l2, band) -> (arm, target, traces)``."""
    out = {}
    for l2, band in paper.TRACKING_CASES:
        arm = paper.arm(l2)
        out[(l2, band)] = arm, paper.target(arm, band), paper.run_tracking(arm, band)
    return out


@pytest.fixture(scope="module")
def recovery():
    return paper.run_recovery_case()


def _summary(arm, trace, target) -> dict:
    held = paper.held_steps(trace, target)
    held_u = held[:-1]
    error = paper.tracking_error(arm, trace, target)
    bound = paper.least_distance(arm, target, trace.t)
    points = np.array([arm.end_effector(theta) for theta in trace.theta])
    angle = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    needed = trace.scalar_certificate[held_u]                # NaN where alpha(H) is not positive definite
    needed = needed[np.isfinite(needed)]
    return dict(
        distance=float(error[held].mean()),
        excess=float((error - bound)[held].mean()),
        margin=float(trace.margin[held].mean()),
        min_margin=float(trace.margin.min()),
        omega=float((2 * trace.Omega[held_u, 0, 0]).mean()),          # scalar OD: Omega = (omega/2) I
        median_decay_needed=float(np.median(needed)) if needed.size else math.nan,
        rotation=float(angle[-1] - angle[0]),
    )


def test_fig1_hero_run_numbers(tracking):
    """Fig. 1's numbers, held over ``1.75 <= t < 3.25`` s."""
    arm, target, traces = tracking[(HERO_L2, "outer")]
    assert paper.held_window(target) == (1.75, 3.25)
    assert target.r1 - arm.safe_radius_interval[1] == pytest.approx(HERO_LEAST_DISTANCE, abs=5e-5)
    summaries = {key: _summary(arm, trace, target) for key, trace in traces.items()}
    for key, distance in HERO_HELD_DISTANCE.items():
        assert summaries[key]["distance"] == pytest.approx(distance, abs=10 ** -len(str(distance).split(".")[1]))
    for key, margin in HERO_HELD_MARGIN.items():
        assert summaries[key]["margin"] == pytest.approx(margin, rel=1e-2)
    for key, rotation in HERO_ROTATION.items():
        assert summaries[key]["rotation"] == pytest.approx(rotation, abs=6e-3)
    assert summaries["scalar"]["omega"] == pytest.approx(HERO_OMEGA, abs=6e-3)
    assert summaries["nominal"]["rotation"] == pytest.approx(paper.TIMELINE["duration"], abs=1e-3)


def test_every_run_keeps_the_arm_safe_on_one_branch(tracking, recovery):
    """Sec. VIII-A: ``0 < theta2 - theta1 < pi`` in every run; Sec. VIII-C: every filter keeps ``lambda_min(H) > 0``."""
    for arm, _, traces in tracking.values():
        for key, trace in traces.items():
            diff = trace.theta[:, 1] - trace.theta[:, 0]
            assert np.all(diff > 0) and np.all(diff < math.pi), key
            if key != "nominal":
                assert trace.margin.min() > 0, key
                assert np.nanmin(trace.safety_residual) > -1e-7, key
        assert traces["nominal"].margin.min() < 0                   # the target does leave the safe set
    for trace in recovery.values():
        diff = trace.theta[:, 1] - trace.theta[:, 0]
        assert np.all(diff > 0) and np.all(diff < math.pi)


def test_tracking_order_and_the_quoted_ranges(tracking):
    """Sec. VIII-C: standard MCBF farthest from the target, then scalar OD, then matrix OD, in all six cases;
    the smallest scalar decay certifying matrix OD's inputs has a median of 24 to 36, a penalty distance
    ``||omega I/2 - Omega_d||_F = (omega - 1) / sqrt 2`` of 17 to 25, and scalar OD's mean omega is 6.3 to 8.3."""
    medians, omegas = [], []
    for (l2, band), (arm, target, traces) in tracking.items():
        s = {key: _summary(arm, trace, target) for key, trace in traces.items()}
        assert s["standard"]["excess"] > s["scalar"]["excess"] > s["matrix"]["excess"] > 0, (l2, band)
        assert s["standard"]["margin"] > s["scalar"]["margin"] > s["matrix"]["margin"] > 0, (l2, band)
        medians.append(s["matrix"]["median_decay_needed"])
        omegas.append(s["scalar"]["omega"])
    low, high = MEDIAN_DECAY_RANGE
    assert low <= min(medians) < low + 1 and high - 0.5 <= max(medians) < high + 0.5
    penalty = [(m - 1) / math.sqrt(2) for m in medians]
    assert (round(min(penalty)), round(max(penalty))) == PENALTY_DISTANCE_RANGE
    assert (round(min(omegas), 1), round(max(omegas), 1)) == MEAN_OMEGA_RANGE


def test_fig3_recovery(recovery):
    """From ``lambda_min(H) = -0.098`` with the target fixed, matrix OD recovers at exactly ``2 beta``:
    per step the margin shrinks by ``1 - 2 beta dt``, tracing ``lambda_min(H(0)) exp(-2 beta t)``;
    without the recovery constraint the arm stays in the unsafe region."""
    for key, trace in recovery.items():
        assert trace.margin[0] == pytest.approx(RECOVERY_MARGIN0, abs=5e-4)
        assert np.linalg.norm(trace.u_nom[0]) < 1e-12                 # the nominal command holds the arm
    dt = paper.RECOVERY["dt"]
    for beta in paper.RECOVERY["betas"]:
        trace = recovery[f"beta={beta:g}"]
        rate = 2 * beta * paper.GAMMA
        np.testing.assert_allclose(recovery_rate(trace), rate, rtol=2e-3)
        np.testing.assert_allclose(trace.margin[1:] / trace.margin[:-1], 1 - rate * dt, rtol=0, atol=1e-5)
        assert trace.margin[-1] / trace.margin[0] == pytest.approx(RECOVERY_LEFT[beta], rel=1e-2)
        assert np.all(trace.margin < 0) and np.all(np.diff(trace.margin) > 0)
    free = recovery[NO_RECOVERY]
    assert np.all(free.margin < 0)
    assert free.margin[-1] / free.margin[0] == pytest.approx(NO_RECOVERY_LEFT, abs=5e-3)
    assert recovery_rate(free)[-1] < 1e-6


def test_band_radius_rejects_an_unknown_band():
    with pytest.raises(ValueError, match="band must be one of"):
        paper.band_radius(paper.arm(), "middle")
