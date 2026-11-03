"""Unit tests of :mod:`odmcbf.filters`.

The filters are exercised mostly on the barrier data of the two-link arm
(``alpha_H`` and ``LgH`` at chosen configurations), plus small synthetic
instances with closed-form solutions. Every returned pair ``(u, Omega)`` is
checked against an independent NumPy recomputation of the barrier LMI.

Two devices recur:

* With ``LgH = 0`` the control drops out of the constraint, so the filter
  returns ``u_nom`` and only decides whether a decay multiplier certifies
  ``Hdot = LfH``. Passing ``LfH = Hdot(theta, u)`` therefore asks whether the
  command ``u`` is certified at ``theta``.
* Rotation of the whole arm about joint 1, ``u = (w, w)``, leaves the margin
  unchanged. At a boundary configuration no scalar decay certifies it, while
  a matrix multiplier does.

Solver resolution: CLARABEL runs with an absolute gap tolerance of ``1e-9``,
so objectives are resolved to about ``1e-9``. A decay variable resting at its
nominal value is then pinned only to about ``sqrt(1e-9 / rho)``, roughly
``5e-4`` for the default ``rho_omega = 5e-3``; the tolerances on ``omega`` and
``Omega`` in that situation are set accordingly.
"""

from __future__ import annotations

import dataclasses
import types

import cvxpy as cp
import numpy as np
import pytest
import scipy.linalg

from odmcbf import (
    DECAY_MODES,
    FilterInfeasible,
    FilterResult,
    SafetyFilter,
    TwoLinkArm,
    min_scalar_decay,
)
from odmcbf import filters as filters_module

#: Zero control directions for the two-link arm: the filter then only certifies ``LfH``.
ZERO_LGH = [np.zeros((2, 2)), np.zeros((2, 2))]

#: Feasibility slack allowed on the returned LMI and recovery constraint.
LMI_TOL = 1e-7


# ------------------------------------------------------------------ helpers
def _sym(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + X.T)


def _lambda_min(X: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(_sym(X))[0])


def _lmi_lambda_min(filt: SafetyFilter, res: FilterResult, alpha_H, LgH, LfH=None) -> float:
    """``lambda_min`` of the barrier LMI at the returned variables, recomputed from scratch.

    The decay term is rebuilt from the mode's own variable (``alpha_H`` for
    ``standard``, ``omega * alpha_H`` for ``scalar``, ``Omega.T A + A Omega``
    for ``matrix``), not from the ``Omega`` the filter reports.
    """
    p = filt.p
    Hdot = np.zeros((p, p)) if LfH is None else np.array(LfH, dtype=float)
    for u_j, G in zip(res.u, LgH):
        Hdot = Hdot + u_j * np.asarray(G)
    if filt.decay == "standard":
        decay_term = alpha_H
    elif filt.decay == "scalar":
        decay_term = res.omega * alpha_H
    else:
        decay_term = res.Omega.T @ alpha_H + alpha_H @ res.Omega
    return _lambda_min(Hdot + decay_term)


def _cost(filt: SafetyFilter, res: FilterResult, u_nom) -> float:
    """The filter's objective evaluated at the returned variables."""
    cost = float(np.sum((res.u - np.asarray(u_nom)) ** 2))
    if filt.decay == "scalar":
        cost += filt.rho_omega * (res.omega - filt.omega_d) ** 2
    elif filt.decay == "matrix":
        cost += filt.rho_Omega * float(
            np.sum((res.Omega - (filt.omega_d / 2) * np.eye(filt.p)) ** 2))
    return cost


def _arm_states(arm: TwoLinkArm, rng: np.random.Generator, n: int):
    """``n`` random ``(theta, u_nom)`` pairs, ``theta2 - theta1`` over the whole circle."""
    for _ in range(n):
        yield rng.uniform(-np.pi, np.pi, size=2), 1.5 * rng.normal(size=2)


def _safe_theta(arm: TwoLinkArm, rng: np.random.Generator, clearance: float = 0.0) -> np.ndarray:
    """Random configuration with ``lo + clearance <= |theta2 - theta1| <= hi - clearance``."""
    lo, hi = arm.safe_theta_diff_interval
    theta1 = rng.uniform(-np.pi, np.pi)
    theta_diff = rng.uniform(lo + clearance, hi - clearance) * rng.choice([-1.0, 1.0])
    return np.array([theta1, theta1 + theta_diff])


def _boundary_thetas(arm: TwoLinkArm, theta1: float):
    """The four boundary configurations ``theta2 - theta1 = +-lo, +-hi`` at a given ``theta1``."""
    lo, hi = arm.safe_theta_diff_interval
    for theta_diff in (lo, -lo, hi, -hi):
        yield np.array([theta1, theta1 + theta_diff])


def _unsafe_thetas(arm: TwoLinkArm, theta1: float):
    """Configurations outside the safe set, in both bands of ``theta2 - theta1``."""
    lo, hi = arm.safe_theta_diff_interval
    for theta_diff in (0.3 * lo, 0.8 * lo, -0.5 * lo,
                       hi + 0.5 * (np.pi - hi), -(hi + 0.2 * (np.pi - hi))):
        yield np.array([theta1, theta1 + theta_diff])


def _margin_rate(arm: TwoLinkArm, theta, u) -> float:
    """``d/dt lambda_min(H)`` under ``thetadot = u``, i.e. ``v0.T Hdot v0``."""
    v0 = np.linalg.eigh(arm.H(theta))[1][:, 0]
    return float(v0 @ arm.Hdot(theta, u) @ v0)


def _random_instance(rng: np.random.Generator, p: int, m: int):
    """Synthetic data with ``alpha_H`` positive definite and ``alpha_H + LfH > 0``.

    ``u = 0`` is then feasible for every mode, with or without recovery.
    """
    B = rng.normal(size=(p, p))
    alpha_H = B @ B.T / p + np.eye(p)
    LfH = _sym(rng.normal(size=(p, p)))
    LfH *= 0.5 / np.linalg.norm(LfH, 2)
    LgH = [_sym(rng.normal(size=(p, p))) for _ in range(m)]
    return alpha_H, LgH, LfH, rng.normal(size=m)


# ------------------------------------------------------------- construction
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_every_decay_mode_builds(decay):
    filt = SafetyFilter(3, 2, decay=decay)
    assert (filt.decay, filt.m, filt.p) == (decay, 3, 2)
    assert DECAY_MODES == ("standard", "scalar", "matrix")


@pytest.mark.parametrize("decay", ["Matrix", "fixed", "", None])
def test_rejects_unknown_decay(decay):
    with pytest.raises(ValueError, match="decay must be one of"):
        SafetyFilter(2, 2, decay=decay)


@pytest.mark.parametrize("omega_d", [0.0, -1.0])
def test_rejects_nonpositive_omega_d(omega_d):
    with pytest.raises(ValueError, match="omega_d must be positive"):
        SafetyFilter(2, 2, omega_d=omega_d)


@pytest.mark.parametrize("rho_omega", [0.0, -1e-3])
def test_rejects_nonpositive_rho_omega(rho_omega):
    with pytest.raises(ValueError, match="rho_omega must be positive"):
        SafetyFilter(2, 2, rho_omega=rho_omega)


@pytest.mark.parametrize("rho_omega", [5e-3, 0.3])
@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_matrix_penalty_is_matched_to_the_scalar_one(p, rho_omega):
    """``rho_Omega = 4 rho_omega / p``, so both penalties agree on ``Omega = (omega / 2) I``."""
    filt = SafetyFilter(2, p, decay="matrix", rho_omega=rho_omega)
    assert filt.rho_Omega == pytest.approx(4 * rho_omega / p, rel=1e-15)
    omega = 1.7
    Omega = (omega / 2) * np.eye(p)
    matrix_penalty = filt.rho_Omega * np.sum((Omega - (filt.omega_d / 2) * np.eye(p)) ** 2)
    assert matrix_penalty == pytest.approx(rho_omega * (omega - filt.omega_d) ** 2, rel=1e-12)


@pytest.mark.parametrize("omega_d", [1.0, 0.4, 3.0])
def test_beta_defaults_to_half_omega_d(omega_d):
    assert SafetyFilter(2, 2, omega_d=omega_d).beta == pytest.approx(omega_d / 2, rel=1e-15)
    assert SafetyFilter(2, 2, omega_d=omega_d, beta=0.1).beta == 0.1


def test_default_solver_is_clarabel_without_mosek(monkeypatch):
    """MOSEK is used when installed, else CLARABEL, each with its tight default options."""
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "SCS"])
    filt = SafetyFilter(2, 2)
    assert filt.solver == "CLARABEL"
    assert filt.solver_opts == filters_module.CLARABEL_PARAMS
    filt.solver_opts["max_iter"] = 1                # a copy, not the module default
    assert filters_module.CLARABEL_PARAMS["max_iter"] == 200

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "MOSEK"])
    filt = SafetyFilter(2, 2)
    assert filt.solver == "MOSEK"
    assert filt.solver_opts == {"mosek_params": filters_module.MOSEK_PARAMS}


def test_explicit_solver_options_reach_the_solve(monkeypatch):
    """Explicit ``solver`` and ``solver_opts`` are kept as given and passed to every solve."""
    filt = SafetyFilter(2, 2, solver="CLARABEL", solver_opts={"max_iter": 50})
    assert (filt.solver, filt.solver_opts) == ("CLARABEL", {"max_iter": 50})
    calls = []
    solve = filt._problem.solve

    def recording_solve(*args, **kwargs):
        calls.append(kwargs)
        return solve(*args, **kwargs)

    monkeypatch.setattr(filt._problem, "solve", recording_solve)
    filt(**_GOOD)
    filt(**_GOOD)
    assert calls == [{"solver": "CLARABEL", "max_iter": 50}] * 2


@pytest.mark.skipif("SCS" not in cp.installed_solvers(), reason="SCS is not installed")
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_another_solver_without_default_options(decay):
    """A solver with no entry in ``DEFAULT_SOLVER_OPTS`` runs with its own defaults.

    SCS is a first-order solver with a default accuracy near ``1e-4``, so the
    residuals only hold to that level, and the answer matches CLARABEL's to it.
    """
    arm = TwoLinkArm()
    theta, u_nom = np.array([0.3, 0.3 + 1.2]), np.array([1.0, 0.4])
    filt = SafetyFilter(2, 2, decay=decay, solver="SCS")
    assert filt.solver_opts == {}
    res = filt(arm.alpha_H(theta), arm.LgH(theta), u_nom)
    reference = SafetyFilter(2, 2, decay=decay)(arm.alpha_H(theta), arm.LgH(theta), u_nom)
    assert res.safety_residual >= -1e-3
    np.testing.assert_allclose(res.u, reference.u, rtol=0, atol=1e-2)


_GOOD = dict(alpha_H=np.eye(2), LgH=ZERO_LGH, u_nom=np.zeros(2), LfH=None)


@pytest.mark.parametrize("bad, match", [
    (dict(alpha_H=np.eye(3)), "alpha_H and LfH must be 2x2"),
    (dict(alpha_H=np.ones(2)), "alpha_H and LfH must be 2x2"),
    (dict(LfH=np.eye(3)), "alpha_H and LfH must be 2x2"),
    (dict(LgH=[np.eye(2)]), "LgH must hold 2 matrices"),
    (dict(LgH=[np.eye(2)] * 3), "LgH must hold 2 matrices"),
    (dict(LgH=[np.eye(2), np.eye(3)]), "LgH must hold 2 matrices"),
    (dict(u_nom=np.zeros(3)), r"u_nom must have shape \(2,\)"),
    (dict(u_nom=np.zeros((2, 1))), r"u_nom must have shape \(2,\)"),
    (dict(u_nom=0.0), r"u_nom must have shape \(2,\)"),
], ids=["alpha_H-3x3", "alpha_H-1d", "LfH-3x3", "LgH-short", "LgH-long",
        "LgH-entry-3x3", "u_nom-3", "u_nom-2x1", "u_nom-scalar"])
def test_call_validates_shapes(bad, match):
    filt = SafetyFilter(2, 2)
    with pytest.raises(ValueError, match=match):
        filt(**{**_GOOD, **bad})


# ------------------------------------------------------------------ results
def test_result_is_frozen():
    res = SafetyFilter(2, 2)(**_GOOD)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.u = np.ones(2)


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_result_fields_per_mode(decay, recovery):
    """``Omega`` is ``I/2`` (standard), ``(omega/2) I`` (scalar) or the solved multiplier (matrix)."""
    arm = TwoLinkArm()
    theta = np.array([0.3, 0.3 + 0.9])
    filt = SafetyFilter(2, 2, decay=decay, recovery=recovery)
    res = filt(arm.alpha_H(theta), arm.LgH(theta), np.array([1.0, 0.4]))

    assert res.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert res.u.shape == (2,) and res.u.dtype == float
    assert res.Omega.shape == (2, 2)
    assert np.isfinite(res.solve_time) and res.solve_time >= 0
    if decay == "standard":
        assert res.omega is None
        np.testing.assert_array_equal(res.Omega, 0.5 * np.eye(2))
    elif decay == "scalar":
        assert isinstance(res.omega, float) and res.omega >= 0
        np.testing.assert_allclose(res.Omega, 0.5 * res.omega * np.eye(2), rtol=0, atol=1e-15)
    else:
        assert res.omega is None
    if recovery and decay != "standard":
        assert isinstance(res.recovery_residual, float)
    else:
        assert res.recovery_residual is None


# ------------------------------------------- already-safe commands pass unchanged
@pytest.mark.parametrize("omega_d", [1.0, 2.0])
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_safe_command_passes_unchanged(arm, rng, decay, omega_d):
    """If ``Hdot(u_nom) + alpha(H) > 0``, every mode returns ``u_nom`` at zero cost.

    The decay sits at its nominal value, ``omega = omega_d`` or
    ``Omega = (omega_d / 2) I``, to the solver's resolution of the flat penalty.
    On the safe set ``alpha(H) >= 0``, so ``omega_d >= 1`` keeps the nominal
    decay feasible.
    """
    filt = SafetyFilter(2, 2, decay=decay, omega_d=omega_d)
    checked = 0
    while checked < 6:
        theta = _safe_theta(arm, rng)
        u_nom = rng.normal(size=2)
        if _lambda_min(arm.Hdot(theta, u_nom) + arm.alpha_H(theta)) < 1e-2:
            continue
        checked += 1
        res = filt(arm.alpha_H(theta), arm.LgH(theta), u_nom)
        np.testing.assert_allclose(res.u, u_nom, rtol=0, atol=1e-7)
        assert abs(res.objective) <= 1e-8
        if decay == "scalar":
            assert res.omega == pytest.approx(filt.omega_d, abs=1e-3)
        elif decay == "matrix":
            np.testing.assert_allclose(res.Omega, (omega_d / 2) * np.eye(2), rtol=0, atol=1e-3)


# ------------------------------------------------------- LMI and recovery
@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_returned_pair_satisfies_the_lmi(arm, rng, decay, recovery):
    """The returned ``(u, Omega)`` satisfies the barrier LMI and the recovery constraint.

    ``safety_residual`` equals an independent recomputation of
    ``lambda_min(Hdot + decay term)``, and ``objective`` equals the cost at the
    returned variables. States are drawn over the whole circle of
    ``theta2 - theta1``, so both safe and unsafe states are covered.
    """
    filt = SafetyFilter(2, 2, decay=decay, recovery=recovery)
    margins = []
    for theta, u_nom in _arm_states(arm, rng, 12):
        alpha_H, LgH = arm.alpha_H(theta), arm.LgH(theta)
        res = filt(alpha_H, LgH, u_nom)
        margins.append(arm.margin(theta))

        assert res.safety_residual >= -LMI_TOL
        assert res.safety_residual == pytest.approx(
            _lmi_lambda_min(filt, res, alpha_H, LgH), abs=1e-12)
        assert res.objective == pytest.approx(_cost(filt, res, u_nom), rel=1e-6, abs=1e-9)
        if recovery and decay == "scalar":
            assert res.omega >= filt.omega_d - LMI_TOL
            assert res.recovery_residual == pytest.approx(res.omega - filt.omega_d, abs=1e-12)
        elif recovery and decay == "matrix":
            lam = _lambda_min(res.Omega)
            assert lam >= filt.beta - LMI_TOL
            assert res.recovery_residual == pytest.approx(lam - filt.beta, abs=1e-12)
        else:
            assert res.recovery_residual is None
    assert min(margins) < 0 < max(margins)


@pytest.mark.parametrize("decay", DECAY_MODES)
def test_general_dimensions_with_drift(rng, decay):
    """``p = 3``, ``m = 4`` with a nonzero drift: the LMI holds and the objective matches."""
    p, m = 3, 4
    filt = SafetyFilter(m, p, decay=decay)
    for _ in range(8):
        alpha_H, LgH, LfH, u_nom = _random_instance(rng, p, m)
        u_nom *= 3.0
        res = filt(alpha_H, LgH, u_nom, LfH=LfH)
        assert res.u.shape == (m,) and res.Omega.shape == (p, p)
        assert res.safety_residual >= -LMI_TOL
        assert res.safety_residual == pytest.approx(
            _lmi_lambda_min(filt, res, alpha_H, LgH, LfH), abs=1e-12)
        assert res.objective == pytest.approx(_cost(filt, res, u_nom), rel=1e-6, abs=1e-9)
        if decay == "matrix":
            assert _lambda_min(res.Omega) >= filt.beta - LMI_TOL


@pytest.mark.parametrize("decay", DECAY_MODES)
def test_only_the_symmetric_part_of_the_data_matters(rng, decay):
    """Skew-symmetric parts of ``alpha_H``, ``LfH`` and ``LgH`` change neither the solution nor residual.

    The LMI constrains ``sym(Hdot + decay term)``, and
    ``sym(Omega.T A + A Omega) = Omega.T sym(A) + sym(A) Omega``, so only the
    symmetric parts of the data enter, both in the SDP and in ``safety_residual``.
    The data lie on a dyadic grid, so ``sym(A + K) = A`` holds exactly in floating
    point and the perturbed solve must reproduce the unperturbed one. Each solve
    uses a fresh filter: CVXPY's first solve and its later, cached solves of the
    same data can differ at the solver's resolution.
    """
    p, m = 3, 2
    filt = SafetyFilter(m, p, decay=decay)

    def solve(*args, **kwargs) -> FilterResult:
        return SafetyFilter(m, p, decay=decay)(*args, **kwargs)

    def grid(X: np.ndarray) -> np.ndarray:
        return np.round(X * 2.0 ** 20) / 2.0 ** 20

    def skew() -> np.ndarray:
        B = grid(rng.normal(size=(p, p)))
        return B - B.T

    for _ in range(4):
        alpha_H, LgH, LfH, u_nom = _random_instance(rng, p, m)
        alpha_H, LfH, LgH = grid(alpha_H), grid(LfH), [grid(G) for G in LgH]
        u_nom *= 3.0
        ref = solve(alpha_H, LgH, u_nom, LfH=LfH)
        res = solve(alpha_H + skew(), [G + skew() for G in LgH], u_nom, LfH=LfH + skew())
        assert ref.objective > 1e-5                  # a constraint binds
        np.testing.assert_allclose(res.u, ref.u, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res.Omega, ref.Omega, rtol=0, atol=1e-12)
        assert res.objective == pytest.approx(ref.objective, rel=1e-12, abs=1e-15)
        assert res.safety_residual == pytest.approx(
            _lmi_lambda_min(filt, res, alpha_H, LgH, LfH), abs=1e-12)


@pytest.mark.parametrize("gamma", [1.0, 2.5])
@pytest.mark.parametrize("omega_d", [1.0, 2.0])
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_recovery_raises_a_negative_margin(arm, decay, omega_d, gamma):
    """Outside the safe set the filtered command raises the margin at a guaranteed rate.

    With ``alpha(H) = gamma H`` and ``H v0 = margin v0``, the LMI gives
    ``d/dt margin = v0.T Hdot v0 >= rate * gamma * |margin|`` with ``rate = 1``
    (standard), ``omega >= omega_d`` (scalar) and ``2 v0.T Omega v0 >= 2 beta =
    omega_d`` (matrix).
    """
    arm = TwoLinkArm(arm.l1, arm.l2, gamma=gamma, sigma_bar=arm.sigma_bar)
    filt = SafetyFilter(2, 2, decay=decay, omega_d=omega_d)
    rate = 1.0 if decay == "standard" else omega_d
    for theta in _unsafe_thetas(arm, theta1=0.7):
        margin = arm.margin(theta)
        assert margin < 0
        res = filt(arm.alpha_H(theta), arm.LgH(theta), np.array([1.0, 0.4]))
        assert _margin_rate(arm, theta, res.u) >= rate * gamma * abs(margin) - LMI_TOL


@pytest.mark.parametrize("beta", [0.25, 0.75])
def test_recovery_constraint_uses_beta(beta):
    """The matrix recovery constraint is ``sym(Omega) >= beta I`` and binds outside the safe set."""
    arm = TwoLinkArm()
    filt = SafetyFilter(2, 2, decay="matrix", beta=beta)
    u_nom = np.array([0.3, -0.2])
    unsafe = np.array([0.1, 0.1 + np.pi / 5])
    safe = np.array([0.1, 0.1 + np.pi / 2])
    res = filt(arm.alpha_H(unsafe), arm.LgH(unsafe), u_nom)
    assert _lambda_min(res.Omega) == pytest.approx(beta, abs=1e-6)
    res = filt(arm.alpha_H(safe), arm.LgH(safe), u_nom)
    assert _lambda_min(res.Omega) == pytest.approx(max(beta, 0.5), abs=1e-3)


@pytest.mark.parametrize("beta", [0.25, 0.75])
def test_recovery_residual_is_the_slack_of_the_beta_constraint(beta):
    """``recovery_residual`` is the slack ``lambda_min(sym(Omega)) - beta`` of the imposed constraint.

    Measured against ``omega_d / 2`` instead, it would read ``-0.25`` at
    ``beta = 0.25``, where the constraint binds, and ``+0.25`` at
    ``beta = 0.75``.
    """
    arm = TwoLinkArm()
    theta = np.array([0.1, 0.1 + np.pi / 5])
    res = SafetyFilter(2, 2, decay="matrix", beta=beta)(
        arm.alpha_H(theta), arm.LgH(theta), np.array([0.3, -0.2]))
    assert res.recovery_residual == pytest.approx(_lambda_min(res.Omega) - beta, abs=1e-9)


# --------------------------------------------------------------- invasiveness
@pytest.mark.parametrize("recovery", [True, False])
def test_invasiveness_ordering(arm, rng, recovery):
    """With matched penalties and ``omega_d = 1``: ``J_matrix <= J_scalar <= J_standard``."""
    filters = {decay: SafetyFilter(2, 2, decay=decay, recovery=recovery) for decay in DECAY_MODES}
    for theta, u_nom in _arm_states(arm, rng, 12):
        J = {decay: filt(arm.alpha_H(theta), arm.LgH(theta), u_nom).objective
             for decay, filt in filters.items()}
        assert J["matrix"] <= J["scalar"] + 1e-9 + 1e-6 * J["scalar"]
        assert J["scalar"] <= J["standard"] + 1e-9 + 1e-6 * J["standard"]


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("omega_d", [0.5, 2.0])
def test_matrix_no_more_invasive_than_scalar_for_any_omega_d(rng, omega_d, recovery):
    """``J_matrix <= J_scalar`` holds for any ``omega_d`` and ``p``, and smaller ``beta`` only helps.

    The scalar problem is the matrix one restricted to ``Omega = (omega / 2) I``
    (with ``beta = omega_d / 2`` matching the scalar recovery ``omega >=
    omega_d``). Checked with ``p = 3``, where the matched penalty depends on ``p``.
    """
    p, m = 3, 2
    kwargs = dict(omega_d=omega_d, recovery=recovery)
    scalar = SafetyFilter(m, p, decay="scalar", **kwargs)
    matrix = SafetyFilter(m, p, decay="matrix", **kwargs)
    loose = SafetyFilter(m, p, decay="matrix", beta=omega_d / 4, **kwargs)
    for _ in range(8):
        alpha_H, LgH, LfH, u_nom = _random_instance(rng, p, m)
        LfH = 4.0 * LfH - 2.0 * alpha_H           # make the constraint active
        J_scalar = scalar(alpha_H, LgH, u_nom, LfH=LfH).objective
        J_matrix = matrix(alpha_H, LgH, u_nom, LfH=LfH).objective
        J_loose = loose(alpha_H, LgH, u_nom, LfH=LfH).objective
        assert J_scalar > 1e-4
        assert J_matrix <= J_scalar + 1e-9 + 1e-6 * J_scalar
        assert J_loose <= J_matrix + 1e-9 + 1e-6 * J_matrix


def test_standard_filter_ignores_omega_d(rng):
    """The standard condition has fixed decay ``Hdot + alpha(H) >= 0``, whatever ``omega_d`` is.

    This is why ``J_scalar <= J_standard`` needs ``omega_d = 1``.
    """
    arm = TwoLinkArm(1.0, 0.6, sigma_bar=0.3)
    reference = SafetyFilter(2, 2, decay="standard")
    others = [SafetyFilter(2, 2, decay="standard", omega_d=w) for w in (0.5, 2.0)]
    for theta, u_nom in _arm_states(arm, rng, 5):
        ref = reference(arm.alpha_H(theta), arm.LgH(theta), u_nom)
        for filt in others:
            res = filt(arm.alpha_H(theta), arm.LgH(theta), u_nom)
            np.testing.assert_allclose(res.u, ref.u, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(res.Omega, 0.5 * np.eye(2))


@pytest.mark.parametrize("recovery", [True, False])
def test_scalar_and_matrix_coincide_in_one_dimension(rng, recovery):
    """For ``p = 1`` the matrix multiplier is ``omega / 2`` and the two problems are the same.

    The matrix multiplier is unsigned while ``omega >= 0``, so without recovery
    the problems coincide only where ``alpha(H) > 0`` (a negative multiplier
    never helps there); with recovery both are bounded below alike. With a
    nonzero control direction both are always feasible.
    """
    scalar = SafetyFilter(2, 1, decay="scalar", recovery=recovery, rho_omega=0.1)
    matrix = SafetyFilter(2, 1, decay="matrix", recovery=recovery, rho_omega=0.1)
    active = 0
    for _ in range(8):
        alpha_H = np.array([[rng.uniform(-1.0 if recovery else 0.05, 1.0)]])
        LgH = [np.array([[g]]) for g in rng.normal(size=2)]
        LfH = np.array([[rng.normal() - 1.0]])
        u_nom = rng.normal(size=2)
        res_s = scalar(alpha_H, LgH, u_nom, LfH=LfH)
        res_m = matrix(alpha_H, LgH, u_nom, LfH=LfH)
        active += res_s.objective > 1e-4
        np.testing.assert_allclose(res_m.u, res_s.u, rtol=0, atol=1e-6)
        assert res_m.objective == pytest.approx(res_s.objective, rel=1e-6, abs=1e-9)
        assert res_m.Omega[0, 0] == pytest.approx(res_s.omega / 2, abs=1e-4)
    assert active >= 4


def _one_dimensional_optimum(decay, a, g, lf, u_nom, omega_d, rho, recovery):
    """Closed-form minimiser of the ``p = m = 1`` filter: a weighted projection onto a half-plane.

    The constraint is ``lf + g u + a omega >= 0`` (``omega = 1`` for standard).
    Returns ``(u, omega, objective)``.
    """
    if decay == "standard":
        violation = max(0.0, -(lf + g * u_nom + a))
        return u_nom + violation / g, None, (violation / g) ** 2
    violation = max(0.0, -(lf + g * u_nom + a * omega_d))
    denom = g ** 2 + a ** 2 / rho
    t = violation / denom
    omega = omega_d + a * t / rho
    if recovery and omega < omega_d:
        return u_nom + violation / g, omega_d, (violation / g) ** 2
    if decay == "scalar" and omega < 0:                 # omega >= 0; the matrix multiplier is unsigned
        violation = max(0.0, -(lf + g * u_nom))
        return u_nom + violation / g, 0.0, (violation / g) ** 2 + rho * omega_d ** 2
    return u_nom + g * t, omega, violation ** 2 / denom


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("decay", DECAY_MODES)
@pytest.mark.parametrize("a, lf", [(0.5, -3.0), (0.5, 0.0), (-0.5, 0.0), (-2.0, -3.0)],
                         ids=["safe-drift", "safe-inactive", "unsafe", "unsafe-deep"])
def test_one_dimensional_closed_form(decay, recovery, a, lf):
    """``p = m = 1`` against the closed-form projection; the drift ``lf`` is honoured.

    ``safe-drift`` is active only because of the drift ``lf = -3``;
    ``safe-inactive`` is the same instance without it. In ``unsafe``
    (``alpha(H) < 0``) the unconstrained optimum lowers the decay below
    ``omega_d``, which the recovery constraint forbids. In ``unsafe-deep`` it
    would be negative: without recovery the scalar decay stops at
    ``omega = 0``, while the matrix multiplier, which is unsigned, does not.
    """
    g, u_nom, omega_d, rho = 2.0, 0.2, 1.0, 0.1
    u_star, omega_star, J_star = _one_dimensional_optimum(
        decay, a, g, lf, u_nom, omega_d, rho, recovery)
    filt = SafetyFilter(1, 1, decay=decay, omega_d=omega_d, rho_omega=rho, recovery=recovery)
    res = filt(np.array([[a]]), [np.array([[g]])], np.array([u_nom]), LfH=np.array([[lf]]))
    assert res.u[0] == pytest.approx(u_star, abs=1e-6)
    assert res.objective == pytest.approx(J_star, rel=1e-6, abs=1e-9)
    if decay == "scalar":
        assert res.omega == pytest.approx(omega_star, abs=2e-4)
    elif decay == "matrix":
        assert res.Omega[0, 0] == pytest.approx(omega_star / 2, abs=2e-4)
    if (a, lf) == (0.5, -3.0):
        assert res.u[0] > u_nom + 0.5                  # moved by the drift


@pytest.mark.parametrize("decay", DECAY_MODES)
def test_drift_is_equivalent_to_shifting_the_command(rng, decay):
    """A drift ``LfH = sum_j c_j LgH_j`` acts like adding ``c`` to the command.

    ``Hdot = sum_j (u_j + c_j) LgH_j``, so the filtered command with that drift
    is the filtered ``u_nom + c`` without it, shifted back by ``c``.
    """
    arm = TwoLinkArm(1.0, 0.6, sigma_bar=0.3)
    filt = SafetyFilter(2, 2, decay=decay)
    for theta, u_nom in _arm_states(arm, rng, 6):
        c = rng.normal(size=2)
        G1, G2 = arm.LgH(theta)
        with_drift = filt(arm.alpha_H(theta), [G1, G2], u_nom, LfH=c[0] * G1 + c[1] * G2)
        shifted = filt(arm.alpha_H(theta), [G1, G2], u_nom + c)
        np.testing.assert_allclose(with_drift.u, shifted.u - c, rtol=0, atol=1e-5)
        assert with_drift.objective == pytest.approx(shifted.objective, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("decay", DECAY_MODES)
def test_rotation_about_joint_1_leaves_the_filtered_command_unchanged(arm, rng, decay):
    """Rotating the whole arm about joint 1 by ``t`` conjugates ``alpha_H`` and ``LgH`` by ``R(t)``.

    The filter is invariant under that congruence: the same ``u`` and objective,
    and ``Omega`` conjugated by ``R(t)``. The objective is resolved to about
    ``1e-9`` and has curvature 2 in ``u``, so ``u`` is resolved only to about
    ``sqrt(1e-9)``; with cvxpy 1.5 the two solves differ by ``1.6e-5``.
    """
    filt = SafetyFilter(2, 2, decay=decay)
    for theta, u_nom in _arm_states(arm, rng, 4):
        t = rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        rotated = theta + t
        np.testing.assert_allclose(arm.alpha_H(rotated), R @ arm.alpha_H(theta) @ R.T,
                                   rtol=0, atol=1e-12)
        res = filt(arm.alpha_H(theta), arm.LgH(theta), u_nom)
        res_rot = filt(arm.alpha_H(rotated), arm.LgH(rotated), u_nom)
        np.testing.assert_allclose(res_rot.u, res.u, rtol=0, atol=1e-4)
        assert res_rot.objective == pytest.approx(res.objective, rel=1e-6, abs=1e-7)
        np.testing.assert_allclose(res_rot.Omega, R @ res.Omega @ R.T, rtol=0, atol=1e-3)


# ------------------------------------------------- the gap at the boundary
CERT_COMMANDS = [(1.0, 1.0), (-0.7, -0.7), (0.0, 1.0), (-0.5, 0.5), (1.0, 0.0),
                 (0.3, -0.4), (2.0, 2.5), (0.0, 0.0)]


@pytest.mark.parametrize("recovery", [True, False])
def test_matrix_certifies_exactly_the_non_decreasing_commands_at_the_boundary(arm, recovery):
    """At a boundary configuration the matrix LMI certifies ``u`` iff ``(u2 - u1) sin 2(theta2 - theta1) >= 0``.

    That includes rotation of the whole arm about joint 1 (``u1 = u2``).
    """
    filt = SafetyFilter(2, 2, decay="matrix", recovery=recovery)
    for theta in _boundary_thetas(arm, theta1=0.3):
        assert abs(arm.margin(theta)) < 1e-12
        for u in CERT_COMMANDS:
            sign = (u[1] - u[0]) * np.sin(2 * (theta[1] - theta[0]))
            if sign >= 0:
                res = filt(arm.alpha_H(theta), ZERO_LGH, np.zeros(2), LfH=arm.Hdot(theta, u))
                assert res.safety_residual >= -LMI_TOL
            else:
                with pytest.raises(FilterInfeasible, match="matrix filter"):
                    filt(arm.alpha_H(theta), ZERO_LGH, np.zeros(2), LfH=arm.Hdot(theta, u))


@pytest.mark.parametrize("recovery", [True, False])
def test_scalar_certifies_exactly_the_margin_increasing_commands_at_the_boundary(arm, recovery):
    """Away from rotation of the whole arm about joint 1, the scalar LMI certifies ``u`` iff
    ``(u2 - u1) sin 2(theta2 - theta1) > 0`` or ``u = 0``."""
    filt = SafetyFilter(2, 2, decay="scalar", recovery=recovery)
    for theta in _boundary_thetas(arm, theta1=0.3):
        for u in CERT_COMMANDS:
            if u[0] == u[1] and u[0] != 0:
                continue                     # rotation: see the next test
            sign = (u[1] - u[0]) * np.sin(2 * (theta[1] - theta[0]))
            if sign > 0 or u == (0.0, 0.0):
                res = filt(arm.alpha_H(theta), ZERO_LGH, np.zeros(2), LfH=arm.Hdot(theta, u))
                assert res.safety_residual >= -LMI_TOL
            else:
                with pytest.raises(FilterInfeasible, match="scalar filter"):
                    filt(arm.alpha_H(theta), ZERO_LGH, np.zeros(2), LfH=arm.Hdot(theta, u))


def test_scalar_filter_rejects_rotation_at_the_boundary():
    """No scalar decay certifies rotation of the whole arm about joint 1 at the boundary.

    At the default arm's boundary configurations, e.g. ``theta = (0, pi/3)``,
    the margin is zero and the scalar certification problem is (weakly)
    infeasible. The filter either reports it as :class:`FilterInfeasible` or,
    when CLARABEL fails outright, passes ``cvxpy.error.SolverError`` through.
    The solver is pinned because that failure mode is CLARABEL's.
    """
    arm = TwoLinkArm()
    filt = SafetyFilter(2, 2, decay="scalar", solver="CLARABEL")
    for theta in _boundary_thetas(arm, theta1=0.0):
        assert abs(arm.margin(theta)) < 1e-15
        with pytest.raises((FilterInfeasible, cp.error.SolverError)):
            filt(arm.alpha_H(theta), ZERO_LGH, np.zeros(2), LfH=arm.Hdot(theta, (1.0, 1.0)))


def test_solver_failure_passes_through(monkeypatch):
    """A solver that fails outright raises ``cvxpy.error.SolverError``, unchanged.

    The solver is replaced by one that fails the way CLARABEL does above, so
    this checks the behaviour independently of which solver is installed.
    """
    filt = SafetyFilter(2, 2)

    def failing_solve(*args, **kwargs):
        raise cp.error.SolverError("Solver 'CLARABEL' failed.")

    monkeypatch.setattr(filt._problem, "solve", failing_solve)
    with pytest.raises(cp.error.SolverError, match="Solver 'CLARABEL' failed"):
        filt(**_GOOD)


@pytest.mark.parametrize("status", [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE, cp.UNBOUNDED,
                                    cp.UNBOUNDED_INACCURATE, cp.settings.INFEASIBLE_OR_UNBOUNDED,
                                    cp.USER_LIMIT, cp.SOLVER_ERROR])
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_every_non_optimal_status_raises(monkeypatch, decay, status):
    """Only ``optimal`` and ``optimal_inaccurate`` are accepted; any other status raises.

    The message names the mode and the status. The problem is stubbed so that
    each status can be produced on demand.
    """
    filt = SafetyFilter(2, 2, decay=decay)
    stub = types.SimpleNamespace(status=status, value=None, solver_stats=None,
                                 solve=lambda *args, **kwargs: None)
    monkeypatch.setattr(filt, "_problem", stub)
    with pytest.raises(FilterInfeasible, match=f"^{decay} filter: {status}$"):
        filt(**_GOOD)


@pytest.mark.parametrize("omega_d", [1.0, 2.0])
@pytest.mark.parametrize("gamma", [1.0, 2.0])
@pytest.mark.parametrize("recovery", [True, False])
def test_cost_of_certifying_rotation_at_the_boundary(arm, recovery, gamma, omega_d):
    """Cheapest matrix multiplier certifying ``u = (w, w)`` at a boundary configuration.

    In the eigenbasis of ``H`` (kernel ``v0``, ``alpha(H) v1 = gamma h1 v1``) the
    rotation gives ``Hdot = w h1 (v0 v1.T + v1 v0.T)`` up to sign, so the
    multiplier needs ``v1.T Omega v0 = -+ w / gamma``. The cheapest completion
    costs ``rho_Omega (w / gamma)**2`` without recovery and ``(4/3)`` times that
    with ``sym(Omega) >= (omega_d / 2) I``, independent of ``omega_d``. The LMI
    has zero slack along ``v0``, which limits the attainable accuracy to ``2e-3``.
    """
    arm = TwoLinkArm(arm.l1, arm.l2, gamma=gamma, sigma_bar=arm.sigma_bar)
    filt = SafetyFilter(2, 2, decay="matrix", recovery=recovery, omega_d=omega_d)
    factor = 4.0 / 3.0 if recovery else 1.0
    for theta in _boundary_thetas(arm, theta1=0.3):
        for w in (1.0, -0.6):
            res = filt(arm.alpha_H(theta), ZERO_LGH, np.zeros(2), LfH=arm.Hdot(theta, (w, w)))
            assert res.objective == pytest.approx(factor * filt.rho_Omega * (w / gamma) ** 2,
                                                  rel=2e-3)


def test_rotation_about_joint_1_at_the_boundary(arm):
    """The paper's gap at ``theta2 - theta1 = lo`` with ``u_nom = (1, 1)``.

    Rotation of the whole arm about joint 1 keeps the margin at zero. The
    matrix LMI certifies it at cost ``J_cert = (4/3) rho_Omega``, so the matrix
    filter costs at most that and moves ``u`` by at most ``sqrt(J_cert)``. No
    scalar decay certifies it: the scalar and standard filters cost more and
    must turn it into a margin-increasing command.
    """
    lo, _ = arm.safe_theta_diff_interval
    theta = np.array([0.2, 0.2 + lo])
    u_nom = np.array([1.0, 1.0])
    res = {decay: SafetyFilter(2, 2, decay=decay)(arm.alpha_H(theta), arm.LgH(theta), u_nom)
           for decay in DECAY_MODES}
    J_cert = 4.0 / 3.0 * SafetyFilter(2, 2).rho_Omega

    assert res["matrix"].safety_residual >= -LMI_TOL
    assert res["matrix"].objective <= J_cert + 1e-9
    assert np.linalg.norm(res["matrix"].u - u_nom) <= np.sqrt(J_cert + 1e-9)
    assert J_cert < res["scalar"].objective < res["standard"].objective
    for decay in ("scalar", "standard"):
        u = res[decay].u
        assert (u[1] - u[0]) * np.sin(2 * lo) > 1e-2


@pytest.mark.parametrize("rho_omega", [5e-3, 1e-4, 1e-6])
def test_rotation_gap_widens_as_rho_omega_shrinks(rho_omega):
    """Default arm at ``theta2 - theta1 = pi/3``: the matrix filter's change vanishes with ``rho_omega``.

    ``J_matrix <= (4/3) rho_Omega`` shrinks linearly, while the standard filter
    does not depend on ``rho_omega`` and the scalar filter stays above the
    matrix certificate's cost.
    """
    arm = TwoLinkArm()
    theta = np.array([0.2, 0.2 + np.pi / 3])
    u_nom = np.array([1.0, 1.0])
    res = {decay: SafetyFilter(2, 2, decay=decay, rho_omega=rho_omega)(
        arm.alpha_H(theta), arm.LgH(theta), u_nom) for decay in DECAY_MODES}
    J_cert = 4.0 / 3.0 * (4 * rho_omega / 2)
    assert res["matrix"].objective <= J_cert + 1e-9
    assert np.linalg.norm(res["matrix"].u - u_nom) <= np.sqrt(J_cert + 1e-9)
    assert res["scalar"].objective > J_cert
    assert res["standard"].objective > 0.6


@pytest.mark.parametrize("u_nom", [(0.0, 1.0), (-0.5, 0.5), (1.0, 2.0)])
def test_margin_increasing_command_accepted_by_matrix_filter(u_nom):
    """At ``theta2 - theta1 = pi/3`` a command with ``u2 > u1`` raises the margin and is certified.

    The matrix filter keeps it almost unchanged: its cost is at most the
    certification cost ``J_cert`` of ``u_nom`` itself.
    """
    arm = TwoLinkArm()
    theta = np.array([0.2, 0.2 + np.pi / 3])
    u_nom = np.array(u_nom)
    assert _margin_rate(arm, theta, u_nom) > 0
    filt = SafetyFilter(2, 2, decay="matrix")
    J_cert = filt(arm.alpha_H(theta), ZERO_LGH, u_nom, LfH=arm.Hdot(theta, u_nom)).objective
    res = filt(arm.alpha_H(theta), arm.LgH(theta), u_nom)
    assert res.objective <= J_cert + 1e-9
    assert np.linalg.norm(res.u - u_nom) <= min(np.sqrt(J_cert + 1e-9), 1e-2)


# ---------------------------------------------------------- min_scalar_decay
@pytest.mark.parametrize("p", [2, 3, 5])
def test_min_scalar_decay_matches_its_definition(rng, p):
    """``Hdot + omega A >= 0`` iff ``omega >= min_scalar_decay(A, Hdot)``.

    Checked against the generalized eigenvalues of ``(-Hdot, A)`` and by the
    sign of ``lambda_min(Hdot + omega A)`` just above and just below the bound.
    """
    for _ in range(20):
        B = rng.normal(size=(p, p))
        A = B @ B.T + 0.1 * np.eye(p)
        Hdot = _sym(rng.normal(size=(p, p)))
        omega = min_scalar_decay(A, Hdot)
        reference = max(0.0, scipy.linalg.eigh(-Hdot, A, eigvals_only=True)[-1])
        assert omega == pytest.approx(reference, rel=1e-9, abs=1e-12)
        if _lambda_min(Hdot) >= 0:
            assert omega == 0.0
            continue
        assert omega > 0
        scale = np.linalg.norm(Hdot, 2) + omega * np.linalg.norm(A, 2)
        assert abs(_lambda_min(Hdot + omega * A)) <= 1e-12 * scale
        assert _lambda_min(Hdot + omega * (1 + 1e-6) * A) > 0
        assert _lambda_min(Hdot + omega * (1 - 1e-6) * A) < 0


def test_min_scalar_decay_is_zero_when_hdot_is_psd(rng):
    A = np.diag([0.3, 2.0, 5.0])
    assert min_scalar_decay(A, np.zeros((3, 3))) == 0.0
    B = rng.normal(size=(3, 3))
    assert min_scalar_decay(A, B @ B.T + 1e-3 * np.eye(3)) == 0.0
    c = rng.normal(size=(3, 1))
    assert 0.0 <= min_scalar_decay(A, c @ c.T) <= 1e-12          # singular, Hdot >= 0
    assert min_scalar_decay(A, c @ c.T - 1e-3 * np.eye(3)) > 0   # not Hdot >= 0


@pytest.mark.parametrize("alpha_H", [
    np.diag([1.0, 0.0]),
    np.diag([1.0, -1.0]),
    -np.eye(2),
    np.zeros((2, 2)),
    TwoLinkArm().alpha_H(np.array([0.0, np.pi / 4])),
], ids=["singular", "indefinite", "negative-definite", "zero", "arm-unsafe"])
def test_min_scalar_decay_requires_positive_definite_alpha(alpha_H):
    with pytest.raises(ValueError, match="positive definite"):
        min_scalar_decay(alpha_H, np.eye(2))


def test_min_scalar_decay_diverges_toward_the_boundary_under_rotation(arm):
    """Rotation of the whole arm about joint 1 needs an unbounded scalar decay near the boundary.

    For ``u = (w, w)`` in the interior of the safe set, with ``H``'s
    eigenvalues ``h_i = sigma_i**2 - sigma_bar**2``,

        min_scalar_decay = |w| (sigma_max**2 - sigma_min**2) / (gamma sqrt(h_min h_max)),

    which grows like ``1 / sqrt(margin)``.
    """
    lo, hi = arm.safe_theta_diff_interval
    offsets = 10.0 ** -np.arange(1, 9)
    for side in (lambda d: lo + d, lambda d: hi - d, lambda d: -(lo + d), lambda d: -(hi - d)):
        for theta1 in (0.0, 1.1):
            bounds = []
            for d in offsets:
                theta = np.array([theta1, theta1 + side(d)])
                sigma_min_sq, sigma_max_sq = arm.singular_values_sq(theta[1] - theta[0])
                h_min, h_max = sigma_min_sq - arm.sigma_bar_sq, sigma_max_sq - arm.sigma_bar_sq
                assert h_min > 0
                for w in (1.0, -2.0):
                    bound = min_scalar_decay(arm.alpha_H(theta), arm.Hdot(theta, (w, w)))
                    expected = abs(w) * (sigma_max_sq - sigma_min_sq) / (
                        arm.gamma * np.sqrt(h_min * h_max))
                    assert bound == pytest.approx(expected, rel=1e-6)
                bounds.append(bound)
            assert np.all(np.diff(bounds) > 0)
            assert bounds[-1] > 1e3


def test_scalar_filter_decay_is_the_larger_of_omega_d_and_min_scalar_decay(arm, rng):
    """With ``LgH = 0`` and ``LfH = Hdot``, the scalar filter picks ``omega = max(omega_d, bound)``."""
    filt = SafetyFilter(2, 2, decay="scalar", recovery=False)
    regimes = set()
    for scale in (0.05, 3.0) * 5:
        theta = _safe_theta(arm, rng, clearance=0.02)
        Hdot = arm.Hdot(theta, scale * rng.normal(size=2))
        bound = min_scalar_decay(arm.alpha_H(theta), Hdot)
        res = filt(arm.alpha_H(theta), ZERO_LGH, np.zeros(2), LfH=Hdot)
        if bound > filt.omega_d:
            regimes.add("bound")
            assert res.omega == pytest.approx(bound, rel=1e-6)
        else:
            regimes.add("omega_d")
            assert res.omega == pytest.approx(filt.omega_d, abs=1e-3)
    assert regimes == {"bound", "omega_d"}


# ------------------------------------------------------------ infeasibility
@pytest.mark.parametrize("recovery", [True, False])
def test_standard_filter_raises_when_nothing_certifies(recovery):
    """No control authority and ``alpha(H) < 0``: ``Hdot + alpha(H) = alpha(H)`` is never ``>= 0``."""
    filt = SafetyFilter(2, 2, decay="standard", recovery=recovery)
    with pytest.raises(FilterInfeasible, match="standard filter: infeasible"):
        filt(-np.eye(2), ZERO_LGH, np.array([0.3, -0.2]))
    assert issubclass(FilterInfeasible, RuntimeError)


@pytest.mark.parametrize("decay", ["scalar", "matrix"])
def test_recovery_forbids_switching_the_decay_off(decay):
    """With ``alpha(H) = -I`` only ``omega <= 0`` or ``sym(Omega) <= 0`` certify; recovery excludes both."""
    filt = SafetyFilter(2, 2, decay=decay, recovery=True)
    with pytest.raises(FilterInfeasible, match=f"{decay} filter"):
        filt(-np.eye(2), ZERO_LGH, np.array([0.3, -0.2]))


@pytest.mark.parametrize("omega_d", [1.0, 0.4])
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("decay", ["scalar", "matrix"])
def test_without_recovery_the_decay_switches_off(decay, p, omega_d):
    """With ``alpha(H) = -I`` and no recovery, the cheapest multiplier is zero.

    The cost is ``rho_omega omega_d**2`` in both modes and for every ``p``,
    because the matched penalties agree on ``Omega = (omega / 2) I``.
    """
    rho_omega = 5e-3
    filt = SafetyFilter(2, p, decay=decay, recovery=False, omega_d=omega_d, rho_omega=rho_omega)
    u_nom = np.array([0.3, -0.2])
    res = filt(-np.eye(p), [np.zeros((p, p))] * 2, u_nom)
    np.testing.assert_allclose(res.u, u_nom, rtol=0, atol=1e-8)
    assert res.objective == pytest.approx(rho_omega * omega_d ** 2, rel=1e-6, abs=2e-9)
    np.testing.assert_allclose(res.Omega, np.zeros((p, p)), rtol=0, atol=1e-5)
    assert res.safety_residual >= -LMI_TOL


@pytest.mark.parametrize("decay", DECAY_MODES)
def test_filter_is_reusable_after_an_infeasible_solve(rng, decay):
    """Per-step data enters through parameters: an infeasible solve leaves no trace.

    A control filter that makes the same feasible solves, without the
    infeasible one, returns identical results. Against a fresh filter the
    agreement is only to the solver's resolution (``u`` to about ``1e-5``),
    because CVXPY's first solve of a problem and its later, cached solves of
    the same data differ slightly.
    """
    arm = TwoLinkArm()
    filt, control = SafetyFilter(2, 2, decay=decay), SafetyFilter(2, 2, decay=decay)
    states = [(arm.alpha_H(theta), arm.LgH(theta), u_nom) for theta, u_nom in _arm_states(arm, rng, 3)]
    for data in states:
        filt(*data)
        control(*data)
    with pytest.raises(FilterInfeasible):
        filt(-np.eye(2), ZERO_LGH, states[0][2])
    for data in states:
        res, expected = filt(*data), control(*data)
        fresh = SafetyFilter(2, 2, decay=decay)(*data)
        np.testing.assert_allclose(res.u, expected.u, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res.Omega, expected.Omega, rtol=0, atol=1e-12)
        assert res.objective == pytest.approx(expected.objective, rel=1e-12, abs=1e-15)
        np.testing.assert_allclose(res.u, fresh.u, rtol=0, atol=1e-4)
        assert res.objective == pytest.approx(fresh.objective, rel=1e-6, abs=1e-9)
