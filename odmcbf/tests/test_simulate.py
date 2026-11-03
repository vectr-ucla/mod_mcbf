"""Unit tests of :mod:`odmcbf.simulate`: rollouts, the controller comparison and its summary.

The ``comparison`` fixture (conftest) runs the notebook's closed loop on the
unit-link arm with ``u_nom = (1, 0.4)`` from ``theta0 = (0, pi/2)``, ``dt =
0.04`` and ``horizon = 8``, with ``omega_d = 1``, ``rho_omega = 5e-3`` and the
recovery constraint on. That setup is restated below as module constants.
"""

from __future__ import annotations

import numpy as np
import pytest

import odmcbf.simulate as simulate
from odmcbf import (
    CONTROLLERS,
    DECAY_MODES,
    SafetyFilter,
    Trace,
    TwoLinkArm,
    min_scalar_decay,
    rollout,
    run_comparison,
    summarize,
)
from odmcbf.simulate import SETTLE_WINDOW

#: Setup of the ``comparison`` fixture.
U_NOM = np.array([1.0, 0.4])
THETA0 = np.array([0.0, np.pi / 2])
DT = 0.04
HORIZON = 8.0
RHO_OMEGA = 5e-3
#: The matrix filter's penalty, ``rho_Omega = (4 / p) rho_omega`` with ``p = 2``.
RHO_OMEGA_MATRIX = 4.0 * RHO_OMEGA / 2

NAMES = tuple(name for name, _, _ in CONTROLLERS)
FILTERED = ("standard", "scalar", "matrix")
PER_STEP_FIELDS = ("decay_deviation", "scalar_certificate", "safety_residual",
                   "recovery_residual", "solve_ms")
FILTER_ONLY_FIELDS = ("decay_deviation", "safety_residual", "recovery_residual", "solve_ms")
SUMMARY_KEYS = {
    "min_margin",
    "joint1_rotation",
    "late_joint1_rate",
    "control_deviation_l2",
    "decay_deviation_settled",
    "mean_solve_ms",
    "median_solve_ms",
    "min_safety_residual",
    "min_recovery_residual",
}


def _alpha_H_is_pd(arm: TwoLinkArm, theta) -> bool:
    return bool(np.linalg.eigvalsh(arm.alpha_H(theta))[0] > 0)


def _with_gamma(arm: TwoLinkArm, gamma: float) -> TwoLinkArm:
    """``arm`` with class-K rate ``gamma``; for ``gamma != 1``, ``alpha_H = gamma H`` differs from ``H``."""
    return TwoLinkArm(arm.l1, arm.l2, gamma=gamma, sigma_bar=arm.sigma_bar)


def _settle_mask(trace: Trace) -> np.ndarray:
    """Steps whose control is held during the last ``SETTLE_WINDOW`` seconds."""
    dt = trace.t[1] - trace.t[0]
    return trace.t[:-1] > trace.t[-1] - SETTLE_WINDOW - dt / 2


# ----------------------------------------------------------------- CONTROLLERS
def test_controllers_table():
    """The comparison runs the nominal command and then each decay mode once."""
    assert NAMES == ("nominal", "standard", "scalar", "matrix")
    assert CONTROLLERS[0][1] is None
    assert tuple(decay for _, decay, _ in CONTROLLERS[1:]) == DECAY_MODES
    labels = [label for _, _, label in CONTROLLERS]
    assert len(set(labels)) == len(labels)
    assert all(isinstance(label, str) and label for label in labels)


# ------------------------------------------------------------ rollout: inputs
@pytest.mark.parametrize("dt", [0.0, -0.01])
def test_rollout_rejects_nonpositive_dt(dt):
    with pytest.raises(ValueError, match="dt must be positive"):
        rollout(TwoLinkArm(), U_NOM, THETA0, dt=dt, horizon=1.0)


@pytest.mark.parametrize("dt, horizon",
                         [(0.03, 0.1), (0.3, 1.0), (1.0, 0.4), (0.8, 0.4), (0.01, 1.001)])
def test_rollout_rejects_dt_not_dividing_horizon(dt, horizon):
    """Including a near miss: 100 steps of 0.01 fall short of 1.001 by 0.1 %, far beyond rounding."""
    with pytest.raises(ValueError, match="dt must divide the horizon"):
        rollout(TwoLinkArm(), U_NOM, THETA0, dt=dt, horizon=horizon)


def test_rollout_accepts_dt_dividing_horizon_up_to_rounding():
    """``0.7 / 0.1`` is ``6.999...`` in floating point; the rollout still takes 7 steps."""
    trace = rollout(TwoLinkArm(), U_NOM, THETA0, dt=0.1, horizon=0.7)
    assert trace.u.shape == (7, 2)
    assert trace.t[-1] == pytest.approx(0.7, abs=1e-12)


@pytest.mark.parametrize("decay", [None, "matrix"])
@pytest.mark.parametrize(
    "u_nom",
    [(1.0,), (1.0, 0.4, 0.0), [[1.0], [0.4]], [[1.0, 0.4]], 1.0],
    ids=["len1", "len3", "column", "row", "scalar"],
)
def test_rollout_rejects_wrong_u_nom_shape(u_nom, decay):
    with pytest.raises(ValueError, match=r"u_nom must have shape \(2,\)"):
        rollout(TwoLinkArm(), u_nom, THETA0, dt=0.1, horizon=0.5, decay=decay)


def test_rollout_rejects_unknown_decay():
    with pytest.raises(ValueError, match="decay must be one of"):
        rollout(TwoLinkArm(), U_NOM, THETA0, dt=0.1, horizon=0.5, decay="exponential")


# ------------------------------------------------------ rollout: nominal runs
def test_nominal_rollout_is_exact_euler(arm):
    """Unfiltered, ``theta[k] = theta0 + k dt u_nom`` on the grid ``t[k] = k dt``.

    ``theta1`` starts negative: the initial condition is stored as given, not wrapped.
    """
    theta0, u_nom, dt = np.array([-0.3, 1.1]), np.array([0.7, -0.25]), 0.05
    trace = rollout(arm, u_nom, theta0, dt=dt, horizon=2.0)
    k = np.arange(41)
    np.testing.assert_allclose(trace.t, k * dt, rtol=0, atol=1e-14)
    np.testing.assert_allclose(trace.theta, theta0 + np.outer(k * dt, u_nom), rtol=0, atol=1e-13)
    np.testing.assert_array_equal(trace.u, np.tile(u_nom, (40, 1)))


def test_nominal_margin_and_certificate(arm):
    """``margin[k] = lambda_min(H(theta[k])) = sigma_min(J)**2 - sigma_bar**2``, independent of ``gamma``.

    The arm has ``gamma = 2.5``, so ``H`` and ``alpha_H`` differ. The command
    closes ``theta2 - theta1`` from ``pi/2`` through zero, so the run crosses
    the safe set, its boundary and the violated region, and the scalar
    certificate of ``u_nom`` is finite exactly where ``alpha_H(theta[k])`` is
    positive definite, where it is ``min_scalar_decay(alpha_H, Hdot)``.
    """
    arm = _with_gamma(arm, 2.5)
    u_nom = np.array([0.2, -0.4])
    trace = rollout(arm, u_nom, (0.0, np.pi / 2), dt=0.05, horizon=4.0)
    expected = np.array([arm.margin(theta) for theta in trace.theta])
    np.testing.assert_allclose(trace.margin, expected, rtol=0, atol=1e-14)
    theta_diff = trace.theta[:, 1] - trace.theta[:, 0]
    closed_form = arm.sigma_min_sq(theta_diff) - arm.sigma_bar_sq
    np.testing.assert_allclose(trace.margin, closed_form, rtol=0, atol=1e-12)
    assert trace.margin.max() > 0 > trace.margin.min()

    pd = np.array([_alpha_H_is_pd(arm, theta) for theta in trace.theta[:-1]])
    assert pd.any() and not pd.all()
    np.testing.assert_array_equal(np.isfinite(trace.scalar_certificate), pd)
    certificate = [min_scalar_decay(arm.alpha_H(trace.theta[k]), arm.Hdot(trace.theta[k], u_nom))
                   for k in np.flatnonzero(pd)]
    np.testing.assert_allclose(trace.scalar_certificate[pd], certificate, rtol=1e-12, atol=0)


def test_nominal_rollout_fills_only_nominal_fields(arm):
    """Unfiltered, ``Omega = I/2`` and every filter-only field is NaN."""
    trace = rollout(arm, U_NOM, (0.3, 0.3 + np.pi / 2), dt=0.1, horizon=1.0)
    assert trace.decay is None
    np.testing.assert_array_equal(trace.Omega, np.broadcast_to(0.5 * np.eye(2), (10, 2, 2)))
    for field in FILTER_ONLY_FIELDS:
        assert np.all(np.isnan(getattr(trace, field))), field


def test_trace_arm_is_the_arm_used(arm):
    for decay in (None, "matrix"):
        trace = rollout(arm, U_NOM, (0.3, 0.3 + np.pi / 2), dt=0.1, horizon=0.3, decay=decay)
        assert trace.arm is arm


@pytest.mark.parametrize("decay", [None, *DECAY_MODES])
def test_trace_field_shapes(decay):
    trace = rollout(TwoLinkArm(), U_NOM, THETA0, dt=0.1, horizon=0.5, decay=decay)
    assert trace.t.shape == (6,)
    assert trace.theta.shape == (6, 2)
    assert trace.u.shape == (5, 2)
    assert trace.margin.shape == (6,)
    assert trace.Omega.shape == (5, 2, 2)
    for field in PER_STEP_FIELDS:
        assert getattr(trace, field).shape == (5,), field
    for field in ("t", "theta", "u", "margin", "Omega", *PER_STEP_FIELDS):
        assert getattr(trace, field).dtype == np.float64, field


@pytest.mark.parametrize(
    "decay, kwargs, name, label",
    [
        (None, {}, "nominal", "nominal"),
        ("standard", {}, "standard", "standard"),
        ("scalar", {}, "scalar", "scalar"),
        ("matrix", {}, "matrix", "matrix"),
        (None, {"name": "k_nom", "label": "unfiltered"}, "k_nom", "unfiltered"),
        ("matrix", {"label": "matrix OD"}, "matrix", "matrix OD"),
        ("scalar", {"name": "od"}, "od", "scalar"),
    ],
)
def test_rollout_name_and_label(decay, kwargs, name, label):
    """``name`` and ``label`` default to the decay mode, or ``"nominal"`` unfiltered."""
    trace = rollout(TwoLinkArm(), U_NOM, THETA0, dt=0.1, horizon=0.2, decay=decay, **kwargs)
    assert (trace.name, trace.label, trace.decay) == (name, label, decay)


# ----------------------------------------------------------- time convention
@pytest.fixture
def recorded_filters(monkeypatch):
    """Replace the rollout's filter class by one that records every solve."""
    instances = []

    class RecordingFilter(SafetyFilter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []
            instances.append(self)

        def __call__(self, alpha_H, LgH, u_nom, LfH=None):
            result = super().__call__(alpha_H, LgH, u_nom, LfH)
            self.calls.append(dict(alpha_H=np.array(alpha_H), LgH=[np.array(G) for G in LgH],
                                   u_nom=np.array(u_nom), LfH=LfH, result=result))
            return result

    monkeypatch.setattr(simulate, "SafetyFilter", RecordingFilter)
    return instances


@pytest.mark.parametrize("gamma", [1.0, 2.5])
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_filter_is_solved_at_theta_k(arm, decay, gamma, recorded_filters):
    """``u[k]`` is the filter's output for the data of ``theta[k]``, not of ``theta[k+1]``.

    The filter receives ``alpha_H = gamma H``, not ``H``, and every per-step
    field of the trace is the one the filter returned at that step.
    """
    arm = _with_gamma(arm, gamma)
    trace = rollout(arm, U_NOM, (0.3, 0.3 + np.pi / 2), dt=0.1, horizon=0.6, decay=decay)
    (filt,) = recorded_filters
    assert (filt.m, filt.p, filt.decay) == (arm.n_controls, arm.dim, decay)
    assert len(filt.calls) == 6
    for k, call in enumerate(filt.calls):
        theta_k = trace.theta[k]
        np.testing.assert_allclose(call["alpha_H"], arm.alpha_H(theta_k), rtol=0, atol=1e-14)
        assert len(call["LgH"]) == 2
        for G, G_expected in zip(call["LgH"], arm.LgH(theta_k)):
            np.testing.assert_allclose(G, G_expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(call["u_nom"], U_NOM)
        # the constraint is driftless: no drift, or an explicit zero one
        assert call["LfH"] is None or not np.any(call["LfH"])
        result = call["result"]
        if _alpha_H_is_pd(arm, theta_k):
            np.testing.assert_allclose(
                trace.scalar_certificate[k],
                min_scalar_decay(arm.alpha_H(theta_k), arm.Hdot(theta_k, result.u)),
                rtol=1e-12, atol=0)
        else:       # the Euler step can leave the safe set; no certificate there
            assert np.isnan(trace.scalar_certificate[k])
        np.testing.assert_array_equal(trace.u[k], result.u)
        np.testing.assert_array_equal(trace.Omega[k], result.Omega)
        assert trace.safety_residual[k] == result.safety_residual
        assert trace.solve_ms[k] == 1e3 * result.solve_time      # milliseconds
        if decay == "standard":
            assert np.isnan(trace.recovery_residual[k]) and np.isnan(trace.decay_deviation[k])
        else:
            assert trace.recovery_residual[k] == result.recovery_residual
        assert not np.allclose(call["alpha_H"], arm.alpha_H(trace.theta[k + 1]))


def test_rollout_forwards_filter_kwargs(recorded_filters):
    trace = rollout(TwoLinkArm(), U_NOM, THETA0, dt=0.1, horizon=0.3, decay="scalar",
                    omega_d=2.0, rho_omega=1e-2, recovery=False)
    (filt,) = recorded_filters
    assert (filt.omega_d, filt.rho_omega, filt.recovery) == (2.0, 1e-2, False)
    assert np.all(np.isnan(trace.recovery_residual))
    np.testing.assert_array_equal(trace.decay_deviation, 2 * trace.Omega[:, 0, 0] - 2.0)


@pytest.mark.parametrize("name", NAMES)
def test_controls_are_held_over_each_step(comparison, name):
    """``u[k]`` acts on all of ``[t[k], t[k+1])``: ``theta[k+1] = theta[k] + dt u[k]``."""
    trace = comparison[name]
    np.testing.assert_array_equal(trace.theta[0], THETA0)
    np.testing.assert_allclose(trace.t, np.arange(201) * DT, rtol=0, atol=1e-14)
    np.testing.assert_allclose(trace.theta[1:], trace.theta[:-1] + DT * trace.u, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "name, closing_rate",
    [
        ("standard", 0.5),
        ("scalar", (0.3 + 2 * RHO_OMEGA) / (0.5 + 4 * RHO_OMEGA)),
        ("matrix", (0.6 + RHO_OMEGA_MATRIX) / (1 + 2 * RHO_OMEGA_MATRIX)),
    ],
)
def test_first_command_is_filtered_at_theta0(comparison, name, closing_rate):
    """``u[0]`` is the filter's closed-form answer at ``theta0 = (0, pi/2)``.

    There ``alpha_H = I/2`` and ``Hdot`` has eigenvalues ``+-(u1 - u2)``, and
    ``u_nom`` closes ``theta2 - theta1`` at ``u1 - u2 = 0.6``. Each filter keeps
    ``u1 + u2 = 1.4`` and reduces the closing rate: the standard filter to
    ``1/2``; the scalar filter to ``omega/2`` with the ``omega`` that balances
    ``(0.6 - omega/2)**2 / 2 + rho_omega (omega - 1)**2``; the matrix filter
    raises ``sym(Omega)`` along the eigenvector ``(1, 1)`` only, at cost
    ``rho_Omega (u1 - u2 - 1/2)**2``.
    """
    u0 = U_NOM - 0.5 * (0.6 - closing_rate) * np.array([1.0, -1.0])
    np.testing.assert_allclose(comparison[name].u[0], u0, rtol=0, atol=1e-7)


# ------------------------------------------------------ the comparison fixture
def test_comparison_matches_controllers(comparison):
    assert tuple(comparison) == NAMES
    arm = comparison["nominal"].arm
    assert isinstance(arm, TwoLinkArm)
    for name, decay, label in CONTROLLERS:
        trace = comparison[name]
        assert (trace.name, trace.decay, trace.label) == (name, decay, label)
        assert trace.arm is arm


def test_only_the_nominal_run_violates_the_constraint(comparison):
    """The nominal run passes the stretched singular configuration; the filters stay safe.

    Its grid passes within ``0.6 dt / 2`` of ``theta2 = theta1``, where
    ``sigma_min(J)**2 ~ sin(theta2 - theta1)**2 / 2 < 1e-4``, so the margin
    comes within ``1e-4`` of its floor ``-sigma_bar**2``.
    """
    nominal = comparison["nominal"]
    sigma_bar_sq = nominal.arm.sigma_bar_sq
    assert nominal.margin.min() < 0
    assert nominal.margin.min() >= -sigma_bar_sq       # sigma_min(J) >= 0
    assert nominal.margin.min() == pytest.approx(-sigma_bar_sq, abs=1e-4)
    for name in FILTERED:
        assert comparison[name].margin.min() >= -1e-6, name


def test_joint1_rotation_ordering(comparison):
    """Joint 1 turns furthest under the matrix filter, then scalar, then standard."""
    rotation = {name: trace.theta[-1, 0] - trace.theta[0, 0] for name, trace in comparison.items()}
    assert rotation["nominal"] == pytest.approx(U_NOM[0] * HORIZON, rel=1e-12)
    assert rotation["nominal"] > rotation["matrix"] > rotation["scalar"] > rotation["standard"] > 0


def test_matrix_filter_settles_on_rotation_about_joint1(comparison):
    """The matrix filter ends up rotating the whole arm about joint 1 at the boundary.

    At a boundary configuration the matrix filter certifies exactly the
    commands with ``(u2 - u1) sin 2(theta2 - theta1) >= 0``; at
    ``theta2 - theta1 = pi/3`` that is ``u2 >= u1``, and the closest such
    command to ``u_nom = (1, 0.4)`` is the rotation ``(0.7, 0.7)``. The
    penalty ``rho_Omega ||Omega - I/2||_F**2`` then shrinks the rate ``a``.
    Certifying a rotation at rate ``a`` costs ``c a**2`` in
    ``||Omega - I/2||_F**2`` with ``0 <= c <= 2`` (an antisymmetric
    ``Omega - I/2`` of size ``a`` suffices and keeps ``sym(Omega) = I/2``), and
    the minimiser ``0.7 / (1 + c rho_Omega / 2)`` of
    ``2 (a - 0.7)**2 + c rho_Omega a**2`` lies in ``[0.7 / (1 + rho_Omega), 0.7]``.
    Each settled joint rate must lie there too, and both joints turn at the
    same rate, which holds ``theta2 - theta1`` (to ``1e-4`` rad/s, a drift of
    at most ``2e-4`` rad over the window).
    """
    trace = comparison["matrix"]
    boundary = trace.arm.safe_theta_diff_interval[0]
    assert boundary == pytest.approx(np.pi / 3, rel=1e-12)
    settled = _settle_mask(trace)
    assert settled.sum() == round(SETTLE_WINDOW / DT)
    theta_diff = trace.theta[1:, 1] - trace.theta[1:, 0]
    assert np.all(theta_diff[settled] - boundary > 0)
    assert np.all(theta_diff[settled] - boundary < 1e-3)
    assert np.all((trace.margin[1:][settled] > 0) & (trace.margin[1:][settled] < 1e-3))
    u = trace.u[settled]
    assert np.all(np.abs(u[:, 0] - u[:, 1]) < 1e-4)                 # rotation about joint 1
    assert np.all(u >= 0.7 / (1 + RHO_OMEGA_MATRIX))
    assert np.all(u <= 0.7)


@pytest.mark.parametrize("name", FILTERED)
def test_filter_residuals(comparison, name):
    """The applied LMI holds at every step, and so does the recovery constraint where imposed.

    The recovery slack is ``omega - omega_d`` for scalar decay and
    ``lambda_min(sym(Omega)) - beta`` for matrix decay, with the default
    ``beta = omega_d / 2``, recomputed here from the applied ``Omega``.
    """
    trace = comparison[name]
    assert np.all(np.isfinite(trace.safety_residual))
    assert trace.safety_residual.min() >= -1e-7
    assert np.all(np.isfinite(trace.solve_ms)) and trace.solve_ms.min() > 0
    if name == "standard":
        assert np.all(np.isnan(trace.recovery_residual))
        return
    assert np.all(np.isfinite(trace.recovery_residual))
    assert trace.recovery_residual.min() >= -1e-7
    if name == "scalar":
        slack = 2 * trace.Omega[:, 0, 0] - 1.0                    # Omega = (omega/2) I
    else:
        sym_Omega = 0.5 * (trace.Omega + trace.Omega.transpose(0, 2, 1))
        slack = np.linalg.eigvalsh(sym_Omega)[:, 0] - 0.5
    np.testing.assert_allclose(trace.recovery_residual, slack, rtol=0, atol=1e-15)


@pytest.mark.parametrize("beta", [0.2, 0.8])
def test_matrix_recovery_residual_is_slack_for_beta(beta):
    """With ``beta`` forwarded, ``recovery_residual`` is the slack ``lambda_min(sym(Omega)) - beta``.

    The imposed recovery constraint is ``sym(Omega) >= beta I``; ``beta = 0.2``
    and ``0.8`` differ from the default ``omega_d / 2 = 1/2``.
    """
    trace = rollout(TwoLinkArm(), U_NOM, THETA0, dt=0.1, horizon=0.2, decay="matrix", beta=beta)
    sym_Omega = 0.5 * (trace.Omega + trace.Omega.transpose(0, 2, 1))
    slack = np.linalg.eigvalsh(sym_Omega)[:, 0] - beta
    assert slack.min() >= -1e-7                          # the constraint does hold
    np.testing.assert_allclose(trace.recovery_residual, slack, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", FILTERED)
def test_decay_deviation_and_Omega(comparison, name):
    """``omega - omega_d`` for scalar decay, ``||Omega - I/2||_F`` for matrix decay, NaN otherwise."""
    trace = comparison[name]
    if name == "standard":
        assert np.all(np.isnan(trace.decay_deviation))
        np.testing.assert_array_equal(trace.Omega, np.broadcast_to(0.5 * np.eye(2), trace.Omega.shape))
    elif name == "scalar":
        omega = 2 * trace.Omega[:, 0, 0]
        np.testing.assert_array_equal(trace.Omega, 0.5 * omega[:, None, None] * np.eye(2))
        np.testing.assert_array_equal(trace.decay_deviation, omega - 1.0)
    else:
        expected = np.linalg.norm(trace.Omega - 0.5 * np.eye(2), axis=(1, 2))
        np.testing.assert_allclose(trace.decay_deviation, expected, rtol=1e-14, atol=0)
    if name != "standard":
        assert np.all(np.isfinite(trace.decay_deviation))


@pytest.mark.parametrize("name", NAMES)
def test_scalar_certificate_where_alpha_H_is_pd(comparison, name):
    """Finite exactly where ``alpha_H(theta[k]) > 0``, and there the smallest certifying decay of ``u[k]``."""
    trace = comparison[name]
    arm = trace.arm
    pd = np.array([_alpha_H_is_pd(arm, theta) for theta in trace.theta[:-1]])
    np.testing.assert_array_equal(np.isfinite(trace.scalar_certificate), pd)
    if name == "nominal":
        assert pd.any() and not pd.all()
    expected = [min_scalar_decay(arm.alpha_H(trace.theta[k]), arm.Hdot(trace.theta[k], trace.u[k]))
                for k in np.flatnonzero(pd)]
    np.testing.assert_allclose(trace.scalar_certificate[pd], expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("name", ["standard", "scalar"])
def test_scalar_certificate_is_below_the_applied_decay(comparison, name):
    """A run with scalar decay ``omega`` certifies ``u[k]``, so its certificate is at most ``omega``.

    The LMI residual ``r`` loosens that by ``max(0, -r) / lambda_min(alpha_H)``.
    The standard filter is scalar decay with ``omega = 1``.
    """
    trace = comparison[name]
    arm = trace.arm
    omega = 2 * trace.Omega[:, 0, 0]
    if name == "standard":
        np.testing.assert_array_equal(omega, 1.0)
    lam_min = np.array([np.linalg.eigvalsh(arm.alpha_H(theta))[0] for theta in trace.theta[:-1]])
    slack = np.maximum(0.0, -trace.safety_residual) / lam_min
    assert np.all(trace.scalar_certificate <= omega + slack + 1e-10)


# ------------------------------------------------------------------ summarize
@pytest.mark.parametrize("name", NAMES)
def test_summarize_keys_and_types(comparison, name):
    summary = summarize(comparison[name], U_NOM)
    assert set(summary) == SUMMARY_KEYS
    assert all(value is None or type(value) is float for value in summary.values())


@pytest.mark.parametrize("name", NAMES)
def test_summarize_matches_trace(comparison, name):
    """Each entry agrees with its definition, recomputed from the trace."""
    trace = comparison[name]
    summary = summarize(trace, U_NOM)
    arm = trace.arm
    dt = np.diff(trace.t)
    settled = _settle_mask(trace)
    k0 = np.flatnonzero(settled)[0]

    assert summary["min_margin"] == pytest.approx(
        min(arm.margin(theta) for theta in trace.theta), rel=1e-14, abs=1e-15)
    assert summary["joint1_rotation"] == pytest.approx(np.sum(dt * trace.u[:, 0]), rel=1e-12)
    # average joint-1 rate over the last SETTLE_WINDOW seconds = displacement / time
    late_rate = (trace.theta[-1, 0] - trace.theta[k0, 0]) / (trace.t[-1] - trace.t[k0])
    assert summary["late_joint1_rate"] == pytest.approx(late_rate, rel=1e-10)
    # L2 norm in time of the piecewise-constant deviation u - u_nom
    deviation_sq = np.sum((trace.u - U_NOM) ** 2, axis=1)
    assert summary["control_deviation_l2"] == pytest.approx(
        np.sqrt(np.sum(dt * deviation_sq)), rel=1e-12, abs=1e-15)

    if name in ("scalar", "matrix"):
        assert summary["decay_deviation_settled"] == pytest.approx(
            np.mean(trace.decay_deviation[settled]), rel=1e-12)
        assert summary["min_recovery_residual"] == trace.recovery_residual.min()
    else:
        assert summary["decay_deviation_settled"] is None
        assert summary["min_recovery_residual"] is None
    if name in FILTERED:
        assert summary["mean_solve_ms"] == pytest.approx(np.mean(trace.solve_ms), rel=1e-12)
        assert summary["median_solve_ms"] == np.median(trace.solve_ms)
        assert summary["min_safety_residual"] == trace.safety_residual.min()


def test_summarize_nominal_run(comparison):
    """The nominal run has no solver data; its joint 1 turns at exactly ``u_nom[0]``."""
    summary = summarize(comparison["nominal"], U_NOM)
    for key in ("decay_deviation_settled", "mean_solve_ms", "median_solve_ms",
                "min_safety_residual", "min_recovery_residual"):
        assert summary[key] is None, key
    assert summary["joint1_rotation"] == pytest.approx(U_NOM[0] * HORIZON, rel=1e-12)
    assert summary["late_joint1_rate"] == U_NOM[0]
    assert summary["control_deviation_l2"] == 0.0
    assert summary["min_margin"] < 0


def test_summarize_nominal_run_in_closed_form():
    """Constant ``u_nom = (0.3, 0.5)`` from ``theta0 = (0.5, 1.7)`` on the unit-link arm, for 3 s.

    ``theta2 - theta1`` opens from 1.2 through ``pi/2`` to 1.8, and the margin
    depends on ``cos(theta2 - theta1)**2``, which is largest at the start, so
    the smallest margin is the initial one. Joint 1 turns ``0.3 * 3`` rad
    (from ``theta1 = 0.5``, not from zero) at the constant rate 0.3, and the
    time-L2 distance to a zero command is ``sqrt(3) ||u_nom||``.
    """
    arm = TwoLinkArm()
    u_nom, theta0, horizon = np.array([0.3, 0.5]), np.array([0.5, 1.7]), 3.0
    trace = rollout(arm, u_nom, theta0, dt=0.05, horizon=horizon)
    assert np.argmin(trace.margin) == 0
    summary = summarize(trace, u_nom)
    assert summary["min_margin"] == arm.margin(theta0)
    assert summary["joint1_rotation"] == pytest.approx(0.3 * horizon, rel=1e-12)
    assert summary["late_joint1_rate"] == pytest.approx(0.3, rel=1e-12)
    assert summary["control_deviation_l2"] == 0.0
    to_zero = summarize(trace, (0.0, 0.0))["control_deviation_l2"]
    assert to_zero == pytest.approx(np.sqrt(horizon) * np.linalg.norm(u_nom), rel=1e-12)


def test_summarize_short_run_uses_whole_run():
    """A run shorter than ``SETTLE_WINDOW`` averages its settled quantities over all of it."""
    trace = rollout(TwoLinkArm(), U_NOM, THETA0, dt=0.1, horizon=0.5, decay="scalar")
    summary = summarize(trace, U_NOM)
    assert summary["late_joint1_rate"] == pytest.approx(np.mean(trace.u[:, 0]), rel=1e-12)
    assert summary["decay_deviation_settled"] == pytest.approx(np.mean(trace.decay_deviation),
                                                               rel=1e-12)


# ---------------------------------------------------------------------- Trace
def test_trace_control_deviation(comparison):
    trace = comparison["standard"]
    np.testing.assert_allclose(trace.control_deviation(U_NOM),
                               np.linalg.norm(trace.u - U_NOM, axis=1), rtol=1e-15)
    np.testing.assert_allclose(trace.control_deviation((0.0, 0.0)),
                               np.linalg.norm(trace.u, axis=1), rtol=1e-15)
    # at theta0 the standard filter moves (1, 0.4) to (0.95, 0.45)
    assert trace.control_deviation(tuple(U_NOM))[0] == pytest.approx(0.05 * np.sqrt(2), abs=1e-7)
    np.testing.assert_array_equal(comparison["nominal"].control_deviation(U_NOM), 0.0)


def test_trace_control_deviation_by_hand():
    nan = np.full(2, np.nan)
    trace = Trace(
        name="hand", label="hand", decay=None, t=np.array([0.0, 1.0, 2.0]),
        theta=np.zeros((3, 2)), u=np.array([[4.0, 5.0], [1.0, 1.0]]), margin=np.zeros(3),
        decay_deviation=nan, scalar_certificate=nan, safety_residual=nan,
        recovery_residual=nan, solve_ms=nan, Omega=np.zeros((2, 2, 2)),
    )
    assert trace.arm is None
    np.testing.assert_array_equal(trace.control_deviation([1.0, 1.0]), [5.0, 0.0])


# -------------------------------------------------------------- run_comparison
def test_run_comparison_default_arm():
    """Without ``arm``, the comparison runs on the unit-link arm with the default threshold."""
    traces = run_comparison(horizon=0.05)
    assert tuple(traces) == NAMES
    arm = traces["nominal"].arm
    assert type(arm) is TwoLinkArm
    reference = TwoLinkArm()
    assert (arm.l1, arm.l2, arm.gamma, arm.sigma_bar) == (
        reference.l1, reference.l2, reference.gamma, reference.sigma_bar)
    for name, decay, label in CONTROLLERS:
        trace = traces[name]
        assert trace.arm is arm
        assert (trace.name, trace.decay, trace.label) == (name, decay, label)
        np.testing.assert_allclose(trace.t, np.arange(6) * 0.01, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(trace.theta[0], [0.0, np.pi / 2])
    np.testing.assert_array_equal(traces["nominal"].u, np.tile([1.0, 0.4], (5, 1)))


def test_run_comparison_uses_its_arm_command_and_start():
    """Each run is ``rollout`` of the given arm, ``u_nom``, ``theta0``, ``dt`` and ``horizon``."""
    arm = TwoLinkArm(1.0, 0.6, gamma=2.5, sigma_bar=0.3)
    u_nom, theta0 = np.array([0.3, -0.2]), np.array([0.4, 2.0])
    traces = run_comparison(arm, u_nom=u_nom, theta0=theta0, dt=0.05, horizon=0.2)
    assert tuple(traces) == NAMES
    for name, decay, label in CONTROLLERS:
        trace = traces[name]
        assert trace.arm is arm
        assert (trace.name, trace.decay, trace.label) == (name, decay, label)
        np.testing.assert_array_equal(trace.theta[0], theta0)
        np.testing.assert_allclose(trace.t, np.arange(5) * 0.05, rtol=0, atol=1e-15)
        expected = rollout(arm, u_nom, theta0, dt=0.05, horizon=0.2, decay=decay)
        np.testing.assert_allclose(trace.u, expected.u, rtol=0, atol=1e-9)
        np.testing.assert_allclose(trace.theta, expected.theta, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(traces["nominal"].u, np.tile(u_nom, (4, 1)))


def test_run_comparison_forwards_filter_kwargs():
    traces = run_comparison(dt=0.05, horizon=0.2, omega_d=2.0, recovery=False)
    for name in ("scalar", "matrix"):
        assert np.all(np.isnan(traces[name].recovery_residual)), name
    np.testing.assert_array_equal(traces["scalar"].decay_deviation,
                                  2 * traces["scalar"].Omega[:, 0, 0] - 2.0)
    np.testing.assert_allclose(traces["matrix"].decay_deviation,
                               np.linalg.norm(traces["matrix"].Omega - np.eye(2), axis=(1, 2)),
                               rtol=1e-14)


# ------------------------------------------------------ nominal controllers
def _constant_controller(t, theta):
    return U_NOM.copy()


@pytest.mark.parametrize("decay", [None, *DECAY_MODES])
def test_constant_controller_matches_constant_command(decay):
    """A controller returning ``u_nom`` everywhere gives the constant command's run, bit for bit."""
    constant = rollout(TwoLinkArm(), U_NOM, THETA0, dt=0.1, horizon=1.0, decay=decay)
    controlled = rollout(TwoLinkArm(), _constant_controller, THETA0, dt=0.1, horizon=1.0, decay=decay)
    for field in ("t", "theta", "u", "margin", "Omega", "u_nom", *PER_STEP_FIELDS):
        if field == "solve_ms":
            continue
        np.testing.assert_array_equal(getattr(controlled, field), getattr(constant, field), err_msg=field)


def test_constant_command_is_recorded():
    trace = rollout(TwoLinkArm(), U_NOM, THETA0, dt=0.1, horizon=0.5, decay="matrix")
    assert trace.u_nom.shape == (5, 2) and trace.u_nom.dtype == np.float64
    np.testing.assert_array_equal(trace.u_nom, np.tile(U_NOM, (5, 1)))


@pytest.mark.parametrize("decay", [None, "matrix"])
def test_controller_is_evaluated_at_t_k_and_theta_k(arm, decay, recorded_filters):
    """The controller sees ``(t[k], theta[k])`` of the run itself, and the filter gets its output."""
    calls = []

    def controller(t, theta):
        calls.append((t, theta.copy()))
        command = np.array([0.4 + 0.3 * t, 0.2 - 0.5 * np.sin(theta[1] - theta[0])])
        theta[:] = 99.0                     # a copy: changing it leaves the run alone
        return command

    theta0 = (0.3, 0.3 + np.pi / 2)
    trace = rollout(arm, controller, theta0, dt=0.1, horizon=0.6, decay=decay)
    assert len(calls) == 6
    for k, (t, theta) in enumerate(calls):
        assert t == k * 0.1
        np.testing.assert_array_equal(theta, trace.theta[k])
        np.testing.assert_array_equal(
            trace.u_nom[k], [0.4 + 0.3 * t, 0.2 - 0.5 * np.sin(theta[1] - theta[0])])
    np.testing.assert_array_equal(trace.theta[0], theta0)
    if decay is None:
        np.testing.assert_array_equal(trace.u, trace.u_nom)
    else:
        (filt,) = recorded_filters
        for k, call in enumerate(filt.calls):
            np.testing.assert_array_equal(call["u_nom"], trace.u_nom[k])
    np.testing.assert_allclose(trace.theta[1:], trace.theta[:-1] + 0.1 * trace.u, rtol=0, atol=1e-15)


@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((2, 1)), np.array([0.0, np.nan]),
                                 np.array([np.inf, 0.0])])
def test_rollout_rejects_bad_controller_output(bad):
    with pytest.raises(ValueError, match=r"must return a finite array of shape \(2,\)"):
        rollout(TwoLinkArm(), lambda t, theta: bad, THETA0, dt=0.1, horizon=0.3, decay="matrix")


def test_control_deviation_defaults_to_the_recorded_command():
    def controller(t, theta):
        return np.array([1.0 + t, -0.5])

    trace = rollout(TwoLinkArm(), controller, THETA0, dt=0.1, horizon=0.5, decay="standard")
    np.testing.assert_array_equal(trace.control_deviation(),
                                  np.linalg.norm(trace.u - trace.u_nom, axis=1))
    np.testing.assert_array_equal(trace.control_deviation(trace.u_nom), trace.control_deviation())
    summary = summarize(trace)
    assert summary["control_deviation_l2"] == pytest.approx(
        np.sqrt(0.1 * np.sum(trace.control_deviation() ** 2)), rel=1e-12)


def test_summarize_defaults_to_the_recorded_command(comparison):
    for name in NAMES:
        assert summarize(comparison[name]) == summarize(comparison[name], U_NOM)


def test_control_deviation_without_a_recorded_command():
    nan = np.full(1, np.nan)
    trace = Trace(
        name="hand", label="hand", decay=None, t=np.array([0.0, 1.0]), theta=np.zeros((2, 2)),
        u=np.array([[3.0, 4.0]]), margin=np.zeros(2), decay_deviation=nan, scalar_certificate=nan,
        safety_residual=nan, recovery_residual=nan, solve_ms=nan, Omega=np.zeros((1, 2, 2)),
    )
    assert trace.u_nom is None
    with pytest.raises(ValueError, match="no recorded nominal command"):
        trace.control_deviation()
    np.testing.assert_array_equal(trace.control_deviation((0.0, 0.0)), [5.0])


def test_run_comparison_with_a_controller():
    traces = run_comparison(u_nom=_constant_controller, dt=0.05, horizon=0.2)
    reference = run_comparison(dt=0.05, horizon=0.2)
    for name in NAMES:
        np.testing.assert_array_equal(traces[name].theta, reference[name].theta)
        np.testing.assert_array_equal(traces[name].u_nom, np.tile(U_NOM, (4, 1)))
