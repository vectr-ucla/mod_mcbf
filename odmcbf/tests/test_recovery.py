"""The recovery constraint in closed loop: ``band_start``, ``hold_at_start``, ``run_recovery``, ``recovery_rate``.

The paper's Recovery Constraint section (Section VII, eq. (25) and Proposition 3):
where ``lambda_min(H) < 0``, the matrix inequality along its unit eigenvector
``v0`` reads ``d/dt lambda_min(H) >= 2 gamma |lambda_min(H)| v0' sym(Omega) v0``
for ``alpha(H) = gamma H``, so under ``sym(Omega) >= beta I`` the margin rises at
least at the relative rate ``2 beta gamma``. ``test_filters.py`` checks this bound
at single states. Here the arm runs closed loop from rest in a band where ``H``
is violated, tracking a target that stays at the start: the nominal command never
raises the margin, so whatever brings the arm back to the safe set is the
filter's decay term alone.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np
import pytest

from odmcbf import (
    BANDS,
    NO_RECOVERY,
    RECOVERY_BETAS,
    TwoLinkArm,
    band_start,
    hold_at_start,
    recovery_rate,
    rollout,
    run_recovery,
)

DT = 0.02
HORIZON = 2.0
DEPTH = 0.3
GAIN = 5.0
OMEGA_D = 1.0
RHO_OMEGA = 5e-3
#: The recovery margins run: the defaults, ``omega_d / 2`` and half of it, and one above ``omega_d / 2``.
BETAS = (*RECOVERY_BETAS, 1.0)
#: Relative tolerance on a recovery rate that the filter should meet exactly.
RATE_TOL = 1e-3
#: Slack on the recovery constraint ``lambda_min(sym Omega) >= beta``.
RECOVERY_TOL = 1e-5
#: Slack on the first-order bound ``d/dt lambda_min(H) >= rate |lambda_min(H)|``, which the
#: filters meet up to their solver's tolerance on the matrix inequality, about 1e-9.
BOUND_TOL = 1e-8
#: The hero arm, ``odmcbf.paper.HERO_ARM``.
HERO = (1.0, 0.75, 0.6)
#: Arms of the ``arm`` fixture whose bands are deep enough that recovering costs
#: the filter more control deviation than dropping the decay: all but
#: ``sigma_bar = 0.1``, whose start is only 0.003 into the band. And the hero arm.
DEEP_BANDS = [(1.0, 1.0, None), (1.0, 0.6, 0.3), (0.5, 1.2, 0.35), (1.5, 0.8, 0.78), HERO]


def _key(beta: float) -> str:
    return f"beta={beta:g}"


@functools.lru_cache(maxsize=None)
def _cached_runs(l1, l2, sigma_bar, gamma, band, gain):
    arm = TwoLinkArm(l1, l2, gamma=gamma, sigma_bar=sigma_bar)
    return run_recovery(arm, band, DEPTH, gain, DT, HORIZON, betas=BETAS,
                        omega_d=OMEGA_D, rho_omega=RHO_OMEGA)


def _runs(arm: TwoLinkArm, band: str, gain: float = GAIN):
    """``run_recovery`` on ``arm`` at every beta of ``BETAS``, shared between tests with the same setup."""
    return _cached_runs(arm.l1, arm.l2, arm.sigma_bar, arm.gamma, band, gain)


def _meets_rate(trace, rate: float) -> bool:
    """``d/dt lambda_min(H) >= rate |lambda_min(H)|`` at every step, up to ``BOUND_TOL``."""
    excess = (recovery_rate(trace) - rate) * -trace.margin[:-1]
    return bool(np.nanmin(excess) >= -BOUND_TOL)


def _v0_sym_Omega_v0(trace) -> np.ndarray:
    """``v0' sym(Omega) v0`` at each step, ``v0`` the unit eigenvector of ``lambda_min(H)``."""
    values = []
    for theta, Omega in zip(trace.theta[:-1], trace.Omega):
        v0 = np.linalg.eigh(trace.arm.H(theta))[1][:, 0]
        values.append(v0 @ (0.5 * (Omega + Omega.T)) @ v0)
    return np.array(values)


# ---------------------------------------------------------------- band_start
@pytest.mark.parametrize("band", BANDS)
def test_band_start_lies_depth_into_the_band(arm, band):
    """``theta1 = 0``, ``0 < theta2 - theta1 < pi``, and ``|p|`` is ``depth`` of the band's width in."""
    theta = band_start(arm, band, DEPTH)
    r_in, r_out = arm.workspace_radii
    r_lo, r_hi = arm.safe_radius_interval
    lo, hi = arm.safe_theta_diff_interval
    diff = theta[1] - theta[0]
    assert theta[0] == 0.0
    radius = np.linalg.norm(arm.end_effector(theta))
    if band == "outer":
        assert 0 < diff < lo
        assert radius == pytest.approx(r_hi + DEPTH * (r_out - r_hi), rel=1e-12)
    else:
        assert hi < diff < math.pi
        assert radius == pytest.approx(r_lo - DEPTH * (r_lo - r_in), rel=1e-12)
    assert arm.margin(theta) < 0


@pytest.mark.parametrize("band", BANDS)
def test_band_start_goes_deeper_with_depth(band):
    """The margin falls with ``depth``, from zero at the boundary of the safe set."""
    arm = TwoLinkArm(*HERO[:2], sigma_bar=HERO[2])
    margins = [arm.margin(band_start(arm, band, depth)) for depth in (1e-9, 0.1, 0.3, 0.6, 0.9)]
    assert margins[0] == pytest.approx(0.0, abs=1e-8)
    assert all(a > b for a, b in zip(margins, margins[1:]))


@pytest.mark.parametrize("band, depth", [("middle", 0.3), ("outer", 0.0), ("inner", 1.0),
                                         ("outer", -0.1), ("inner", math.nan)])
def test_band_start_rejects_bad_band_and_depth(band, depth):
    with pytest.raises(ValueError):
        band_start(TwoLinkArm(), band, depth)


# ------------------------------------------------------------- hold_at_start
def test_hold_at_start_is_the_tracking_controller_with_a_fixed_target(arm, rng):
    """``J(theta) u_nom(t, theta) = k (p(theta0) - p(theta))`` for any ``t``; zero at ``theta0``."""
    theta0 = band_start(arm, "outer", DEPTH)
    u_nom = hold_at_start(arm, theta0, GAIN)
    assert np.linalg.norm(u_nom(0.0, theta0)) <= 1e-12
    assert np.linalg.norm(u_nom(3.7, theta0)) <= 1e-12
    p0 = arm.end_effector(theta0)
    for theta, t in zip(theta0 + rng.normal(scale=0.2, size=(20, 2)), rng.uniform(0, 5, size=20)):
        expected = GAIN * (p0 - arm.end_effector(theta))
        np.testing.assert_allclose(arm.jacobian(theta) @ u_nom(t, theta), expected, rtol=1e-12, atol=1e-14)


# ------------------------------------------------------------- recovery_rate
def test_recovery_rate_is_the_margin_rate_relative_to_the_margin():
    trace = _runs(TwoLinkArm(), "outer")[_key(0.5)]
    rate = recovery_rate(trace)
    assert rate.shape == (len(trace.u),)
    for k in (0, len(rate) // 2, len(rate) - 1):
        expected = trace.arm.margin_rate(trace.theta[k], trace.u[k]) / -trace.margin[k]
        assert rate[k] == pytest.approx(expected, rel=1e-14)


def test_recovery_rate_is_nan_inside_the_safe_set():
    """From ``theta2 - theta1 = pi/2``, the arm's largest margin, every entry is NaN."""
    trace = rollout(TwoLinkArm(), (0.3, 0.1), (0.0, math.pi / 2), dt=0.1, horizon=0.5, decay="matrix")
    assert np.all(trace.margin > 0)
    assert np.all(np.isnan(recovery_rate(trace)))


def test_recovery_rate_needs_the_arm():
    trace = dataclasses.replace(_runs(TwoLinkArm(), "outer")[NO_RECOVERY], arm=None)
    with pytest.raises(ValueError, match="no arm"):
        recovery_rate(trace)


# -------------------------------------------------------------- run_recovery
def test_run_recovery_runs_matrix_od_at_each_beta_and_without_recovery():
    arm = TwoLinkArm(*HERO[:2], sigma_bar=HERO[2])
    runs = _runs(arm, "outer")
    assert list(runs) == [_key(beta) for beta in BETAS] + [NO_RECOVERY]
    start = band_start(arm, "outer", DEPTH)
    steps = round(HORIZON / DT)
    for key, trace in runs.items():
        assert (trace.name, trace.decay, repr(trace.arm)) == (key, "matrix", repr(arm))
        assert trace.u.shape == (steps, 2)
        np.testing.assert_array_equal(trace.theta[0], start)
    for beta in BETAS:
        trace = runs[_key(beta)]
        assert trace.label == f"matrix OD, beta = {beta:g}"
        # the recorded slack is against this run's own beta
        assert np.all(np.isfinite(trace.recovery_residual))
        sym = 0.5 * (trace.Omega[0] + trace.Omega[0].T)
        assert trace.recovery_residual[0] == pytest.approx(np.linalg.eigvalsh(sym)[0] - beta, abs=1e-9)
    assert runs[NO_RECOVERY].label == "matrix OD, no recovery"
    assert np.all(np.isnan(runs[NO_RECOVERY].recovery_residual))


def test_run_recovery_defaults():
    """Unit links, the outer band, 0.3 deep, ``RECOVERY_BETAS``, and the filters' own defaults."""
    runs = run_recovery(dt=0.1, horizon=0.2)
    assert list(runs) == [_key(beta) for beta in RECOVERY_BETAS] + [NO_RECOVERY]
    trace = runs[_key(0.5)]
    assert (trace.arm.l1, trace.arm.l2) == (1.0, 1.0)
    np.testing.assert_array_equal(trace.theta[0], band_start(TwoLinkArm(), "outer", 0.3))
    assert trace.recovery_residual[0] == pytest.approx(0.0, abs=RECOVERY_TOL)


@pytest.mark.parametrize("kwargs", [dict(recovery=False), dict(beta=0.3), dict(decay="scalar")])
def test_run_recovery_sets_the_filter_itself(kwargs):
    with pytest.raises(TypeError, match=next(iter(kwargs))):
        run_recovery(**kwargs)


@pytest.mark.parametrize("betas", [(), (0.5, 0.0), (-0.5,), (0.5, math.nan), (0.5, 0.5)])
def test_run_recovery_rejects_bad_betas(betas):
    with pytest.raises(ValueError, match="betas"):
        run_recovery(betas=betas, dt=0.1, horizon=0.2)


# ------------------------------------------------------------- the closed loop
@pytest.mark.parametrize("band", BANDS)
def test_nominal_command_never_raises_the_margin(arm, band):
    """The target stays at the start: the command is zero there and never pushes out of the band.

    So the unfiltered arm stays where it starts, and in every filtered run the
    nominal command, at the run's own state, has ``d/dt lambda_min(H) <= 0``.
    """
    theta0 = band_start(arm, band, DEPTH)
    nominal = rollout(arm, hold_at_start(arm, theta0, GAIN), theta0, dt=DT, horizon=HORIZON)
    np.testing.assert_allclose(nominal.theta, np.broadcast_to(theta0, nominal.theta.shape), rtol=0, atol=1e-12)
    np.testing.assert_allclose(nominal.margin, nominal.margin[0], rtol=0, atol=1e-12)
    for trace in _runs(arm, band).values():
        assert np.linalg.norm(trace.u_nom[0]) <= 1e-12
        rates = [arm.margin_rate(theta, u) for theta, u in zip(trace.theta[:-1], trace.u_nom)]
        assert max(rates) <= 1e-12


@pytest.mark.parametrize("beta", [0.5, 1.0])
@pytest.mark.parametrize("band", BANDS)
def test_matrix_od_recovers_at_exactly_its_recovery_rate(manuscript_arm, band, beta):
    """For ``beta >= omega_d / 2`` the rate is ``2 beta gamma``, not more.

    The penalty pulls ``sym(Omega)`` toward ``Omega_d = (omega_d/2) I``, no higher
    than ``beta I``, and the matrix inequality along ``v0`` pulls it lower, so the
    recovery constraint binds; and since the nominal command pulls back into the
    band, the filter raises the margin no faster than it must. Per step the margin
    then shrinks by ``1 - 2 beta gamma dt``, up to terms of order ``dt**2``.
    """
    arm = manuscript_arm
    trace = _runs(arm, band)[_key(beta)]
    rate = 2 * beta * arm.gamma
    np.testing.assert_allclose(recovery_rate(trace), rate, rtol=RATE_TOL)
    np.testing.assert_allclose(trace.recovery_residual, 0.0, atol=RECOVERY_TOL)
    ratio = trace.margin[1:] / trace.margin[:-1]
    np.testing.assert_allclose(ratio, 1 - rate * DT, rtol=0, atol=(rate * DT) ** 2)


@pytest.mark.parametrize("band", BANDS)
def test_matrix_od_below_half_omega_d_meets_its_bound(arm, band):
    """``beta = 0.25 < omega_d / 2``: the penalty pulls ``sym(Omega)`` up toward ``Omega_d``, so only the
    bound ``2 beta gamma`` is guaranteed; on the thin bands of ``sigma_bar = 0.1`` the rate exceeds it."""
    trace = _runs(arm, band)[_key(0.25)]
    assert _meets_rate(trace, 0.5 * arm.gamma)
    assert np.nanmin(trace.recovery_residual) >= -RECOVERY_TOL


@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("params", DEEP_BANDS, ids=lambda p: f"l1={p[0]:g}-l2={p[1]:g}-sb={p[2]}")
def test_matrix_od_below_half_omega_d_takes_its_rate_on_deep_bands(params, band):
    """Where recovering costs control deviation, ``beta = 0.25`` is taken exactly: the rate is ``0.5 gamma``."""
    arm = TwoLinkArm(*params[:2], sigma_bar=params[2])
    trace = _runs(arm, band)[_key(0.25)]
    np.testing.assert_allclose(recovery_rate(trace), 0.5 * arm.gamma, rtol=RATE_TOL)
    np.testing.assert_allclose(trace.recovery_residual, 0.0, atol=RECOVERY_TOL)


@pytest.mark.parametrize("band", BANDS)
def test_margin_rises_monotonically_and_stays_negative(manuscript_arm, band):
    """Under the constraint the arm approaches the boundary of the safe set from outside, as the nominal
    command holds it back; the smaller ``beta``, the slower."""
    runs = _runs(manuscript_arm, band)
    for beta in BETAS:
        trace = runs[_key(beta)]
        assert np.all(np.diff(trace.margin) > 0)
        assert np.all(trace.margin < 0)
    ends = [runs[key].margin[-1] for key in (_key(1.0), _key(0.5), _key(0.25), NO_RECOVERY)]
    assert ends[0] > ends[1] > ends[2] > ends[3]


@pytest.mark.parametrize("gain", [1.0, 20.0])
def test_recovery_rate_does_not_depend_on_the_nominal_gain(gain):
    """However hard the nominal command pulls back, the margin rises at ``2 beta gamma``."""
    arm = TwoLinkArm(*HERO[:2], sigma_bar=HERO[2])
    runs = _runs(arm, "outer", gain=gain)
    for beta in BETAS:
        np.testing.assert_allclose(recovery_rate(runs[_key(beta)]), 2 * beta * arm.gamma, rtol=RATE_TOL)


@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("params", DEEP_BANDS, ids=lambda p: f"l1={p[0]:g}-l2={p[1]:g}-sb={p[2]}")
def test_matrix_od_stalls_without_the_recovery_constraint(params, band):
    """Without it, the filter drops the decay along ``v0`` and the arm comes to rest in the band.

    Recovering costs control deviation, and dropping the decay only the penalty
    on ``Omega - Omega_d``; with the nominal command pulling back, ``v0' sym(Omega)
    v0`` falls to near zero, and so does the rate. With the constraint at
    ``beta = omega_d / 2``, the same filter has taken the margin to
    ``(1 - gamma dt)^N`` of its start.
    """
    arm = TwoLinkArm(*params[:2], sigma_bar=params[2])
    runs = _runs(arm, band)
    recovered = (1 - arm.gamma * DT) ** len(runs[NO_RECOVERY].u)
    assert runs[_key(0.5)].margin[-1] / runs[_key(0.5)].margin[0] == pytest.approx(recovered, rel=0.02)
    free = runs[NO_RECOVERY]
    decay = _v0_sym_Omega_v0(free)
    assert decay[-1] <= 0.1 * decay[0]
    assert recovery_rate(free)[-1] <= 0.05 * arm.gamma
    assert np.linalg.norm(free.u[-1]) <= 5e-3
    assert free.margin[-1] / free.margin[0] >= 0.7
    assert np.all(free.margin < 0)


@pytest.mark.parametrize("band", BANDS)
def test_without_the_recovery_constraint_nothing_guarantees_the_rate(arm, band):
    """On every arm the filter falls below the rate ``beta = omega_d / 2`` would guarantee, even on the thin
    bands of ``sigma_bar = 0.1``, where recovering costs so little that the penalty keeps most of it."""
    assert np.nanmin(recovery_rate(_runs(arm, band)[NO_RECOVERY])) < 0.95 * OMEGA_D * arm.gamma


@pytest.mark.parametrize("band", BANDS)
def test_standard_and_scalar_od_recover_at_least_at_gamma(manuscript_arm, band):
    """Not drawn by the demo: under the same command the standard MCBF recovers at least at ``gamma`` and
    scalar OD with ``omega >= omega_d`` at least at ``omega_d gamma``; in the band a larger ``omega`` asks for
    a faster recovery along ``v0``, so scalar OD takes ``omega = omega_d`` and coincides with the standard MCBF."""
    arm = manuscript_arm
    theta0 = band_start(arm, band, DEPTH)
    u_nom = hold_at_start(arm, theta0, GAIN)
    traces = {decay: rollout(arm, u_nom, theta0, dt=DT, horizon=HORIZON, decay=decay, omega_d=OMEGA_D,
                             rho_omega=RHO_OMEGA, recovery=True) for decay in ("standard", "scalar")}
    for trace in traces.values():
        assert _meets_rate(trace, OMEGA_D * arm.gamma)
        assert np.all(np.diff(trace.margin) > 0)
        assert np.all(trace.margin < 0)
    omega_d_multiplier = np.broadcast_to(0.5 * OMEGA_D * np.eye(2), traces["scalar"].Omega.shape)
    np.testing.assert_allclose(traces["scalar"].Omega, omega_d_multiplier, rtol=0, atol=1e-6)
    np.testing.assert_allclose(traces["scalar"].margin, traces["standard"].margin, rtol=1e-4, atol=1e-9)
