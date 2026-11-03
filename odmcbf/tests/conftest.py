"""Shared fixtures for the test suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")           # headless: figures are rendered, never shown

import numpy as np
import pytest

from odmcbf import TwoLinkArm, run_comparison

#: Admissible ``(l1, l2, sigma_bar)`` choices: equal links with the default
#: threshold, link 1 longer, link 2 longer, and ``sigma_bar`` near 0 and near
#: ``min(l1, l2)``. ``None`` is the default threshold, which puts the boundary
#: at ``theta2 - theta1 = pi/3``.
ARM_PARAMETERS = [
    (1.0, 1.0, None),
    (1.0, 0.6, 0.3),
    (0.5, 1.2, 0.35),
    (1.0, 1.0, 0.1),
    (1.5, 0.8, 0.78),
]

#: The notebook's closed-loop setup.
U_NOM = np.array([1.0, 0.4])
THETA0 = np.array([0.0, np.pi / 2])


def _arm_id(params) -> str:
    l1, l2, sigma_bar = params
    return f"l1={l1:g}-l2={l2:g}-sb={'default' if sigma_bar is None else f'{sigma_bar:g}'}"


@pytest.fixture(params=ARM_PARAMETERS, ids=_arm_id)
def arm(request) -> TwoLinkArm:
    """One arm per admissible ``(l1, l2, sigma_bar)`` choice."""
    l1, l2, sigma_bar = request.param
    return TwoLinkArm(l1, l2, sigma_bar=sigma_bar)


#: For the manuscript's formulas, where the class-K rate ``gamma`` appears: the
#: arms above with ``gamma = 1``, and two of them again with ``gamma = 2.5``.
MANUSCRIPT_ARM_PARAMETERS = [(*params, 1.0) for params in ARM_PARAMETERS] + [
    (1.0, 1.0, None, 2.5),
    (1.0, 0.6, 0.3, 2.5),
]


def _manuscript_arm_id(params) -> str:
    return f"{_arm_id(params[:3])}-gamma={params[3]:g}"


@pytest.fixture(params=MANUSCRIPT_ARM_PARAMETERS, ids=_manuscript_arm_id)
def manuscript_arm(request) -> TwoLinkArm:
    """One arm per ``(l1, l2, sigma_bar, gamma)`` of ``MANUSCRIPT_ARM_PARAMETERS``."""
    l1, l2, sigma_bar, gamma = request.param
    return TwoLinkArm(l1, l2, gamma=gamma, sigma_bar=sigma_bar)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def comparison():
    """The notebook's four closed loops, on a coarser step (``dt = 0.04``, about 0.5 s)."""
    return run_comparison(TwoLinkArm(), u_nom=U_NOM, theta0=THETA0, dt=0.04,
                          horizon=8.0, omega_d=1.0, rho_omega=5e-3, recovery=True)
