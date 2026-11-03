"""Planar two-link manipulability barrier in relative joint angles (not in the paper).

``q = (q1, q2)`` holds the base angle and the elbow angle measured from link 1, so
the link angles are ``(q1, q1 + q2)``. ``J``, ``H``, their derivatives and the
filters' control cost are all in terms of the relative joint rates ``qdot``.

:class:`RelativeTwoLinkArm` has the part of :class:`~odmcbf.TwoLinkArm`'s interface
that the filters, the rollouts and the plotting code use, so they work with it unchanged. ``main_relative.ipynb``
runs it.
"""

from __future__ import annotations

import numpy as np

#: Planar rotation generator; ``R(q1) = expm(q1 * S)``.
S = np.array([[0.0, -1.0], [1.0, 0.0]])

#: Squared singular-value threshold, placing the active boundary at q2 = pi/2.
SIGMA_BAR_SQ = (3.0 - np.sqrt(5.0)) / 2.0

#: Boundary elbow angle of the constraint above.
Q2_BOUNDARY = np.pi / 2


def jacobian(q) -> np.ndarray:
    """End-effector Jacobian ``J(q)`` for unit link lengths."""
    q1, q2 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-s1 - s12, -s12],
                     [c1 + c12, c12]])


def M(q) -> np.ndarray:
    """``M(q) = J(q) J(q).T``."""
    Jq = jacobian(q)
    return Jq @ Jq.T


def djacobian_dq2(q) -> np.ndarray:
    """``dJ / dq2``. The Jacobian depends on ``q1`` only through a rotation."""
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-c12, -c12],
                     [-s12, -s12]])


def dM_dq1(q) -> np.ndarray:
    """``dM / dq1 = S M - M S``, the generator of the rigid rotation."""
    Mq = M(q)
    return S @ Mq - Mq @ S


def dM_dq2(q) -> np.ndarray:
    """``dM / dq2``."""
    X = djacobian_dq2(q) @ jacobian(q).T
    return X + X.T


def arm_points(q) -> np.ndarray:
    """Base, elbow and end-effector positions, as a ``3 x 2`` array."""
    theta = np.cumsum(q)
    links = np.column_stack([np.cos(theta), np.sin(theta)])
    return np.vstack([np.zeros(2), np.cumsum(links, axis=0)])


class RelativeTwoLinkArm:
    """The singularity-avoidance constraint, in the form the filters expect."""

    n_controls = 2
    dim = 2

    def __init__(self, gamma: float = 1.0, sigma_bar_sq: float = SIGMA_BAR_SQ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = float(gamma)
        self.sigma_bar_sq = float(sigma_bar_sq)

    def H(self, q) -> np.ndarray:
        """Matrix barrier ``H(q) = M(q) - sigma_bar**2 I``."""
        return M(q) - self.sigma_bar_sq * np.eye(2)

    def alpha_H(self, q) -> np.ndarray:
        """Spectral class-K term ``alpha(H(q)) = gamma H(q)``."""
        return self.gamma * self.H(q)

    def LgH(self, q):
        """Control directions ``[dM / dq1, dM / dq2]``; the constraint is driftless."""
        return [dM_dq1(q), dM_dq2(q)]

    def Hdot(self, q, u) -> np.ndarray:
        """``Hdot(q, u) = u1 dM/dq1 + u2 dM/dq2``."""
        G1, G2 = self.LgH(q)
        return u[0] * G1 + u[1] * G2

    def margin(self, q) -> float:
        """Safety margin ``lambda_min(H(q))``; nonnegative inside the safe set."""
        return float(np.linalg.eigvalsh(self.H(q))[0])

    # Used by the animation and snapshot figures.
    reach = 2.0
    l1 = 1.0
    l2 = 1.0

    @property
    def sigma_bar(self):
        return float(np.sqrt(self.sigma_bar_sq))

    def jacobian(self, q):
        return jacobian(q)

    def M(self, q):
        return M(q)

    def arm_points(self, q):
        return arm_points(q)

    def __repr__(self):
        return (f"RelativeTwoLinkArm(gamma={self.gamma:g}, "
                f"sigma_bar_sq={self.sigma_bar_sq:.6g})")
