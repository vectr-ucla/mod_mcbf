"""Singularity avoidance for a planar two-link arm, in absolute joint angles.

This module implements the barrier of ``two_link_manipulator_kinematics.md``.
Both joint angles are absolute, ``theta = (theta1, theta2)``, measured
counter-clockwise from the horizontal, ``l1`` and ``l2`` are the link lengths,
and the arm is velocity controlled, ``thetadot = u``. Joint 1 is at the
origin, joint 2 at ``l1 (cos theta1, sin theta1)``, and the end effector at

    p(theta) = l1 (cos theta1, sin theta1) + l2 (cos theta2, sin theta2),

with Jacobian

    J(theta) = dp / dtheta = [[-l1 sin theta1, -l2 sin theta2],
                              [ l1 cos theta1,  l2 cos theta2]].

With ``M = J J.T`` the matrix constraint

    H(theta) = M(theta) - sigma_bar**2 I >= 0   <=>   sigma_min(J) >= sigma_bar

keeps the arm clear of singular configurations. The constraint is driftless,

    Hdot(theta, u) = u1 dM/dtheta1 + u2 dM/dtheta2,
    dM/dtheta_i = l_i**2 [[2 s_i c_i,       s_i**2 - c_i**2],
                          [s_i**2 - c_i**2, -2 s_i c_i     ]],

with ``s_i = sin theta_i`` and ``c_i = cos theta_i``, and the class-K term
is linear, ``alpha(H) = gamma H``.

The spectrum of ``H`` depends on ``theta`` only through ``theta2 - theta1``,
since ``tr M = l1**2 + l2**2`` and ``det M = (l1 l2 sin(theta2 - theta1))**2``.
Advancing both joints together rotates the whole arm about joint 1, which
conjugates ``M`` and ``H`` by a rotation and leaves the safety margin
unchanged. That motion is the one a scalar decay variable cannot certify on
the boundary of the safe set.

The module is organised in two layers:

:class:`TwoLinkKinematics`
    Geometry only: positions, the Jacobian, ``M`` and its joint derivatives,
    and the closed-form spectrum of ``M`` as a function of ``theta2 - theta1``.
:class:`TwoLinkArm`
    The kinematics plus the barrier ``H``, in the form the safety filters
    expect (``alpha_H``, ``LgH``, ``Hdot``, ``n_controls``, ``dim``).
"""

from __future__ import annotations

import numpy as np

#: Value of ``theta2 - theta1`` at which the default threshold puts the
#: boundary. The safe set is then ``pi/3 <= |theta2 - theta1| <= 2 pi/3``
#: (mod ``2 pi``) for any link lengths.
DEFAULT_BOUNDARY = np.pi / 3


def band_half_width(l1: float, l2: float, sigma_bar: float) -> float:
    """Half-width, in ``theta2 - theta1``, of each band where ``H`` is not positive definite.

    For ``0 <= sigma_bar < min(l1, l2)`` the set ``lambda_min(H) <= 0`` is
    ``|theta2 - theta1| <= w`` or ``|theta2 - theta1| >= pi - w`` (mod ``2 pi``),
    where ``w`` in ``[0, pi/2)`` has

        cos(w)**2 = (l1**2 - sigma_bar**2) (l2**2 - sigma_bar**2) / (l1**2 l2**2),
        sin(w)**2 = sigma_bar**2 (l1**2 + l2**2 - sigma_bar**2) / (l1**2 l2**2).

    With ``a = sigma_bar / l1``, ``b = sigma_bar / l2`` and
    ``c_i = (l_i - sigma_bar) (l_i + sigma_bar) / l_i**2``, these are
    ``cos(w) = sqrt(c1 c2)`` and ``sin(w) = hypot(a, b sqrt(c1))``. The factors
    ``l_i -+ sigma_bar`` are exact near ``sigma_bar = l_i``, ``hypot`` never
    squares the small ratios, and ``arctan2`` combines the two. That keeps
    full relative precision for small ``sigma_bar``, down to subnormal
    ratios, and for ``sigma_bar`` near ``min(l1, l2)``, where ``arccos`` of
    the first expression would cancel. For ``sigma_bar >= min(l1, l2)`` the
    set is every configuration and ``pi/2`` is returned.
    """
    if sigma_bar >= min(l1, l2):
        return float(np.pi / 2)
    c1 = max(((l1 - sigma_bar) / l1) * ((l1 + sigma_bar) / l1), 0.0)
    c2 = max(((l2 - sigma_bar) / l2) * ((l2 + sigma_bar) / l2), 0.0)
    sin_w = np.hypot(sigma_bar / l1, (sigma_bar / l2) * np.sqrt(c1))
    return float(np.arctan2(sin_w, np.sqrt(c1) * np.sqrt(c2)))


def _dM_dtheta(length: float, angle: float) -> np.ndarray:
    """``d/dtheta_i`` of ``l_i**2 e(theta_i) e(theta_i).T``, ``e = (-sin, cos)``."""
    s, c = np.sin(angle), np.cos(angle)
    return length ** 2 * np.array([[2 * s * c, s * s - c * c],
                                   [s * s - c * c, -2 * s * c]])


class TwoLinkKinematics:
    """Geometry of a planar two-link arm in absolute joint angles.

    ``theta = (theta1, theta2)`` holds the angles of both links, each measured
    counter-clockwise from the horizontal, so ``theta2 = theta1`` is the fully
    stretched arm and ``theta2 = theta1 + pi`` the folded one. Every spectral
    quantity of ``M = J J.T`` is a function of ``theta2 - theta1`` alone and
    is given in closed form below.
    """

    def __init__(self, l1: float = 1.0, l2: float = 1.0) -> None:
        if not (np.isfinite(l1) and np.isfinite(l2)) or l1 <= 0 or l2 <= 0:
            raise ValueError("link lengths must be positive and finite")
        self.l1 = float(l1)
        self.l2 = float(l2)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(l1={self.l1:g}, l2={self.l2:g})"

    @property
    def reach(self) -> float:
        """Largest distance of the end effector from joint 1, ``l1 + l2``."""
        return self.l1 + self.l2

    @property
    def workspace_radii(self) -> tuple[float, float]:
        """Inner and outer radius of the workspace, ``(|l1 - l2|, l1 + l2)``.

        ``|p|**2 = l1**2 + l2**2 + 2 l1 l2 cos(theta2 - theta1)``, so the
        workspace is the annulus between these radii. Its outer circle is
        reached only with ``theta2 = theta1`` and its inner circle only with
        ``theta2 = theta1 + pi``, where ``J`` is singular.
        """
        return abs(self.l1 - self.l2), self.l1 + self.l2

    # ------------------------------------------------------------ positions
    def joint2(self, theta) -> np.ndarray:
        """Position of joint 2, ``l1 (cos theta1, sin theta1)``."""
        theta1 = float(theta[0])
        return self.l1 * np.array([np.cos(theta1), np.sin(theta1)])

    def end_effector(self, theta) -> np.ndarray:
        """End-effector position ``p(theta)``."""
        theta2 = float(theta[1])
        return self.joint2(theta) + self.l2 * np.array([np.cos(theta2), np.sin(theta2)])

    def arm_points(self, theta) -> np.ndarray:
        """Joint 1, joint 2 and end-effector positions, as a ``3 x 2`` array."""
        return np.vstack([np.zeros(2), self.joint2(theta), self.end_effector(theta)])

    # ------------------------------------------------ Jacobian and M = J J.T
    def jacobian(self, theta) -> np.ndarray:
        """End-effector Jacobian ``J(theta) = dp / dtheta``."""
        theta1, theta2 = float(theta[0]), float(theta[1])
        l1, l2 = self.l1, self.l2
        return np.array([[-l1 * np.sin(theta1), -l2 * np.sin(theta2)],
                         [l1 * np.cos(theta1), l2 * np.cos(theta2)]])

    def M(self, theta) -> np.ndarray:
        """``M(theta) = J(theta) J(theta).T``."""
        Jt = self.jacobian(theta)
        return Jt @ Jt.T

    def dM_dtheta1(self, theta) -> np.ndarray:
        """``dM / dtheta1``; it depends on ``theta1`` alone."""
        return _dM_dtheta(self.l1, float(theta[0]))

    def dM_dtheta2(self, theta) -> np.ndarray:
        """``dM / dtheta2``; it depends on ``theta2`` alone."""
        return _dM_dtheta(self.l2, float(theta[1]))

    # ------------------------------- spectrum of M versus theta2 - theta1
    def singular_values_sq(self, theta_diff):
        """Eigenvalues of ``M`` as ``(sigma_min**2, sigma_max**2)``; accepts arrays.

        ``theta_diff`` is ``theta2 - theta1``. The eigenvalues depend on it
        alone, through

            tr M = l1**2 + l2**2,
            det M = (l1 l2 sin(theta2 - theta1))**2.

        The smaller one is evaluated as ``det M / sigma_max**2``, which avoids
        the cancellation in ``(tr M - sqrt(tr M**2 - 4 det M)) / 2`` near the
        singular configurations, where it vanishes.
        """
        theta_diff = np.asarray(theta_diff, dtype=float)
        l1, l2 = self.l1, self.l2
        trace = l1 ** 2 + l2 ** 2
        gap = np.sqrt((l1 ** 2 - l2 ** 2) ** 2
                      + (2 * l1 * l2 * np.cos(theta_diff)) ** 2)
        mu_max = 0.5 * (trace + gap)          # >= tr M / 2 > 0
        det = (l1 * l2 * np.sin(theta_diff)) ** 2
        return det / mu_max, mu_max

    def sigma_min_sq(self, theta_diff):
        """``sigma_min(J)**2`` as a function of ``theta2 - theta1``; accepts arrays."""
        return self.singular_values_sq(theta_diff)[0]

    @property
    def max_sigma_min_sq(self) -> float:
        """Largest value of ``sigma_min(J)**2``, ``min(l1**2, l2**2)``.

        It is attained at ``theta2 - theta1 = +-pi/2``. Any threshold
        ``sigma_bar < min(l1, l2)`` leaves the safe set an interior.
        """
        return min(self.l1, self.l2) ** 2


class TwoLinkArm(TwoLinkKinematics):
    """A two-link arm with the singularity-avoidance constraint attached.

    Parameters
    ----------
    l1, l2:
        Link lengths.
    gamma:
        Rate of the linear class-K term ``alpha(H) = gamma H``.
    sigma_bar:
        Lower bound ``sigma_bar`` on the smallest singular value of ``J``. It
        must lie strictly between zero and ``min(l1, l2)`` for the safe set to
        have an interior. By default it is ``sigma_min(J)`` at
        ``theta2 - theta1 = DEFAULT_BOUNDARY``, which puts the boundary there.

    The class exposes the constraint in the form the filters expect:
    ``n_controls``, ``dim``, ``alpha_H``, ``LgH`` and ``Hdot``.
    """

    n_controls = 2
    dim = 2

    def __init__(
        self,
        l1: float = 1.0,
        l2: float = 1.0,
        *,
        gamma: float = 1.0,
        sigma_bar: float | None = None,
    ) -> None:
        super().__init__(l1, l2)
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be positive")
        if sigma_bar is None:
            sigma_bar = float(np.sqrt(self.sigma_min_sq(DEFAULT_BOUNDARY)))
        upper = min(self.l1, self.l2)
        if not 0.0 < sigma_bar < upper:
            raise ValueError(
                f"sigma_bar must lie in (0, {upper:g}) = (0, min(l1, l2)); "
                f"got {sigma_bar:g}"
            )
        self.gamma = float(gamma)
        self.sigma_bar = float(sigma_bar)

    @classmethod
    def from_boundary(
        cls,
        boundary: float,
        l1: float = 1.0,
        l2: float = 1.0,
        *,
        gamma: float = 1.0,
    ) -> TwoLinkArm:
        """Arm whose safe set has its boundary at ``theta2 - theta1 = boundary``.

        The threshold is ``sigma_bar = sigma_min(J)`` at that value. Because
        the margin depends on ``cos(theta2 - theta1)**2``, the boundary is
        also at ``pi - boundary`` and at the negatives of both. ``boundary``
        may not be a multiple of ``pi / 2``: at ``theta2 - theta1 = 0`` or
        ``pi`` the arm is singular, and at ``+-pi / 2`` the safe set has no
        interior.
        """
        boundary = float(boundary)
        if not np.isfinite(boundary) or np.isclose(np.sin(2 * boundary), 0.0, atol=1e-12):
            raise ValueError(
                f"boundary must not be a multiple of pi/2; got {boundary:g}"
            )
        sigma_bar = float(np.sqrt(TwoLinkKinematics(l1, l2).sigma_min_sq(boundary)))
        return cls(l1, l2, gamma=gamma, sigma_bar=sigma_bar)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(l1={self.l1:g}, l2={self.l2:g}, "
                f"gamma={self.gamma:g}, sigma_bar={self.sigma_bar:.6g})")

    @property
    def sigma_bar_sq(self) -> float:
        """``sigma_bar**2``."""
        return self.sigma_bar ** 2

    @property
    def safe_theta_diff_interval(self) -> tuple[float, float]:
        """Bounds ``(lo, hi)`` in ``(0, pi)`` of ``|theta2 - theta1|`` on the safe set.

        The safe set, ``det H >= 0``, is

            cos(theta2 - theta1)**2 <= (l1**2 - sigma_bar**2) (l2**2 - sigma_bar**2) / (l1**2 l2**2),

        that is ``lo <= |theta2 - theta1| <= hi`` (mod ``2 pi``) with
        ``hi = pi - lo``; ``theta1`` is unrestricted. By default
        ``lo = DEFAULT_BOUNDARY``.
        """
        lo = band_half_width(self.l1, self.l2, self.sigma_bar)
        return lo, float(np.pi - lo)

    @property
    def safe_radius_interval(self) -> tuple[float, float]:
        """Bounds ``(r_lo, r_hi)`` on ``|p|`` over the safe set.

        With ``|p|**2 = l1**2 + l2**2 + 2 l1 l2 cos(theta2 - theta1)``, the
        safe set ``det H >= 0`` reads, in the workspace,

            (|p|**2 - l1**2 - l2**2)**2 <= 4 (l1**2 - sigma_bar**2) (l2**2 - sigma_bar**2),

        the annulus ``r_lo <= |p| <= r_hi`` with

            r_lo**2, r_hi**2 = l1**2 + l2**2 -+ 2 sqrt((l1**2 - sigma_bar**2) (l2**2 - sigma_bar**2)).

        The two configurations reaching a point ``p`` have opposite values of
        ``theta2 - theta1`` and hence the same margin, so whether ``H >= 0``
        holds depends on ``p`` alone. ``H`` is violated on the rest of the
        workspace: a band next to its outer circle and a band next to its
        inner circle (``workspace_radii``). The boundary circles are the paths
        of the end effector when the whole arm rotates about joint 1.

        ``r_lo`` is evaluated from ``r_lo**2 r_hi**2 = (l1**2 - l2**2)**2 +
        4 sigma_bar**2 (l1**2 + l2**2 - sigma_bar**2)``, which avoids the
        cancellation in the formula above for small ``sigma_bar``.
        """
        l1, l2, r = self.l1, self.l2, self.sigma_bar_sq
        root = np.sqrt((l1 ** 2 - r) * (l2 ** 2 - r))
        r_hi_sq = l1 ** 2 + l2 ** 2 + 2 * root
        r_lo = np.sqrt((((l1 - l2) * (l1 + l2)) ** 2 + 4 * r * (l1 ** 2 + l2 ** 2 - r))
                       / r_hi_sq)
        # Rounding only: exactly, |l1 - l2| < r_lo < r_hi < l1 + l2.
        r_lo = max(r_lo, abs(l1 - l2))
        r_hi = min(np.sqrt(r_hi_sq), l1 + l2)
        return float(r_lo), float(r_hi)

    # ------------------------------------------------------------- barrier
    def H(self, theta) -> np.ndarray:
        """Matrix barrier ``H(theta) = M(theta) - sigma_bar**2 I``."""
        return self.M(theta) - self.sigma_bar_sq * np.eye(self.dim)

    def alpha_H(self, theta) -> np.ndarray:
        """Spectral class-K term ``alpha(H(theta)) = gamma H(theta)``."""
        return self.gamma * self.H(theta)

    def LgH(self, theta) -> list[np.ndarray]:
        """Control directions ``[dM/dtheta1, dM/dtheta2]``; the constraint is driftless."""
        return [self.dM_dtheta1(theta), self.dM_dtheta2(theta)]

    def Hdot(self, theta, u) -> np.ndarray:
        """``Hdot(theta, u) = u1 dM/dtheta1 + u2 dM/dtheta2``."""
        G1, G2 = self.LgH(theta)
        return float(u[0]) * G1 + float(u[1]) * G2

    def margin(self, theta) -> float:
        """Safety margin ``lambda_min(H(theta))``; nonnegative inside the safe set."""
        return float(np.linalg.eigvalsh(self.H(theta))[0])

    def margin_rate(self, theta, u) -> float:
        """``d/dt lambda_min(H)`` under ``thetadot = u``, that is ``v0.T Hdot(theta, u) v0``.

        ``v0`` is the unit eigenvector of ``lambda_min(H(theta))``. The derivative
        exists where that eigenvalue is simple. Since ``tr H = l1**2 + l2**2 -
        2 sigma_bar**2 > 0``, it is simple wherever the margin is below
        ``tr H / 2``, in particular outside the safe set and on its boundary. The
        eigenvalues coincide only for ``l1 = l2`` at ``theta2 - theta1 = +-pi/2``,
        where this raises ``ValueError``.
        """
        lam, vectors = np.linalg.eigh(self.H(theta))
        if lam[1] - lam[0] <= 1e-12 * abs(lam[1]):
            raise ValueError("lambda_min(H) is repeated here, so it has no derivative")
        v0 = vectors[:, 0]
        return float(v0 @ self.Hdot(theta, u) @ v0)
