"""The manuscript's eigenbasis blocks and its constructive certificates, for the tests.

At a state with ``H >= 0``, the manuscript's eq. (11) works in an orthonormal
eigenbasis ``V = [V0 V1]`` of ``H``, with ``V0`` spanning ``ker H`` and ``V1``
the positive eigenspace, so that

    V.T H V = diag(0, Lambda1),   V.T alpha(H) V = diag(0, A),
    V.T Hdot V = [[Z00, Z01], [Z01.T, Z11]].

The helpers below compute these blocks, the multipliers built in the proofs of
Theorem 1 and Proposition 3, and the conditions of Propositions 1 and 2. They
are written from the manuscript, independently of :mod:`odmcbf.filters`.
``certify`` and ``scalar_never_certifies`` at the end are the only ones that
call the filters.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pytest

from odmcbf import FilterInfeasible, FilterResult, SafetyFilter

#: Why a case is skipped when CLARABEL fails outright and SCS is not there to decide instead.
SOLVER_FAILED = "CLARABEL failed outright on this weakly infeasible problem, and SCS is not installed"


def sym(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + X.T)


def lambda_min(X: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(sym(X))[0])


@dataclass(frozen=True)
class Blocks:
    """Eq. (11) at one state: the eigenbasis split and the blocks of ``Hdot``."""

    V0: np.ndarray      # (p, k) orthonormal basis of ker H
    V1: np.ndarray      # (p, p - k) orthonormal basis of the positive eigenspace
    A: np.ndarray       # (p - k, p - k) V1.T alpha(H) V1, positive definite
    Z00: np.ndarray     # (k, k)
    Z01: np.ndarray     # (k, p - k)
    Z11: np.ndarray     # (p - k, p - k)

    @property
    def V(self) -> np.ndarray:
        return np.hstack([self.V0, self.V1])

    def to_matrix(self, Omega00, Omega01, Omega10, Omega11) -> np.ndarray:
        """``Omega`` from its blocks in the basis ``V``, back in the original basis."""
        blocks = np.block([[np.atleast_2d(Omega00), np.atleast_2d(Omega01)],
                           [np.atleast_2d(Omega10), np.atleast_2d(Omega11)]])
        return self.V @ blocks @ self.V.T

    def in_basis(self, X: np.ndarray) -> np.ndarray:
        """``V.T X V``."""
        return self.V.T @ X @ self.V


def blocks(H: np.ndarray, alpha_H: np.ndarray, Hdot: np.ndarray, kernel_tol: float = 1e-9) -> Blocks:
    """Split at a state with ``H >= 0``; eigenvalues below ``kernel_tol * ||H||`` form the kernel."""
    lam, V = np.linalg.eigh(sym(H))
    scale = max(1.0, float(np.abs(lam).max()))
    if lam[0] < -kernel_tol * scale:
        raise ValueError("blocks() needs H >= 0")
    k = int(np.sum(lam <= kernel_tol * scale))
    V0, V1 = V[:, :k], V[:, k:]
    Z = V.T @ sym(Hdot) @ V
    return Blocks(V0=V0, V1=V1, A=sym(V1.T @ alpha_H @ V1),
                  Z00=Z[:k, :k], Z01=Z[:k, k:], Z11=Z[k:, k:])


def od_lmi(Hdot: np.ndarray, alpha_H: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """The matrix of eq. (16), ``Hdot + Omega.T alpha(H) + alpha(H) Omega``."""
    return Hdot + Omega.T @ alpha_H + alpha_H @ Omega


def theorem1_certificate(b: Blocks, c: float, Omega00=None, Omega01=None) -> np.ndarray:
    """``Omega10 = -A^{-1} Z01.T``, ``Omega11 = c A^{-1}``, the other blocks free (proof of Theorem 1)."""
    k, r = b.V0.shape[1], b.V1.shape[1]
    A_inv = np.linalg.inv(b.A)
    Omega00 = np.zeros((k, k)) if Omega00 is None else Omega00
    Omega01 = np.zeros((k, r)) if Omega01 is None else Omega01
    return b.to_matrix(Omega00, Omega01, -A_inv @ b.Z01.T, c * A_inv)


def proposition3_certificate(b: Blocks, c: float, mu: float) -> np.ndarray:
    """Theorem 1's choice with ``Omega01 = -Omega10.T`` and ``Omega00 = mu I`` (proof of Proposition 3)."""
    k = b.V0.shape[1]
    A_inv = np.linalg.inv(b.A)
    Omega10 = -A_inv @ b.Z01.T
    return b.to_matrix(mu * np.eye(k), -Omega10.T, Omega10, c * A_inv)


def large_enough_c(b: Blocks, margin: float = 1.0) -> float:
    """A ``c > 0`` with ``Z11 + 2 c I`` positive definite, with ``margin`` to spare."""
    return max(0.0, -lambda_min(b.Z11)) / 2 + margin


def proposition1_holds(b: Blocks, tol: float = 1e-9) -> bool:
    """Eq. (12): (i) ``Z00 >= 0`` and (ii) ``(I - Z00 Z00^+) Z01 = 0``.

    For symmetric ``Z00``, ``I - Z00 Z00^+`` projects onto ``ker Z00``. The
    kernel is taken with an absolute threshold, ``tol`` times the scale of
    ``Hdot``'s blocks: a ``Z00`` that is zero up to rounding has no range,
    whereas ``np.linalg.pinv`` would invert the rounding.
    """
    k = b.Z00.shape[0]
    if k == 0:
        return True
    scale = max(1.0, float(np.abs(b.Z00).max()), float(np.abs(b.Z01).max(initial=0.0)))
    lam, U = np.linalg.eigh(sym(b.Z00))
    if lam[0] < -tol * scale:
        return False
    kernel = U[:, lam <= tol * scale]
    residual = kernel @ (kernel.T @ b.Z01)
    return bool(np.abs(residual).max(initial=0.0) <= tol * scale)


def proposition2_bound(b: Blocks) -> float:
    """Eq. (13): the smallest scalar decay, for ``H > 0`` or ``Z00 > 0``."""
    A_inv_sqrt = _inv_sqrt(b.A)
    coupling = (b.Z01.T @ np.linalg.solve(b.Z00, b.Z01)
                if b.Z00.size else np.zeros_like(b.Z11))
    return max(0.0, float(np.linalg.eigvalsh(sym(A_inv_sqrt @ (coupling - b.Z11) @ A_inv_sqrt))[-1]))


def proposition2_blowup_bound(b: Blocks) -> float:
    """Eq. (14): ``(||Z01.T v||**2 / lambda_min(Z00) - lambda_max(Z11)) / lambda_max(A)``."""
    lam, vecs = np.linalg.eigh(sym(b.Z00))
    w = b.Z01.T @ vecs[:, 0]
    return (float(w @ w) / lam[0] - np.linalg.eigvalsh(b.Z11)[-1]) / np.linalg.eigvalsh(b.A)[-1]


def _inv_sqrt(A: np.ndarray) -> np.ndarray:
    lam, V = np.linalg.eigh(sym(A))
    return V @ np.diag(lam ** -0.5) @ V.T


# -------------------------------------------------------- queries to the filters
def certify(decay: str, alpha_H: np.ndarray, Hdot: np.ndarray, **kwargs) -> FilterResult:
    """Solve the filter with no control authority: does a multiplier certify ``Hdot``?

    With ``LgH = 0`` the control drops out, and the filter only decides whether
    a decay multiplier of the given mode certifies ``Hdot = LfH``.
    """
    p = alpha_H.shape[0]
    filt = SafetyFilter(2, p, decay=decay, **kwargs)
    return filt(alpha_H, [np.zeros((p, p))] * 2, np.zeros(2), LfH=Hdot)


def scalar_never_certifies(alpha_H: np.ndarray, Hdot: np.ndarray, **kwargs) -> None:
    """The scalar filter raises, or returns an ``omega`` whose LMI is violated.

    In the gap of Proposition 1 the scalar problem is only weakly infeasible:
    a large ``omega`` brings ``lambda_min`` arbitrarily close to zero, and the
    solver may then return ``optimal_inaccurate``. The residual is recomputed
    from the returned ``omega``, so its sign is exact up to the rounding of
    ``eigvalsh``, and it must be negative beyond that rounding. No fixed
    tolerance would do: the residual can be as small as ``1e-10``.

    CLARABEL sometimes fails outright on these problems, and the filter passes
    its ``SolverError`` through. SCS then decides instead, and without SCS the
    case is skipped.
    """
    try:
        res = certify("scalar", alpha_H, Hdot, **kwargs)
    except FilterInfeasible:
        return
    except cp.error.SolverError:
        if "SCS" not in cp.installed_solvers():
            pytest.skip(SOLVER_FAILED)
        try:
            res = certify("scalar", alpha_H, Hdot, solver="SCS", **kwargs)
        except FilterInfeasible:
            return
    X = Hdot + res.omega * alpha_H
    assert res.safety_residual == pytest.approx(lambda_min(X), rel=1e-6, abs=1e-15)
    assert res.safety_residual < -1e-13 * np.linalg.norm(X, 2)
