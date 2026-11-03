"""Convex safety filters for matrix control barrier functions.

Three decay families are supported for the matrix barrier condition, all of
which are semidefinite programs in the control ``u`` and a decay multiplier:

``standard``
    ``Hdot + alpha(H) >= 0``, the fixed-decay MCBF condition.
``scalar``
    ``Hdot + omega * alpha(H) >= 0`` with ``omega >= 0``, scalar optimal decay.
``matrix``
    ``Hdot + Omega.T @ alpha(H) + alpha(H) @ Omega >= 0`` with a full matrix
    multiplier ``Omega``, the optimal-decay condition proposed in the paper.

The filters are written for a general control-affine system: the caller
supplies the spectral class-K term ``alpha(H(x))``, the drift ``Lf H(x)``, and
the control directions ``Lg_j H(x)``, so that

    Hdot(x, u) = Lf H(x) + sum_j u_j * Lg_j H(x).

The problem is built once and per-step data enters through CVXPY parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cvxpy as cp
import numpy as np

DECAY_MODES = ("standard", "scalar", "matrix")

#: Tight solver tolerances, so that the reported LMI residuals are meaningful.
MOSEK_PARAMS = {
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-9,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-9,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-9,
}
CLARABEL_PARAMS = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9,
                   "tol_feas": 1e-9, "max_iter": 200}

DEFAULT_SOLVER_OPTS = {
    "MOSEK": {"mosek_params": MOSEK_PARAMS},
    "CLARABEL": CLARABEL_PARAMS,
}


@dataclass(frozen=True)
class FilterResult:
    """Outcome of one safety-filter solve.

    The residuals are recomputed in NumPy from the returned decision variables,
    so they certify the solution independently of the solver's own report.
    """

    u: np.ndarray
    Omega: np.ndarray
    omega: Optional[float]
    objective: float
    status: str
    solve_time: float
    safety_residual: float
    recovery_residual: Optional[float]


class SafetyFilter:
    """Minimally invasive safety filter for a matrix CBF constraint.

    Parameters
    ----------
    n_controls, dim:
        Number of control inputs ``m`` and size ``p`` of the matrix constraint.
    decay:
        One of :data:`DECAY_MODES`.
    omega_d:
        Nominal decay rate. ``omega_d = 1`` reproduces the standard condition
        at ``Omega_d = (1/2) I`` and is the setting under which the matrix
        filter is no more invasive than the standard filter.
    rho_omega:
        Penalty on the *scalar* decay deviation. The matrix penalty is derived
        as ``rho_Omega = (4 / p) * rho_omega`` so that the two agree on the
        restriction ``Omega = (omega / 2) I``; with matched penalties the
        optimal values obey ``J_matrix <= J_scalar <= J_standard``.
    recovery:
        Impose the recovery constraint (``omega >= omega_d`` for scalar decay,
        ``sym(Omega) >= beta I`` for matrix decay), which drives negative
        eigenvalues of ``H`` back toward zero.
    beta:
        Recovery margin for the matrix filter. Defaults to ``omega_d / 2``,
        the largest value for which the invasiveness bound still holds.
    solver:
        CVXPY solver name. Defaults to MOSEK when installed, else CLARABEL.
    solver_opts:
        Keyword arguments for ``Problem.solve``. Defaults to the tight
        tolerances of :data:`DEFAULT_SOLVER_OPTS` for MOSEK and CLARABEL, and
        to the solver's own defaults otherwise.
    """

    def __init__(
        self,
        n_controls: int,
        dim: int,
        decay: str = "matrix",
        omega_d: float = 1.0,
        rho_omega: float = 5e-3,
        recovery: bool = True,
        beta: Optional[float] = None,
        solver: Optional[str] = None,
        solver_opts: Optional[dict] = None,
    ) -> None:
        if decay not in DECAY_MODES:
            raise ValueError(f"decay must be one of {DECAY_MODES}, got {decay!r}")
        if omega_d <= 0:
            raise ValueError("omega_d must be positive")
        if rho_omega <= 0:
            raise ValueError("rho_omega must be positive")

        self.m = int(n_controls)
        self.p = int(dim)
        self.decay = decay
        self.omega_d = float(omega_d)
        self.rho_omega = float(rho_omega)
        self.rho_Omega = 4.0 * self.rho_omega / self.p
        self.recovery = bool(recovery)
        self.beta = float(omega_d / 2 if beta is None else beta)

        if solver is None:
            solver = "MOSEK" if "MOSEK" in cp.installed_solvers() else "CLARABEL"
        self.solver = solver
        if solver_opts is None:
            solver_opts = dict(DEFAULT_SOLVER_OPTS.get(solver, {}))
        self.solver_opts = solver_opts

        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        p, m = self.p, self.m
        self._A = cp.Parameter((p, p), symmetric=True)       # alpha(H(x))
        self._LfH = cp.Parameter((p, p), symmetric=True)     # drift term
        self._LgH = [cp.Parameter((p, p), symmetric=True) for _ in range(m)]
        self._u_nom = cp.Parameter(m)

        u = cp.Variable(m)
        Hdot = self._LfH + sum(u[j] * self._LgH[j] for j in range(m))
        cost = cp.sum_squares(u - self._u_nom)
        constraints = []
        omega = None
        Omega = None

        if self.decay == "standard":
            decay_term = self._A
        elif self.decay == "scalar":
            omega = cp.Variable(nonneg=True)
            decay_term = omega * self._A
            cost = cost + self.rho_omega * cp.square(omega - self.omega_d)
            if self.recovery:
                constraints.append(omega >= self.omega_d)
        else:  # matrix
            Omega = cp.Variable((p, p))
            decay_term = Omega.T @ self._A + self._A @ Omega
            cost = cost + self.rho_Omega * cp.sum_squares(
                Omega - (self.omega_d / 2) * np.eye(p)
            )
            if self.recovery:
                constraints.append((Omega + Omega.T) / 2 >> self.beta * np.eye(p))

        X = Hdot + decay_term
        constraints.append((X + X.T) / 2 >> 0)

        self._u = u
        self._omega = omega
        self._Omega = Omega
        self._problem = cp.Problem(cp.Minimize(cost), constraints)

    # ------------------------------------------------------------------ solve
    def __call__(
        self,
        alpha_H: np.ndarray,
        LgH: Sequence[np.ndarray],
        u_nom: np.ndarray,
        LfH: Optional[np.ndarray] = None,
    ) -> FilterResult:
        """Filter ``u_nom`` against the matrix barrier condition at one state."""
        p, m = self.p, self.m
        alpha_H = np.asarray(alpha_H, dtype=float)
        LgH = [np.asarray(G, dtype=float) for G in LgH]
        u_nom = np.asarray(u_nom, dtype=float)
        LfH = np.zeros((p, p)) if LfH is None else np.asarray(LfH, dtype=float)
        if alpha_H.shape != (p, p) or LfH.shape != (p, p):
            raise ValueError(f"alpha_H and LfH must be {p}x{p}")
        if len(LgH) != m or any(G.shape != (p, p) for G in LgH):
            raise ValueError(f"LgH must hold {m} matrices of shape {p}x{p}")
        if u_nom.shape != (m,):
            raise ValueError(f"u_nom must have shape ({m},)")

        self._A.value = _sym(alpha_H)
        self._LfH.value = _sym(LfH)
        for param, G in zip(self._LgH, LgH):
            param.value = _sym(G)
        self._u_nom.value = u_nom

        self._problem.solve(solver=self.solver, **self.solver_opts)
        if self._problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise FilterInfeasible(f"{self.decay} filter: {self._problem.status}")

        u = np.asarray(self._u.value, dtype=float).reshape(m)
        if self.decay == "standard":
            omega = None
            Omega = 0.5 * np.eye(p)
        elif self.decay == "scalar":
            omega = float(self._omega.value)
            Omega = 0.5 * omega * np.eye(p)
        else:
            omega = None
            Omega = np.asarray(self._Omega.value, dtype=float)

        # Independent a-posteriori check of the returned pair (u, Omega).
        Hdot = LfH + sum(u[j] * LgH[j] for j in range(m))
        X = Hdot + Omega.T @ alpha_H + alpha_H @ Omega
        safety_residual = float(np.linalg.eigvalsh(_sym(X))[0])

        recovery_residual = None
        if self.recovery and self.decay != "standard":
            recovery_residual = float(
                np.linalg.eigvalsh(_sym(Omega))[0] - self.beta
                if self.decay == "matrix"
                else omega - self.omega_d
            )

        stats = self._problem.solver_stats
        solve_time = float(stats.solve_time) if stats.solve_time else float("nan")
        return FilterResult(
            u=u,
            Omega=Omega,
            omega=omega,
            objective=float(self._problem.value),
            status=self._problem.status,
            solve_time=solve_time,
            safety_residual=safety_residual,
            recovery_residual=recovery_residual,
        )


class FilterInfeasible(RuntimeError):
    """Raised when the solver returns a status other than optimal or optimal_inaccurate.

    A solver that fails outright, as CLARABEL can on the weakly infeasible problems
    at the boundary, raises ``cvxpy.error.SolverError`` instead, which the filter
    passes through unchanged.
    """


def min_scalar_decay(alpha_H: np.ndarray, Hdot: np.ndarray) -> float:
    """Smallest scalar decay certifying ``Hdot`` where ``alpha(H)`` is positive definite.

    This is the interior case of Proposition 2 of the paper: the condition
    ``Hdot + omega * alpha(H) >= 0`` holds exactly for

        omega >= max{0, lambda_max(A^{-1/2} (-Hdot) A^{-1/2})},  A = alpha(H).

    The bound diverges as the constraint approaches a boundary point whose
    kernel is coupled to the positive eigenspace, which is the obstruction the
    matrix multiplier removes.
    """
    lam, V = np.linalg.eigh(_sym(alpha_H))
    if lam.min() <= 0:
        raise ValueError("min_scalar_decay requires alpha(H) positive definite")
    A_inv_sqrt = V @ np.diag(lam ** -0.5) @ V.T
    congruent = A_inv_sqrt @ _sym(Hdot) @ A_inv_sqrt
    return float(max(0.0, -np.linalg.eigvalsh(congruent)[0]))


def _sym(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + X.T)
