"""The implementation checked against the paper's results, Sections III to VII.

Each test names the result it checks, numbered as in the arXiv version
(arXiv:2610.18675), with the LaTeX label of each equation in brackets:

=========================================  ===============================================================
Section IV                                 standard and scalar OD are restrictions of matrix OD
Eq. (7)                                    ``omega >= 0``, while the matrix multiplier is unsigned
Eq. (17) [eq:decayblocks]                  the kernel block of the decay term vanishes for every Omega
Theorem 1                                  matrix OD certifies a boundary control iff ``Z00 >= 0``
Remark 1, Corollary 1, eq. (18)            ``d/dt lambda_min(H) = v0.T Hdot v0`` at the boundary [eq:eigderiv]
Proposition 1, eq. (12), Tab. I            scalar OD certifies iff ``Z00 >= 0`` and ``Z01`` in range(``Z00``)
Example 1, eq. (10)                        ``Hdot = [[0, b], [b, c]]``: ``det = -b**2`` for every ``omega``
Proposition 2, eqs. (13), (14)             the smallest scalar decay [eq:scalarcost] and its blow-up
Eq. (19) [eq:perdirection], Theorem 2      per-direction decay and forward invariance
Theorem 3, eqs. (15), (21)                 well-posedness, strict feasibility [eq:odcond_strict], continuity
Theorem 4, eqs. (23), (24)                 OD filters are no more invasive than the standard filter
Eqs. (25), (26), Prop. 3                   recovery [eq:recovery] drives negative eigenvalues up; beta <= omega_d / 2
=========================================  ===============================================================

Section VIII, singularity avoidance, is in ``test_paper.py`` (the paper's experiments and
closed forms) and ``test_manuscript_example.py`` (rotation about joint 1 on other arms, and
the relative-angle example).

The filters are exercised on the two-link arm, whose zero eigenvalue on the
boundary is simple, and on synthetic instances with ``p = 3`` and kernels of
dimension one and two, built in a random orthonormal basis. The eigenbasis
blocks and the multipliers of the proofs come from ``eigenblocks.py``, which
follows the manuscript and not :mod:`odmcbf.filters`.

``certify`` asks whether one control is certified at one state: with
``LgH = 0`` the control drops out of the filter, which then only decides
whether a decay multiplier certifies ``Hdot = LfH``.
"""

from __future__ import annotations

import numpy as np
import pytest

from odmcbf import (
    DECAY_MODES,
    FilterInfeasible,
    SafetyFilter,
    TwoLinkArm,
    min_scalar_decay,
    rollout,
)

from .eigenblocks import (
    blocks,
    certify,
    lambda_min,
    large_enough_c,
    od_lmi,
    proposition1_holds,
    proposition2_blowup_bound,
    proposition2_bound,
    proposition3_certificate,
    scalar_never_certifies,
    sym,
    theorem1_certificate,
)

#: Feasibility slack allowed on a returned LMI or recovery constraint.
LMI_TOL = 1e-7
#: The default matrix penalty, ``rho_Omega = (4 / p) rho_omega`` with ``rho_omega = 5e-3``, ``p = 2``.
RHO_OMEGA_MATRIX = 4 * 5e-3 / 2


# ------------------------------------------------------------------ helpers
@pytest.fixture
def arm(manuscript_arm):
    """Here ``arm`` also varies ``gamma``: conftest's arms, and two of them with ``gamma = 2.5``."""
    return manuscript_arm


def _boundary_thetas(arm: TwoLinkArm, theta1: float = 0.3):
    lo, hi = arm.safe_theta_diff_interval
    for theta_diff in (lo, -lo, hi, -hi):
        yield np.array([theta1, theta1 + theta_diff])


def _increasing(arm: TwoLinkArm, theta) -> np.ndarray:
    """A unit command that raises the margin at a boundary configuration (well-posedness)."""
    sign = np.sign(np.sin(2 * (theta[1] - theta[0])))
    return sign * np.array([-1.0, 1.0]) / np.sqrt(2)


def _random_states(rng, n):
    for _ in range(n):
        yield rng.uniform(-np.pi, np.pi, size=2), 1.5 * rng.normal(size=2)


def _unsafe_thetas(arm: TwoLinkArm, theta1: float = -0.4):
    """Configurations halfway into each band where ``H`` is violated."""
    lo, hi = arm.safe_theta_diff_interval
    for theta_diff in (0.5 * lo, -0.5 * lo, 0.5 * (hi + np.pi), -0.5 * (hi + np.pi)):
        yield np.array([theta1, theta1 + theta_diff])


def _instance(rng, Z00, Z01, Z11, lam1, gamma=1.0):
    """``H >= 0`` with kernel dimension ``len(Z00)``, and ``Hdot`` with the given blocks, in a random basis."""
    Z00, Z11 = np.atleast_2d(Z00).astype(float), np.atleast_2d(Z11).astype(float)
    Z01 = np.asarray(Z01, dtype=float).reshape(Z00.shape[0], Z11.shape[0])
    k, r = Z00.shape[0], Z11.shape[0]
    Q = np.linalg.qr(rng.normal(size=(k + r, k + r)))[0]
    H = Q @ np.diag(np.r_[np.zeros(k), lam1]) @ Q.T
    Hdot = Q @ np.block([[Z00, Z01], [Z01.T, Z11]]) @ Q.T
    return sym(H), gamma * sym(H), sym(Hdot)


#: Synthetic boundary instances, ``p = 3``: (id, Z00, Z01, Z11, Prop. 1 holds, Z00 >= 0).
INSTANCES = [
    ("k2-Z00-pd",        np.diag([0.8, 0.3]), [[0.4], [-0.6]], [[-0.5]], True, True),
    ("k2-in-range",      np.diag([0.8, 0.0]), [[0.4], [0.0]],  [[-0.5]], True, True),
    ("k2-out-of-range",  np.diag([0.8, 0.0]), [[0.4], [0.3]],  [[-0.5]], False, True),
    ("k2-zero",          np.zeros((2, 2)),    [[0.0], [0.0]],  [[-0.5]], True, True),
    ("k2-graze",         np.zeros((2, 2)),    [[0.2], [-0.5]], [[0.4]],  False, True),
    ("k2-indefinite",    np.diag([0.8, -0.1]), [[0.4], [0.3]], [[-0.5]], False, False),
    ("k1-pd",            [[0.5]],  [[0.3, -0.1]], np.diag([-0.5, 0.2]), True, True),
    ("k1-graze",         [[0.0]],  [[0.3, -0.1]], np.diag([-0.5, 0.2]), False, True),
    ("k1-zero",          [[0.0]],  [[0.0, 0.0]],  np.diag([-0.5, 0.2]), True, True),
    ("k1-negative",      [[-0.2]], [[0.3, -0.1]], np.diag([-0.5, 0.2]), False, False),
]


def _synthetic(rng, case, gamma=1.0):
    _, Z00, Z01, Z11, _, _ = case
    k = np.atleast_2d(Z00).shape[0]
    lam1 = np.array([1.3, 0.7])[: 3 - k]
    return _instance(rng, Z00, Z01, Z11, lam1, gamma=gamma)


# ----------------------------------------------------- Sections III and IV
@pytest.mark.parametrize("recovery", [True, False])
def test_standard_and_scalar_certificates_are_matrix_certificates(arm, rng, recovery):
    """Section IV: ``Omega = I/2`` is the standard condition and ``Omega = (omega/2) I`` scalar OD.

    Hence every control either filter certifies is certified by eq. (16) at the
    same state, and the matrix filter's cost of certifying it is at most the
    penalty of that restricted multiplier: zero for the standard filter
    (``omega_d = 1``), ``rho_omega (omega - 1)**2`` for scalar OD.
    """
    for theta, u_nom in _random_states(rng, 6):
        alpha_H, LgH = arm.alpha_H(theta), arm.LgH(theta)
        for decay in ("standard", "scalar"):
            res = SafetyFilter(2, 2, decay=decay, recovery=recovery)(alpha_H, LgH, u_nom)
            restricted = RHO_OMEGA_MATRIX * float(np.sum((res.Omega - 0.5 * np.eye(2)) ** 2))
            cert = certify("matrix", alpha_H, arm.Hdot(theta, res.u), recovery=recovery)
            assert cert.safety_residual >= -LMI_TOL
            assert cert.objective <= restricted + 1e-8


@pytest.mark.parametrize("p", [2, 3])
def test_scalar_decay_is_nonnegative_and_the_matrix_multiplier_unsigned(rng, p):
    """Eq. (7) restricts scalar OD to ``omega >= 0``; the OD-SDP, eq. (21), leaves ``Omega`` unconstrained.

    With ``alpha(H) = -0.8 I`` and no recovery constraint, keeping the command
    ``k_nom``, which lowers ``H``, would need a negative decay. The scalar
    filter stops at ``omega = 0`` and changes the command instead; the matrix
    filter takes ``sym(Omega)`` negative definite and pays far less.
    """
    alpha_H = -0.8 * np.eye(p)
    LgH = [np.eye(p), sym(rng.normal(size=(p, p)))]
    u_nom = np.array([-3.0, 0.0])
    scalar = SafetyFilter(2, p, decay="scalar", recovery=False)(alpha_H, LgH, u_nom)
    assert scalar.omega >= 0
    assert scalar.omega == pytest.approx(0.0, abs=1e-6)
    assert scalar.safety_residual >= -LMI_TOL
    matrix = SafetyFilter(2, p, decay="matrix", recovery=False)(alpha_H, LgH, u_nom)
    assert np.linalg.eigvalsh(sym(matrix.Omega))[-1] < 0
    assert matrix.objective < 0.1 * scalar.objective


# ---------------------------------------------------------------- Section IV
def test_alpha_H_shares_eigenvectors_and_kernel_with_H(arm, rng):
    """Section II: ``alpha(H)`` shares the eigenvectors of ``H`` and, on the boundary, its kernel."""
    for theta in rng.uniform(-np.pi, np.pi, size=(10, 2)):
        H, A = arm.H(theta), arm.alpha_H(theta)
        np.testing.assert_allclose(A @ H, H @ A, rtol=0, atol=1e-13)
    for theta in _boundary_thetas(arm):
        v0 = np.linalg.eigh(arm.H(theta))[1][:, 0]
        assert np.linalg.norm(arm.alpha_H(theta) @ v0) <= 1e-14 * (arm.l1 + arm.l2) ** 2


def _boundary_cases(arm, rng):
    """Boundary configurations of the arm with random commands, and the synthetic instances."""
    for theta in _boundary_thetas(arm):
        H, A = arm.H(theta), arm.alpha_H(theta)
        for u in rng.normal(size=(3, 2)):
            yield H, A, arm.Hdot(theta, u)
    for case in INSTANCES:
        yield _synthetic(rng, case, gamma=arm.gamma)


def test_kernel_block_of_the_decay_term_vanishes(arm, rng):
    """Eq. (17): in the eigenbasis the decay term is ``[[0, Omega10.T A], [A Omega10, A Omega11 + Omega11.T A]]``.

    Its kernel block is zero for every ``Omega``, and ``Omega00``, ``Omega01``
    do not enter it at all.
    """
    for H, A, Hdot in _boundary_cases(arm, rng):
        b = blocks(H, A, Hdot)
        k = b.V0.shape[1]
        for _ in range(3):
            Omega = rng.normal(size=H.shape)
            D = b.in_basis(Omega.T @ A + A @ Omega)
            O = b.in_basis(Omega)
            scale = 1e-12 * (1 + np.abs(A).max()) * (1 + np.abs(Omega).max())
            np.testing.assert_allclose(D[:k, :k], 0.0, rtol=0, atol=scale)
            np.testing.assert_allclose(D[k:, :k], b.A @ O[k:, :k], rtol=0, atol=scale)
            np.testing.assert_allclose(D[k:, k:], b.A @ O[k:, k:] + O[k:, k:].T @ b.A,
                                       rtol=0, atol=scale)
            changed = O.copy()
            changed[:k, :] += rng.normal(size=changed[:k, :].shape)    # Omega00 and Omega01
            Omega2 = b.V @ changed @ b.V.T
            np.testing.assert_allclose(Omega2.T @ A + A @ Omega2, Omega.T @ A + A @ Omega,
                                       rtol=0, atol=10 * scale)


def test_theorem1_constructive_certificate(arm, rng):
    """Theorem 1, proof of sufficiency: ``Omega10 = -A^{-1} Z01.T``, ``Omega11 = c A^{-1}``.

    In the eigenbasis the LMI of eq. (16) becomes ``diag(Z00, Z11 + 2 c I)``,
    whatever ``Omega00`` and ``Omega01`` are, so it holds for large ``c``
    exactly when ``Z00 >= 0``.
    """
    for H, A, Hdot in _boundary_cases(arm, rng):
        b = blocks(H, A, Hdot)
        k, r = b.Z01.shape
        c = large_enough_c(b)
        for Omega00, Omega01 in ((None, None), (rng.normal(size=(k, k)), rng.normal(size=(k, r)))):
            Omega = theorem1_certificate(b, c, Omega00=Omega00, Omega01=Omega01)
            X = b.in_basis(od_lmi(Hdot, A, Omega))
            expected = np.block([[b.Z00, np.zeros_like(b.Z01)],
                                 [np.zeros_like(b.Z01.T), b.Z11 + 2 * c * np.eye(b.Z11.shape[0])]])
            np.testing.assert_allclose(X, expected, rtol=0, atol=1e-11)
            assert (lambda_min(X) >= -1e-11) == (lambda_min(b.Z00) >= -1e-11)


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("case", INSTANCES, ids=[c[0] for c in INSTANCES])
def test_theorem1_matrix_filter_certifies_iff_tangent_cone(rng, case, recovery):
    """Theorem 1 and Proposition 3(a), ``p = 3``: the matrix filter certifies iff ``Z00 >= 0``.

    With a two-dimensional kernel ``Z00`` is a matrix, and a singular ``Z00``
    with ``Z01`` outside its range is still certified.
    """
    _, A, Hdot = _synthetic(rng, case)
    if case[5]:
        res = certify("matrix", A, Hdot, recovery=recovery)
        assert res.safety_residual >= -LMI_TOL
        if recovery:
            assert lambda_min(res.Omega) >= 0.5 - LMI_TOL
    else:
        with pytest.raises(FilterInfeasible, match="matrix filter"):
            certify("matrix", A, Hdot, recovery=recovery)


@pytest.mark.parametrize("recovery", [True, False])
def test_zero_kernel_block_forces_the_coupling_block(arm, rng, recovery):
    """Where ``Z00 = 0`` every certificate has ``A Omega10 = -Z01.T``, the block of Theorem 1's proof.

    By eq. (17) the kernel block of the LMI is then zero, and a positive
    semidefinite matrix with a zero diagonal block has zero off-diagonal
    blocks too. So the matrix filter's multiplier must have exactly the
    coupling block of the proof. At the arm's boundary this is rotation about
    joint 1.
    """
    cases = [(arm.H(theta), arm.alpha_H(theta), arm.Hdot(theta, (w, w)))
             for theta in _boundary_thetas(arm) for w in (1.0, -0.6)]
    cases += [_synthetic(rng, case) for case in INSTANCES if case[0] in ("k2-graze", "k1-graze")]
    for H, A, Hdot in cases:
        b = blocks(H, A, Hdot)
        assert np.abs(b.Z00).max() <= 1e-12 * (1 + np.abs(Hdot).max())
        res = certify("matrix", A, Hdot, recovery=recovery)
        Omega10 = b.V1.T @ res.Omega @ b.V0
        np.testing.assert_allclose(b.A @ Omega10, -b.Z01.T, rtol=0,
                                   atol=1e-4 * (1 + np.abs(b.Z01).max()))


def test_remark1_margin_derivative_at_the_boundary(arm, rng):
    """Remark 1, eq. (18): where the zero eigenvalue is simple, ``d/dt lambda_min(H) = v0.T Hdot v0``."""
    h = 1e-4
    for theta in _boundary_thetas(arm):
        v0 = np.linalg.eigh(arm.H(theta))[1][:, 0]
        for u in rng.normal(size=(4, 2)):
            margin = [arm.margin(theta + s * h * u) for s in (-2, -1, 1, 2)]
            fd = (margin[0] - 8 * margin[1] + 8 * margin[2] - margin[3]) / (12 * h)
            assert fd == pytest.approx(v0 @ arm.Hdot(theta, u) @ v0, abs=1e-9)


def test_corollary1_lambda_min_condition_iff_matrix_od(arm):
    """Corollary 1: at the boundary, ``d/dt lambda_min(H) >= 0`` iff eq. (16) is satisfiable.

    The commands include rotation about joint 1, along which ``lambda_min``
    stays zero; both conditions hold for it.
    """
    commands = [(1.0, 1.0), (-0.5, -0.5), (0.0, 1.0), (1.0, 0.0), (0.3, -0.4), (-2.0, 0.5)]
    h = 1e-4
    for theta in _boundary_thetas(arm):
        for u in map(np.array, commands):
            margin = [arm.margin(theta + s * h * u) for s in (-2, -1, 1, 2)]
            rate = (margin[0] - 8 * margin[1] + 8 * margin[2] - margin[3]) / (12 * h)
            if u[0] == u[1]:
                assert abs(rate) <= 1e-9
            else:
                assert abs(rate) > 1e-3
            if rate >= -1e-9:
                assert certify("matrix", arm.alpha_H(theta), arm.Hdot(theta, u)).safety_residual >= -LMI_TOL
            else:
                with pytest.raises(FilterInfeasible):
                    certify("matrix", arm.alpha_H(theta), arm.Hdot(theta, u))


# --------------------------------------------------------------- Section III
@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("case", INSTANCES, ids=[c[0] for c in INSTANCES])
def test_proposition1_scalar_certificate_conditions(rng, case, recovery):
    """Proposition 1, eq. (12), and Table I: scalar OD certifies iff (i) ``Z00 >= 0`` and
    (ii) ``(I - Z00 Z00^+) Z01 = 0``.

    The conditions are evaluated with the pseudoinverse from the blocks of the
    instance; the scalar filter must certify exactly when they hold. The matrix
    filter certifies whenever (i) holds, so the two differ exactly on the
    instances where (i) holds and (ii) fails.
    """
    H, A, Hdot = _synthetic(rng, case)
    holds = proposition1_holds(blocks(H, A, Hdot))
    assert holds == case[4]
    if holds:
        res = certify("scalar", A, Hdot, recovery=recovery)
        assert res.safety_residual >= -LMI_TOL
    else:
        scalar_never_certifies(A, Hdot, recovery=recovery)


P2_EXAMPLES = [(1.0, 0.5, 0.0, 1.0), (0.7, -1.0, 2.0, 1.0), (2.0, 1e-3, -1.0, 1.0),
               (1.0, 1.0, -5.0, 2.5), (0.3, 0.2, 0.1, 0.5)]


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("lam, b, c, gamma", P2_EXAMPLES)
def test_example1_p2_counterexample(rng, lam, b, c, gamma, recovery):
    """Example 1 of Section III, eq. (10): ``H = diag(0, lam)``, ``Hdot = [[0, b], [b, c]]``, ``b != 0``.

    The tangent-cone condition holds (``Z00 = 0``), the block ``Omega10 =
    -b / alpha(lam)`` certifies the control, and the matrix filter's multiplier
    has exactly that block. No scalar decay certifies it: at any ``omega`` the
    scalar LMI has determinant ``-b**2``, so whatever ``omega`` the scalar
    filter returns leaves a negative residual. The instance is posed in a
    random orthonormal basis.
    """
    H, A, Hdot = _instance(rng, [[0.0]], [[b]], [[c]], [lam], gamma=gamma)
    Q = np.linalg.eigh(H)[1]                            # the instance's basis, up to signs
    Z = Q.T @ Hdot @ Q
    np.testing.assert_allclose([Z[0, 0], abs(Z[0, 1]), Z[1, 1]], [0.0, abs(b), c], rtol=0, atol=1e-14)
    res = certify("matrix", A, Hdot, recovery=recovery)
    assert res.safety_residual >= -LMI_TOL
    Omega10 = (Q.T @ res.Omega @ Q)[1, 0]
    assert Omega10 == pytest.approx(-Z[0, 1] / (gamma * lam), rel=1e-4, abs=1e-5)
    scalar_never_certifies(A, Hdot, recovery=recovery)


def _scalar_decay(A, Hdot, omega_d=1.0, recovery=False) -> float:
    return certify("scalar", A, Hdot, recovery=recovery, omega_d=omega_d).omega


@pytest.mark.parametrize("recovery", [True, False])
def test_proposition2_at_the_boundary(arm, rng, recovery):
    """Proposition 2, eq. (13), on the boundary: with ``Z00 > 0`` the scalar filter picks
    ``omega = max(omega_d, omega*)``, ``omega*`` from the Schur complement.

    The commands raise the margin (``Z00 > 0``) while rotating the whole arm
    about joint 1, which is what makes ``omega*`` exceed ``omega_d``. The
    synthetic instances have one- and two-dimensional kernels.
    """
    cases = []
    for theta in _boundary_thetas(arm):
        for w, eps in ((1.0, 0.3), (-0.8, 0.1), (0.1, 1.0)):
            cases.append((arm.H(theta), arm.alpha_H(theta),
                          arm.Hdot(theta, w * np.ones(2) + eps * _increasing(arm, theta))))
    cases += [_synthetic(rng, case) for case in INSTANCES if case[0] in ("k2-Z00-pd", "k1-pd")]
    regimes = set()
    for H, A, Hdot in cases:
        b = blocks(H, A, Hdot)
        assert lambda_min(b.Z00) > 0
        expected = max(1.0, proposition2_bound(b))
        regimes.add(expected > 1.0)
        assert _scalar_decay(A, Hdot, recovery=recovery) == pytest.approx(
            expected, rel=1e-6, abs=1e-3)
    assert regimes == {True, False}


def test_proposition2_in_the_interior_is_min_scalar_decay(arm, rng):
    """Proposition 2 with ``H > 0``: the kernel blocks are empty and eq. (13) is ``min_scalar_decay``."""
    lo, hi = arm.safe_theta_diff_interval
    for _ in range(20):
        theta1 = rng.uniform(-np.pi, np.pi)
        theta = np.array([theta1, theta1 + rng.choice([-1, 1]) * rng.uniform(lo + 1e-3, hi - 1e-3)])
        Hdot = arm.Hdot(theta, 2 * rng.normal(size=2))
        b = blocks(arm.H(theta), arm.alpha_H(theta), Hdot)
        assert b.Z00.shape == (0, 0)
        assert min_scalar_decay(arm.alpha_H(theta), Hdot) == pytest.approx(
            proposition2_bound(b), rel=1e-10, abs=1e-14)


def test_proposition2_blowup(arm):
    """Eq. (14): as ``lambda_min(Z00) -> 0`` with ``Z01`` bounded away from zero, ``omega*`` diverges.

    The command is rotation about joint 1 at unit rate plus ``eps`` times a
    margin-raising command. The scalar filter's decay obeys the lower bound of
    eq. (14), and ``omega Z00`` tends to ``Z01**2 / alpha(lambda_H)`` for the
    rotation, ``|Z01| = lambda_H`` (Section VIII-A, the text before eq. (32)), that is
    to ``lambda_H / gamma`` with ``lambda_H = tr H``. The matrix filter
    certifies the same commands at a bounded cost, at most the penalty
    ``2 rho_Omega / gamma**2`` of the rotation certificate of eq. (33).

    ``eps`` stops at ``1e-3``, where ``omega*`` lies between about ``800`` and
    ``1e4`` on these arms; around ``4e4`` CLARABEL starts to fail on the scalar
    problem.
    """
    for theta in _boundary_thetas(arm):
        A, H = arm.alpha_H(theta), arm.H(theta)
        limit = np.trace(H) / arm.gamma
        errors = []
        for eps in (1e-1, 1e-2, 1e-3):
            u = np.ones(2) + eps * _increasing(arm, theta)
            Hdot = arm.Hdot(theta, u)
            b = blocks(H, A, Hdot)
            omega = _scalar_decay(A, Hdot)
            assert omega >= proposition2_blowup_bound(b) * (1 - 1e-5)      # tight for p = 2
            assert omega == pytest.approx(proposition2_bound(b), rel=1e-5)
            errors.append(abs(omega * b.Z00[0, 0] / limit - 1))
            cert = certify("matrix", A, Hdot)
            assert cert.objective <= 2 * RHO_OMEGA_MATRIX / arm.gamma ** 2 + 1e-9
        assert np.all(np.diff(errors) < 1e-6)
        assert errors[-1] < 5e-3


# ----------------------------------------------------------------- Section V
@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_per_direction_decay(arm, rng, decay, recovery):
    """Eq. (19): for each unit eigenvector ``v`` of ``H``, ``v.T Hdot v >= -2 alpha(lambda) v.T sym(Omega) v``.

    Only the symmetric part of ``Omega`` enters: ``v.T (Omega.T A + A Omega) v
    = 2 alpha(lambda) v.T sym(Omega) v`` exactly. States cover both sides of
    the boundary.
    """
    filt = SafetyFilter(2, 2, decay=decay, recovery=recovery)
    signs = set()
    states = [*_random_states(rng, 6), *((theta, 1.5 * rng.normal(size=2)) for theta in _unsafe_thetas(arm))]
    for theta, u_nom in states:
        A = arm.alpha_H(theta)
        res = filt(A, arm.LgH(theta), u_nom)
        Hdot = arm.Hdot(theta, res.u)
        lam, V = np.linalg.eigh(arm.H(theta))
        for lam_i, v in zip(lam, V.T):
            alpha = arm.gamma * lam_i
            signs.add(np.sign(alpha))
            decay_term = v @ (res.Omega.T @ A + A @ res.Omega) @ v
            assert decay_term == pytest.approx(2 * alpha * (v @ sym(res.Omega) @ v), abs=1e-12)
            assert v @ Hdot @ v >= -2 * alpha * (v @ sym(res.Omega) @ v) - LMI_TOL
    assert signs == {-1.0, 1.0}


def _safe_start(arm: TwoLinkArm, theta1: float = 0.3) -> np.ndarray:
    lo, hi = arm.safe_theta_diff_interval
    return np.array([theta1, theta1 + 0.5 * (lo + hi)])


@pytest.mark.parametrize("decay", DECAY_MODES)
def test_theorem2_forward_invariance(arm, decay):
    """Theorem 2: from inside the safe set, every filtered closed loop stays inside it.

    The commands close ``theta2 - theta1`` toward the lower boundary, open it
    toward the upper one, and spin the arm fast; unfiltered, each leaves the
    safe set within the horizon.

    Theorem 2 is about continuous time, while the rollout holds each control
    for a step ``dt``. The step is ``0.03 / gamma``, the same decay per step
    for every arm: with ``dt = 0.03`` and ``gamma = 2.5`` the sampled loop
    leaves the safe set by up to ``5e-3``.
    """
    theta0 = _safe_start(arm)
    dt = 0.03 / arm.gamma
    for u_nom in ((1.0, 0.4), (1.0, 1.6), (-2.0, 1.0)):
        nominal = rollout(arm, u_nom, theta0, dt=dt, horizon=3.0)
        assert nominal.margin.min() < 0
        trace = rollout(arm, u_nom, theta0, dt=dt, horizon=3.0, decay=decay)
        assert trace.margin.min() >= -1e-9
        assert np.nanmin(trace.safety_residual) >= -LMI_TOL


def test_theorem3_the_safe_set_is_well_posed(arm):
    """Definition 6 and Theorem 3: at every boundary configuration some control has ``Z00 > 0``,
    and with Theorem 1's multiplier the strict condition, eq. (15), then holds.
    """
    for theta in _boundary_thetas(arm):
        H, A = arm.H(theta), arm.alpha_H(theta)
        Hdot = arm.Hdot(theta, _increasing(arm, theta))
        b = blocks(H, A, Hdot)
        assert lambda_min(b.Z00) > 0.1 * (arm.l1 * arm.l2)
        Omega = theorem1_certificate(b, large_enough_c(b))
        assert lambda_min(od_lmi(Hdot, A, Omega)) >= 0.5 * lambda_min(b.Z00)


@pytest.mark.parametrize("recovery", [True, False])
def test_theorem3_minimiser_is_continuous_near_the_boundary(arm, rng, recovery):
    """Theorem 3: the OD-SDP, eq. (21), is feasible on a neighbourhood of the safe set and its
    minimiser is continuous there.

    Configurations on the boundary are perturbed into and out of the safe
    set; the filtered command moves by at most a small multiple of the
    perturbation. Theorem 3 is stated without the recovery constraint; the
    filter with it (Proposition 3) is checked the same way.
    """
    filt = SafetyFilter(2, 2, decay="matrix", recovery=recovery)
    for theta in _boundary_thetas(arm):
        for u_nom in (np.array([1.0, 0.4]), 1.5 * rng.normal(size=2)):
            u0 = filt(arm.alpha_H(theta), arm.LgH(theta), u_nom).u
            for delta in (1e-2, -1e-2, 1e-3, -1e-3, 1e-4, -1e-4):
                moved = theta + np.array([0.0, delta])
                u = filt(arm.alpha_H(moved), arm.LgH(moved), u_nom).u
                assert np.linalg.norm(u - u0) <= 50 * abs(delta) + 1e-5


# ---------------------------------------------------------------- Section VI
@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("rho_omega", [1e-4, 0.3, 10.0])
def test_theorem4_od_filters_are_no_more_invasive(arm, rng, rho_omega, recovery):
    """Theorem 4, eq. (23), and eq. (24), for penalties from ``1e-4`` to ``10``, with ``omega_d = 1``.

    ``||u_OD - k_nom||**2 <= J_OD <= ||u_std - k_nom||**2`` for both OD filters,
    and ``J_matrix <= J_scalar``. With recovery, ``beta = omega_d / 2``
    (Proposition 3(b)).
    """
    filters = {decay: SafetyFilter(2, 2, decay=decay, rho_omega=rho_omega, recovery=recovery)
               for decay in DECAY_MODES}
    active = 0
    for theta, u_nom in _random_states(rng, 6):
        res = {decay: filt(arm.alpha_H(theta), arm.LgH(theta), u_nom)
               for decay, filt in filters.items()}
        J_std = float(np.sum((res["standard"].u - u_nom) ** 2))
        active += J_std > 1e-4
        for decay in ("scalar", "matrix"):
            J = res[decay].objective
            tol = 1e-9 + 1e-6 * J_std
            assert float(np.sum((res[decay].u - u_nom) ** 2)) <= J + tol
            assert J <= J_std + tol
        assert res["matrix"].objective <= res["scalar"].objective + 1e-9 + 1e-6 * J_std
    assert active >= 2


@pytest.mark.parametrize("beta", [0.25, 0.5, 0.75, 1.0])
def test_proposition3b_needs_beta_at_most_half_omega_d(arm, rng, beta):
    """Proposition 3(b): with ``sym(Omega) >= beta I`` and ``beta <= omega_d / 2``, Theorem 4 holds.

    Its proof needs the nominal multiplier ``Omega_d = I/2`` to stay feasible,
    and for ``beta > omega_d / 2`` the conclusion fails at this state: ``k_nom``
    already satisfies the standard LMI strictly, so ``J_std = 0``, while the
    matrix filter pays ``p rho_Omega (beta - omega_d / 2)**2``, the distance
    from ``Omega_d`` to ``beta I``.
    """
    lo, hi = arm.safe_theta_diff_interval
    theta = np.array([0.4, 0.4 + 0.5 * (lo + hi)])
    A, LgH = arm.alpha_H(theta), arm.LgH(theta)
    u_nom = np.array([0.02, -0.01])
    assert lambda_min(arm.Hdot(theta, u_nom) + A) > 1e-3
    filt = SafetyFilter(2, 2, decay="matrix", beta=beta)
    res = filt(A, LgH, u_nom)
    expected = 2 * filt.rho_Omega * max(0.0, beta - 0.5) ** 2
    assert res.objective == pytest.approx(expected, rel=1e-5, abs=1e-8)
    np.testing.assert_allclose(res.u, u_nom, rtol=0, atol=1e-6)
    assert lambda_min(res.Omega) >= beta - LMI_TOL


# --------------------------------------------------------------- Section VII
@pytest.mark.parametrize("beta", [0.5, 0.25])
@pytest.mark.parametrize("spectrum", [(-1.0, -1.0, 0.5), (-1.0, -0.3, 0.5), (-0.4, -0.4, -0.4)],
                         ids=["repeated", "distinct", "all-negative"])
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_recovery_raises_every_negative_eigenvalue(rng, decay, spectrum, beta):
    """Eq. (26) and the repeated-eigenvalue case after it.

    For a negative eigenvalue ``lambda`` with eigenspace basis ``Vl``, the
    recovery constraint gives ``Vl.T Hdot Vl >= -2 beta alpha(lambda) I > 0``,
    so every negative eigenvalue rises at a rate proportional to its size.
    Here ``omega_d = 2 beta``: the scalar filter's ``omega >= omega_d`` gives
    the same bound, and the standard filter's ``Omega = I/2`` gives it with
    ``beta = 1/2``. ``p = 3``, ``gamma = 2``, and one control direction is the
    identity, so the filter is feasible.
    """
    gamma = 2.0
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    lam = np.array(spectrum)
    alpha_H = gamma * Q @ np.diag(lam) @ Q.T
    LgH = [np.eye(3), sym(rng.normal(size=(3, 3))), sym(rng.normal(size=(3, 3)))]
    filt = SafetyFilter(3, 3, decay=decay, omega_d=2 * beta, beta=beta)
    rate = 0.5 if decay == "standard" else beta
    for _ in range(3):
        u_nom = rng.normal(size=3) - np.array([3.0, 0.0, 0.0])      # pushes H down
        res = filt(alpha_H, LgH, u_nom)
        Hdot = sum(u_j * G for u_j, G in zip(res.u, LgH))
        for value in np.unique(lam[lam < 0]):
            Vl = Q[:, lam == value]
            bound = -2 * rate * gamma * value
            assert bound > 0
            assert lambda_min(Vl.T @ Hdot @ Vl) >= bound - LMI_TOL


@pytest.mark.parametrize("beta", [0.5, 0.2])
def test_proposition3a_constructive_certificate(arm, rng, beta):
    """Proposition 3(a), proof: ``Omega01 = -Omega10.T`` and ``Omega00 = mu I`` make
    ``sym(Omega) = diag(mu I, c A^{-1})``, so both the LMI and ``sym(Omega) >= beta I`` hold.
    """
    for H, A, Hdot in _boundary_cases(arm, rng):
        b = blocks(H, A, Hdot)
        if lambda_min(b.Z00) < -1e-12:
            continue
        c = max(beta * np.linalg.eigvalsh(b.A)[-1], large_enough_c(b))
        Omega = proposition3_certificate(b, c, mu=beta)
        k = b.V0.shape[1]
        S = b.in_basis(sym(Omega))
        np.testing.assert_allclose(S[:k, k:], 0.0, rtol=0, atol=1e-12 * (1 + np.abs(Omega).max()))
        np.testing.assert_allclose(S[:k, :k], beta * np.eye(k), rtol=0, atol=1e-12)
        np.testing.assert_allclose(S[k:, k:], c * np.linalg.inv(b.A), rtol=1e-12, atol=1e-12)
        assert lambda_min(sym(Omega)) >= beta - 1e-12
        assert lambda_min(od_lmi(Hdot, A, Omega)) >= -1e-11


@pytest.mark.parametrize("band", ["outer", "inner"])
@pytest.mark.parametrize("omega_d", [1.0, 2.0])
@pytest.mark.parametrize("decay", DECAY_MODES)
def test_recovery_in_closed_loop(arm, decay, omega_d, band):
    """Section VII in closed loop: started outside the safe set, the margin rises monotonically
    while negative, at least as fast as ``|lambda_min(t)| <= |lambda_min(0)| exp(-r t)``.

    The envelope follows from eq. (26), ``d lambda_min / dt >= r |lambda_min|``,
    by the comparison lemma, with ``r = gamma`` for the standard filter and
    ``gamma omega_d`` for scalar OD (``omega >= omega_d``) and matrix OD
    (``2 beta = omega_d``). The start lies halfway into the band next to the
    stretched (outer) or folded (inner) configurations; the command drives
    further in. The sampled loop keeps to the envelope with about ``5e-5``
    to spare per step.
    """
    lo, hi = arm.safe_theta_diff_interval
    theta0 = np.array([0.2, 0.2 + (0.5 * lo if band == "outer" else 0.5 * (hi + np.pi))])
    u_nom = np.array([1.0, 0.4]) if band == "outer" else np.array([1.0, 1.6])
    trace = rollout(arm, u_nom, theta0, dt=0.01, horizon=1.0, decay=decay, omega_d=omega_d)
    margin = trace.margin
    assert margin[0] < 0
    rate = arm.gamma * (1.0 if decay == "standard" else omega_d)
    negative = margin[:-1] < 0
    assert np.all(np.diff(margin)[negative] > 0)
    envelope = margin[0] * np.exp(-rate * trace.t)
    assert np.all(np.where(margin < 0, margin / envelope, 0.0) <= 1.0 + 1e-9)
