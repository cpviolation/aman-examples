r"""Lineshapes of the BaBar 2008 / Belle 2018 :math:`D^0 \to K^0_S \pi^+ \pi^-` model.

The expressions reproduce the conventions of the Belle C++ implementation (and of
:mod:`amplitf.dynamics` in `cpviolation/AmpliTF
<https://github.com/cpviolation/AmpliTF>`_, used by `cpviolation/tfa_examples
<https://github.com/cpviolation/tfa_examples>`_):

- Breit-Wigner lineshapes with Blatt-Weisskopf form factors for the resonance and the
  :math:`D^0` vertex, normalised at the nominal mass, and without orbital barrier
  factors :math:`(q/q_0)^L`;
- the LASS parametrisation of the :math:`K\pi` S-wave in the form of BaBar 2008;
- the five-channel :math:`\pi\pi` S-wave K-matrix (Anisovich-Sarantsev) with the
  P-vector production approximation.

They are defined as :func:`ampform.sympy.unevaluated` expressions, so they render as
LaTeX and unfold to their definition with :meth:`~sympy.core.basic.Basic.doit`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import sympy as sp
from ampform.dynamics import BreakupMomentumSquared
from ampform.sympy import NumPyPrintable, unevaluated
from ampform.sympy.math import ComplexSqrt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sympy.printing.latex import LatexPrinter
    from sympy.printing.numpy import NumPyPrinter


@unevaluated(real=True)
class BreakupMomentum(sp.Expr):
    r"""Two-body break-up momentum :math:`q(m; m_a, m_b)`, real square root."""

    m: Any
    ma: Any
    mb: Any
    _latex_repr_ = R"q\left({m}; {ma}, {mb}\right)"

    def evaluate(self) -> sp.Expr:
        m, ma, mb = self.args
        return sp.sqrt(BreakupMomentumSquared(m**2, ma, mb))


@unevaluated(real=True)
class BlattWeisskopfRatio(sp.Expr):
    r"""Blatt-Weisskopf form factor normalised at :math:`q_0`.

    .. math::

        F_L(q, q_0) = \sqrt{\frac{h_L(q_0 d)}{h_L(q d)}},\quad
        h_0 = 1,\ h_1(z) = 1 + z^2,\ h_2(z) = 9 + 3z^2 + z^4.
    """

    q: Any
    q0: Any
    d: Any
    L: int = 0
    _latex_repr_ = R"F_{{{L}}}\left({q}, {q0}\right)"

    def evaluate(self) -> sp.Expr:
        q, q0, d, L = self.args

        def h(z: sp.Expr) -> sp.Expr:
            z2 = z**2
            return {
                0: sp.Integer(1),
                1: 1 + z2,
                2: 9 + z2 * (3 + z2),
                3: 225 + z2 * (45 + z2 * (6 + z2)),
                4: 11025 + z2 * (1575 + z2 * (135 + z2 * (10 + z2))),
            }[int(L)]

        return sp.sqrt(h(q0 * d) / h(q * d))


@unevaluated(real=False)
class BelleBreitWigner(sp.Expr):
    r"""Breit-Wigner lineshape of a resonance :math:`R \to a b` in :math:`D \to R c`.

    .. math::

        \frac{F_L^R(p, p_0)\, F_L^D(q, q_0)}{m_0^2 - s - i m_0 \Gamma(s)},\quad
        \Gamma(s) = \Gamma_0 \left(\frac{p}{p_0}\right)^{2L+1} \frac{m_0}{\sqrt{s}}
        \left[F_L^R(p, p_0)\right]^2

    with :math:`p` the momentum of :math:`a` in the :math:`R` frame and :math:`q` that
    of :math:`c` in the :math:`D` frame; :math:`p_0, q_0` at :math:`\sqrt{s} = m_0`. The
    radii are :math:`d_R` and :math:`d_D`. ``mD0`` replaces :math:`m_D` in :math:`q_0`
    (e.g. for resonances with :math:`m_0 + m_c > m_D`).
    """

    s: Any
    m0: Any
    Gamma0: Any
    ma: Any
    mb: Any
    mc: Any
    mD: Any
    mD0: Any
    dR: Any
    dD: Any
    L: int = 1
    _latex_repr_ = R"\mathcal{{R}}_{{{L}}}\left({s}; {m0}, {Gamma0}\right)"

    def evaluate(self) -> sp.Expr:  # ruff: ignore[too-many-locals]
        s, m0, gamma0, ma, mb, mc, mD, mD0, dR, dD, L = self.args
        m = sp.sqrt(s)
        p = BreakupMomentum(m, ma, mb)
        p0 = BreakupMomentum(m0, ma, mb)
        q = BreakupMomentum(mD, m, mc)
        q0 = BreakupMomentum(mD0, m0, mc)
        ff_r = BlattWeisskopfRatio(p, p0, dR, L)
        ff_d = BlattWeisskopfRatio(q, q0, dD, L)
        width = gamma0 * (p / p0) ** (2 * L + 1) * (m0 / m) * ff_r**2
        return ff_r * ff_d / (m0**2 - s - sp.I * m0 * width)


@unevaluated(real=False)
class LASS(sp.Expr):
    r"""LASS :math:`K\pi` S-wave in the BaBar 2008 form.

    .. math::

        R e^{i(\phi_R + 2\phi_F)} \frac{q\cot\delta_B + iq}{q\cot\delta_B - iq}
        \frac{m_0^2 \Gamma_0 / q_0}{m_0^2 - s - i m_0 \Gamma(s)}
        + F e^{i\phi_F} (\cos\phi_F + \sin\phi_F \cot\delta_B)
        \frac{\sqrt{s}}{q\cot\delta_B - iq},

    with :math:`q\cot\delta_B = 1/a + r q^2/2` and :math:`\Gamma(s) = \Gamma_0
    \frac{q}{q_0}\frac{m_0}{\sqrt{s}}`.
    """

    s: Any
    a: Any
    r: Any
    m0: Any
    Gamma0: Any
    ma: Any
    mb: Any
    R: Any
    phiR: Any
    F: Any
    phiF: Any
    _latex_repr_ = R"\mathrm{{LASS}}\left({s}\right)"

    def evaluate(self) -> sp.Expr:  # ruff: ignore[too-many-locals]
        s, a, r, m0, gamma0, ma, mb, R, phi_r, F, phi_f = self.args
        m = sp.sqrt(s)
        q = BreakupMomentum(m, ma, mb)
        q0 = BreakupMomentum(m0, ma, mb)
        cot = 1 / (a * q) + r * q / 2
        width = gamma0 * q / m * m0 / q0
        resonant = m0**2 * gamma0 / q0 / (m0**2 - s - sp.I * m0 * width)
        nonresonant = m / (q * cot - sp.I * q)
        return (
            R
            * sp.exp(sp.I * (phi_r + 2 * phi_f))
            * (q * cot + sp.I * q)
            / (q * cot - sp.I * q)
            * resonant
            + F
            * sp.exp(sp.I * phi_f)
            * (sp.cos(phi_f) + sp.sin(phi_f) * cot)
            * nonresonant
        )


@unevaluated(real=False)
class TwoBodyPhaseSpaceFactor(sp.Expr):
    r"""K-matrix phase-space factor :math:`\rho(s) = \sqrt{1 - (m_a + m_b)^2/s}` (imaginary below threshold)."""

    s: Any
    ma: Any
    mb: Any
    _latex_repr_ = R"\rho\left({s}; {ma}, {mb}\right)"

    def evaluate(self) -> sp.Expr:
        s, ma, mb = self.args
        return ComplexSqrt(1 - (ma + mb) ** 2 / s)


@unevaluated(real=False)
class MultiMesonPhaseSpaceFactor(sp.Expr):
    r""":math:`4\pi` phase-space factor of the Belle K-matrix.

    Above :math:`s = 1\,\mathrm{GeV}^2` it is :math:`\sqrt{1 - 16 m_\pi^2/s}`; below,
    the polynomial approximation of the Belle/BaBar code, :math:`\sqrt{1 - 16 m_\pi^2}
    \, (1.2274 + 0.00370909/s^2 - 0.111203/s - 6.39017 s + 16.8358 s^2 - 21.8845 s^3 +
    11.3153 s^4)`.
    """

    s: Any
    m: Any
    _latex_repr_ = R"\rho_{{4\pi}}\left({s}\right)"

    def evaluate(self) -> sp.Expr:
        s, m = self.args
        below = sp.sqrt(1 - 16 * m**2) * (
            sp.Float(1.2274)
            + sp.Float(0.00370909) / s**2
            - sp.Float(0.111203) / s
            - sp.Float(6.39017) * s
            + sp.Float(16.8358) * s**2
            - sp.Float(21.8845) * s**3
            + sp.Float(11.3153) * s**4
        )
        above = ComplexSqrt(1 - 16 * m**2 / s)
        return sp.Piecewise((above, s > 1), (below, True))


class FirstRowInverseProduct(NumPyPrintable):
    r""":math:`\sum_j (M^{-1})_{0j} v_j` for an :math:`n \times n` matrix :math:`M`.

    A symbolic inverse of the 5x5 K-matrix propagator would be far too large, so the
    inverse is left to the numerical backend (``linalg.inv`` of NumPy or JAX). The
    arguments are :math:`n`, then the :math:`n^2` entries of :math:`M` (row-major) and
    the :math:`n` entries of :math:`v`; they may be scalars or arrays of events.
    """

    is_commutative = True

    def __new__(cls, n: int, *entries: sp.Expr, **kwargs: Any) -> Self:
        n = int(n)
        if len(entries) != n * n + n:
            msg = f"Expected {n * n} matrix and {n} vector entries, got {len(entries)}"
            raise ValueError(msg)
        return sp.Expr.__new__(cls, sp.Integer(n), *map(sp.sympify, entries), **kwargs)

    @property
    def n(self) -> int:
        return int(self.args[0])

    @property
    def matrix(self) -> sp.Matrix:
        n = self.n
        return sp.Matrix(n, n, self.args[1 : 1 + n * n])

    @property
    def vector(self) -> sp.Matrix:
        return sp.Matrix(self.args[1 + self.n**2 :])

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:  # ruff: ignore[unused-method-argument]
        n = printer._print(self.n)  # ruff: ignore[private-member-access]
        return Rf"\left[\mathbf{{M}}^{{-1}} \mathbf{{v}}\right]_0^{{({n})}}"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:  # ruff: ignore[unused-method-argument]
        names = {"stack", "broadcast_arrays", "moveaxis", "reshape", "sum", "linalg"}
        printer.module_imports[printer._module].update(names)  # ruff:ignore[private-member-access]
        n = self.n

        def batch(entries: Sequence[sp.Expr], shape: str) -> str:
            values = ", ".join(printer._print(e) for e in entries)  # ruff: ignore[private-member-access]
            stacked = f"stack(broadcast_arrays({values}))"
            return f"moveaxis(reshape({stacked}, ({shape}, -1)), -1, 0)"

        matrix = batch(self.args[1 : 1 + n * n], f"{n}, {n}")
        vector = batch(self.args[1 + n * n :], f"{n}")
        return f"sum(linalg.inv({matrix})[:, 0, :] * {vector}, axis=-1)"


def kmatrix_amplitude(  # ruff: ignore[too-many-positional-arguments]
    s: sp.Expr,
    pole_masses: Sequence[sp.Expr],
    pole_couplings: Sequence[Sequence[sp.Expr]],
    f_scatt: Sequence[Sequence[sp.Expr]],
    s0_scatt: sp.Expr,
    betas: Sequence[sp.Expr],
    f_prod: Sequence[sp.Expr],
    s0_prod: sp.Expr,
    channel_masses: Sequence[Sequence[sp.Expr]],
    sA: sp.Expr | float = 1,
    sA0: sp.Expr | float = -0.15,
    m_threshold: sp.Expr | float = 0.13957,
) -> FirstRowInverseProduct:
    r""":math:`\pi\pi` S-wave K-matrix amplitude with the P-vector approximation.

    .. math::

        A(s) = \sum_j \left[(I - i K \rho)^{-1}\right]_{0j} P_j(s),

    .. math::

        K_{ij} = \left(\sum_\alpha \frac{g_i^\alpha g_j^\alpha}{m_\alpha^2 - s}
        + f_{ij}^\mathrm{scatt} \frac{1 - s_0^\mathrm{scatt}}{s - s_0^\mathrm{scatt}}\right)
        \frac{1 - s_{A0}}{s - s_{A0}} \left(s - \frac{s_A m_\mathrm{th}^2}{2}\right),

    .. math::

        P_j = \sum_\alpha \frac{\beta_\alpha g_j^\alpha}{m_\alpha^2 - s}
        + f_j^\mathrm{prod} \frac{1 - s_0^\mathrm{prod}}{s - s_0^\mathrm{prod}}.

    Channels: :math:`\pi\pi, K\bar{K}, 4\pi, \eta\eta, \eta\eta'`; ``channel_masses``
    gives the two masses of each two-body channel and the pion mass of the :math:`4\pi`
    channel (:class:`MultiMesonPhaseSpaceFactor`).
    """
    n = len(pole_masses)
    adler = (1 - sA0) / (s - sA0) * (s - sA * m_threshold**2 / 2)
    rho = [
        TwoBodyPhaseSpaceFactor(s, *channel_masses[0]),
        TwoBodyPhaseSpaceFactor(s, *channel_masses[1]),
        MultiMesonPhaseSpaceFactor(s, channel_masses[2][0]),
        TwoBodyPhaseSpaceFactor(s, *channel_masses[3]),
        TwoBodyPhaseSpaceFactor(s, *channel_masses[4]),
    ]

    def k(i: int, j: int) -> sp.Expr:
        poles = sum(
            pole_couplings[a][i] * pole_couplings[a][j] / (pole_masses[a] ** 2 - s)
            for a in range(n)
        )
        return (poles + f_scatt[i][j] * (1 - s0_scatt) / (s - s0_scatt)) * adler

    matrix = [int(i == j) - sp.I * k(i, j) * rho[j] for i in range(n) for j in range(n)]
    p_vector = [
        sum(
            betas[a] * pole_couplings[a][j] / (pole_masses[a] ** 2 - s)
            for a in range(n)
        )
        + f_prod[j] * (1 - s0_prod) / (s - s0_prod)
        for j in range(n)
    ]
    return FirstRowInverseProduct(n, *matrix, *p_vector)
