r"""The BESIII :math:`D^0 \to \pi^+ \pi^- \pi^+ \pi^-` amplitude model.

The model of BESIII, Chin. Phys. C 48 (2024) 083001 (arXiv:2312.02524), from a joint
amplitude analysis of :math:`D^0 \to \pi^+\pi^-\pi^+\pi^-` and
:math:`\pi^+\pi^-\pi^0\pi^0`; only the :math:`\pi^+\pi^-\pi^+\pi^-` amplitudes are built
here. The final state is ordered as :math:`(\pi^+, \pi^-, \pi^+, \pi^-)`.

The formalism follows the paper (GPUPWA):

- covariant Zemach tensors :math:`\tilde{t}^{(L)}` and the spin factors of its Table 4,
  with :math:`r = p_b - p_c` for :math:`a \to b c` in the order written there;
- unnormalised Blatt-Weisskopf factors :math:`B_L(q)`, radii 5.0 GeV\ :sup:`-1` for the
  :math:`D^0` and 3.0 GeV\ :sup:`-1` for the resonances;
- relativistic Breit-Wigner propagators :math:`1/(m_0^2 - s - i\sqrt{s}\,\Gamma(s))`
  with :math:`\Gamma(s) = \Gamma_0 (q/q_0)^{2L+1} (m_0^2/s) (B_L(q)/B_L(q_0))^2`;
  Gounaris-Sakurai for :math:`\rho(770)` and :math:`\rho(1450)`; Flatté for the
  :math:`f_0(980)` (BES, PLB 607 (2005) 243); constant widths for :math:`a_1(1420)` and
  :math:`a_2(1320)`;
- running widths of :math:`a_1(1260)`, :math:`a_1(1640)` and :math:`\pi(1300)` from
  their :math:`3\pi` amplitudes (:math:`\pi^+\pi^+\pi^-` and :math:`\pi^+\pi^0\pi^0`),
  integrated over the Dalitz plot (:func:`compute_running_width_knots`, stored in
  :data:`RUNNING_WIDTH_KNOTS`);
- the :math:`\pi\pi` S-wave from the five-channel K-matrix of BaBar, PRD 78 (2008)
  034023 (as AmpGen's ``kMatrix.opt``), with production vector :math:`s_0^\mathrm{prod}
  = -5` GeV², and its two-S-wave expansion, Eq. (14) of the paper;
- the amplitudes are summed over the exchanges of the identical pions.

As in AmpGen, which caches the sub-expressions shared by its decay chains, the model is
evaluated in stages (:class:`FourPionModel`): the composite four-momenta, then orbital
tensors, lineshapes and K-matrix terms of each pion pair and triplet, then spin factors
are computed once per event as columns (:meth:`FourPionModel.data`), and the chains are
short expressions of these columns. Compiling the model takes seconds instead of minutes
for the chains written in the four-momenta.

Where the paper leaves a convention open, it was fixed by reproducing the published fit
fractions, interference fractions and :math:`F_+`: the isospin factors of the paper's
Eqs. (17)-(22) are already part of the fitted couplings (the isoscalar terms of
:math:`a_1^\pm` and :math:`\pi(1300)^\pm` enter with factor 1, the :math:`\rho^0\rho^0`
and :math:`(\pi\pi)_S(\pi\pi)_S` pairings once each), the :math:`c_{[\pi\pi,K\bar{K}]}`
term is symmetric in the two pion pairs, and the Gounaris-Sakurai width has no barrier
factor. With these, the fit fractions agree with the paper's within their uncertainties
(:math:`\chi^2 \approx 2` for 19 values), every interference fraction has the paper's
sign, and :math:`F_+ \approx 75\%` (paper: :math:`75.2 \pm 1.9\%`). The largest
difference is the :math:`(\pi\pi)_S(\pi\pi)_S` fit fraction, about 49% against
:math:`62.8 \pm 10.8\%`.

Resonance masses and widths are those of the paper (fitted :math:`a_1(1260)` and
:math:`\pi(1300)`, COMPASS for the :math:`a_1(1420)`) or of the PDG 2022. The couplings
of the sub-decays of :math:`a_1(1260)` and :math:`\pi(1300)` are fixed to the paper's
values; the :math:`D^0` couplings, including the production-vector coefficients of the
S-waves, are the model's free couplings.
"""

from __future__ import annotations

import cmath
import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import sympy as sp
from ampform.sympy import NumPyPrintable
from tensorwaves.function.sympy import create_function

from aman_examples.ampgen import covariant as cov
from aman_examples.ampgen import lineshapes as ls
from aman_examples.ampgen.model import AmpGenModel
from aman_examples.phasespace import PairPhaseSpaceGenerator

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sympy.printing.latex import LatexPrinter
    from sympy.printing.numpy import NumPyPrinter
    from tensorwaves.interface import DataSample

FINAL_STATE = ["pi+", "pi-", "pi+", "pi-"]
M_D0 = 1.86484
M_PI, M_PI0, M_K, M_K0 = 0.13957039, 0.1349768, 0.493677, 0.497611
R_D, R_RESONANCE = 5.0, 3.0
S0_PROD = -5.0

RESONANCES = {  # (mass, width) in GeV
    "rho(770)": (0.77526, 0.1491),
    "rho(1450)": (1.465, 0.400),
    "f2(1270)": (1.2755, 0.1867),
    "a1(1260)": (1.193, 0.487),  # fitted in the paper
    "a1(1420)": (1.411, 0.161),  # COMPASS, PRD 98 (2018) 092003
    "a1(1640)": (1.655, 0.254),
    "a2(1320)": (1.3182, 0.1070),
    "pi(1300)": (1.534, 0.610),  # fitted in the paper
}
F0_FLATTE = (0.965, 0.165, 4.21)  # mass, g_pipi, g_KK/g_pipi (BES, PLB 607 (2005) 243)


def _polar(magnitude: float, phase: float) -> complex:
    return cmath.rect(magnitude, phase)


A1_DECAYS = {  # relative to a1(1260) -> rho pi [S]; Table 6 of the paper
    "rho_S": 1.0,
    "rho_D": _polar(0.060, 0.01),
    "f2_P": _polar(0.311, -1.58),
    "S_pole0": _polar(0.83, -2.18),
    "S_prod0": _polar(2.47, 0.34),
    "S_prod1": _polar(6.59, 1.76),
}
PI1300_DECAYS = {
    "rho_P": 1.0,
    "S_pole0": _polar(5.0, -0.64),
    "S_prod1": _polar(43.8, 0.34),
}

COUPLINGS = {  # D0 couplings (Table 5 of the paper), pi+pi-pi+pi- terms
    "a1(1260)+ pi-": 100.0,
    "a1(1260)- pi+": _polar(35.3, 0.23),
    "a1(1420)+ pi-": _polar(19.0, 2.70),
    "a1(1640)+ pi-": _polar(20.1, -2.07),
    "a1(1640)- pi+": _polar(10.5, -1.26),
    "a2(1320)+ pi-": _polar(0.23, -2.92),
    "a2(1320)- pi+": _polar(0.30, -0.47),
    "pi(1300)+ pi-": _polar(76.3, -2.325),
    "pi(1300)- pi+": _polar(65.1, -2.631),
    "rho rho [S]": _polar(6.1, -3.10),
    "rho rho [P]": _polar(6.17, 1.62),
    "rho rho [D]": _polar(4.54, -3.06),
    "rho rho(1450) [P]": _polar(13.9, 0.68),
    "rho rho(1450) [D]": _polar(5.6, 3.08),
    "rho S [beta1]": _polar(8.4, -1.68),
    "rho S [f_pipi]": _polar(40.7, -0.50),
    "rho S [f_KK]": _polar(121, 1.73),
    "S S [a11]": _polar(2224, -1.044),
    "S S [a12]": _polar(7287, 1.727),
    "S S [b2_pipi]": _polar(8816, -1.107),
    "S S [c_pipi_pipi]": _polar(2433, 1.796),
    "S S [c_pipi_KK]": _polar(5417, 2.68),
    "f2 S [f_pipi]": _polar(18.3, -1.39),
    "f2 S [f_KK]": _polar(56, 2.29),
}
FIXED = {"a1(1260)+ pi-"}

FIT_FRACTIONS = {  # %, pi+pi-pi+pi- column of Table 5: (value, total uncertainty)
    "a1(1260)+ pi-": (82.2, 16.3),
    "a1(1260)- pi+": (10.3, 2.9),
    "a1(1420)+ pi-": (0.6, 0.3),
    "a1(1640)+ pi-": (1.7, 1.0),
    "a1(1640)- pi+": (0.5, 0.5),
    "a2(1320)+ pi-": (0.2, 0.15),
    "a2(1320)- pi+": (0.4, 0.15),
    "pi(1300)+ pi-": (32.3, 5.2),
    "pi(1300)- pi+": (23.5, 4.6),
    "rho rho [S]": (1.7, 0.8),
    "rho rho [P]": (9.8, 1.3),
    "rho rho [D]": (23.1, 3.2),
    "rho rho(1450) [P]": (1.0, 0.7),
    "rho rho(1450) [D]": (1.5, 1.4),
    "rho S": (2.7, 1.8),
    "S S": (62.8, 10.8),
    "f2 S": (1.8, 1.4),
}
F_PLUS = (75.2, 1.9)  # %, CP-even fraction of pi+pi-pi+pi-


# --- tensors (components (x, y, z, E), upper indices; see ampgen.covariant)
def _t1(pa, pb, pc):
    r"""Upper-index :math:`\tilde{t}^{(1)}(a \to bc)`, :math:`r = p_b - p_c`."""
    return cov.orbital_pwave(pa, cov.add(pb, pc, -1))


def _t2(pa, pb, pc):
    r"""Upper-index :math:`\tilde{t}^{(2)}(a \to bc)`."""
    return cov.orbital_dwave(pa, cov.add(pb, pc, -1))


def _p1(p):
    r"""Upper-index :math:`P^{(1)\mu\nu}(p) = -g^{\mu\nu} + p^\mu p^\nu/p^2`."""
    return [[-x for x in row] for row in cov.spin1_projector(p)]


def _lower2(t):
    m = cov.METRIC
    return [[m[i] * m[j] * t[i][j] for j in cov.INDICES] for i in cov.INDICES]


def _p1_times(p, v):
    r""":math:`P^{(1)\mu\nu}(p)\, v_\nu` for an upper-index vector :math:`v`."""
    proj, low = _p1(p), cov.lower(v)
    return [sum(proj[i][j] * low[j] for j in cov.INDICES) for i in cov.INDICES]


def _p2_vector(p, t_a, t_t):
    r""":math:`P^{(2)\mu\nu\rho\sigma}(p)\, t_{a,\nu}\, t_{T,\rho\sigma}`, upper index.

    For :math:`t_a` transverse to :math:`p` and a symmetric :math:`t_T`, this is
    :math:`-P^{(1)\mu\nu} (t_T t_a)_\nu + \frac{1}{3} (P^{(1)}_{\rho\sigma}
    t_T^{\rho\sigma})\, t_a^\mu`.
    """
    idx, metric = cov.INDICES, cov.METRIC
    low_a = cov.lower(t_a)
    t_ta = [sum(t_t[i][j] * low_a[j] for j in idx) for i in idx]  # (t_T t_a)^mu
    proj = _p1(p)
    trace = sum(
        metric[a] * metric[b] * proj[a][b] * t_t[a][b] for a in idx for b in idx
    )
    projected = _p1_times(p, t_ta)
    return [-projected[i] + trace * t_a[i] / 3 for i in idx]


def _mass2(p):
    return cov.dot(p, p)


# --- lineshapes
def _barrier(L: int, q2: sp.Expr, radius: float) -> sp.Expr:
    """BESIII's unnormalised Blatt-Weisskopf factor :math:`B_L(q)`."""
    qr2 = sp.Float(1 / radius**2)
    if L == 0:
        return sp.Integer(1)
    if L == 1:
        return sp.sqrt(2 / (q2 + qr2))
    if L == 2:  # ruff: ignore[magic-value-comparison]
        return sp.sqrt(13 / (q2**2 + 3 * q2 * qr2 + 9 * qr2**2))
    raise NotImplementedError(L)


def _q2(parent, d1, d2):
    return ls.q2(_mass2(parent), _mass2(d1), _mass2(d2))


def _two_body_width(s, name, L, m1=M_PI, m2=M_PI):
    m0, g0 = RESONANCES[name]
    q2 = sp.Max(ls.q2(s, m1**2, m2**2), 0)  # 0 below threshold (pi0 pi0 in 3pi widths)
    q02 = ls.q2(m0**2, m1**2, m2**2)
    ratio = (_barrier(L, q2, R_RESONANCE) / _barrier(L, q02, R_RESONANCE)) ** 2
    return g0 * (q2 / q02) ** sp.Rational(2 * L + 1, 2) * m0**2 / s * ratio


def _rbw(s, mass, width):
    return 1 / (mass**2 - s - sp.I * sp.sqrt(s) * width)


def _gounaris_sakurai(s, name):
    m0, g0 = RESONANCES[name]
    mpi = M_PI
    # below the pi+ pi- threshold (only for pi0 in the 3pi widths): q -> 0
    q = sp.sqrt(sp.Max(ls.q2(s, mpi**2, mpi**2), 1e-12))
    q0 = float(np.sqrt(m0**2 / 4 - mpi**2))
    m = sp.sqrt(s)

    def h(mm, qq):
        return 2 / sp.pi * qq / mm * sp.log((mm + 2 * qq) / (2 * mpi))

    h0 = float(h(m0, q0))
    dh = h0 * (1 / (8 * q0**2) - 1 / (2 * m0**2)) + 1 / (2 * np.pi * m0**2)
    f = g0 * m0**2 / q0**3 * (q**2 * (h(m, q) - h0) + (m0**2 - s) * q0**2 * dh)
    d = (
        3 / np.pi * mpi**2 / q0**2 * np.log((m0 + 2 * q0) / (2 * mpi))
        + m0 / (2 * np.pi * q0)
        - mpi**2 * m0 / (np.pi * q0**3)
    )
    width = g0 * (q / q0) ** 3 * (m0 / m)
    return (1 + d * g0 / m0) / (m0**2 - s + f - sp.I * m0 * width)


def _flatte_f0(s):
    m0, g_pipi, ratio = F0_FLATTE

    def rho(m1, m2):
        return ls.complex_sqrt((1 - (m1 + m2) ** 2 / s) * (1 - (m1 - m2) ** 2 / s))

    rho_pipi = rho(M_PI, M_PI) * 2 / 3 + rho(M_PI0, M_PI0) / 3
    rho_kk = (rho(M_K, M_K) + rho(M_K0, M_K0)) / 2
    return 1 / (m0**2 - s - sp.I * g_pipi * (rho_pipi + ratio * rho_kk))


_KMATRIX = {  # BaBar 2008 / AmpGen kMatrix.opt
    "pole_masses": [0.651, 1.2036, 1.55817, 1.21, 1.82206],
    "pole_couplings": [
        [0.22889, -0.55377, 0, -0.39899, -0.34639],
        [0.94128, 0.55095, 0, 0.39065, 0.31503],
        [0.36856, 0.23888, 0.55639, 0.1834, 0.18681],
        [0.3365, 0.40907, 0.85679, 0.19906, -0.00984],
        [0.18171, -0.17558, -0.79658, -0.00355, 0.22358],
    ],
    "f_scatt": [0.23399, 0.15044, -0.20545, 0.32825, 0.35412],
    "s0_scatt": -3.92637,
    "sA0": -0.15,
    "sA": 1.0,
    "s0_prod": S0_PROD,
}
M_ETAP = 0.95778  # BaBar 2008 and PDG (AmpGen's kMatrix uses 0.96778)


def _kmatrix_vector(term: str, s: sp.Expr) -> list[sp.Expr]:
    """P-vector of a term: ``pole<a>`` (g^a/(m_a^2 - s)) or ``prod<channel>``."""
    kind, index = term[:4], int(term[4:])
    if kind == "pole":
        pole = sp.Float(_KMATRIX["pole_masses"][index]) ** 2
        return [sp.Float(g) / (pole - s) for g in _KMATRIX["pole_couplings"][index]]
    unit = [sp.Integer(int(j == index)) for j in range(5)]
    return [u * (1 - S0_PROD) / (s - S0_PROD) for u in unit]


def _kmatrix_matrix(s: sp.Expr) -> list[sp.Expr]:
    r""":math:`1 - iK\rho` of the five-channel K-matrix, row-major.

    As :func:`.kmatrix_ampgen`, with BaBar's phase-space factors :math:`\rho = \sqrt{1 -
    (m_1 + m_2)^2/s}` and the PDG :math:`\eta'` mass.
    """
    km = _KMATRIX
    mpi = ls.KMATRIX_MPI
    adler = (1 - km["sA0"]) * (s - km["sA"] * mpi**2 / 2) / (s - km["sA0"])
    scatt = (1 - km["s0_scatt"]) / (s - km["s0_scatt"])
    four_pi = sp.Piecewise(
        (ls.phsp_two_body(s, 2 * mpi, 2 * mpi), s > 1),
        (
            sum(
                c * s**i
                for i, c in enumerate((
                    0.00051,
                    -0.01933,
                    0.13851,
                    -0.20840,
                    -0.29744,
                    0.13655,
                    1.07885,
                ))
            ),
            True,
        ),
    )
    rho = [
        ls.phsp_two_body(s, mpi, mpi),
        ls.phsp_two_body(s, ls.KMATRIX_MK, ls.KMATRIX_MK),
        four_pi,
        ls.phsp_two_body(s, ls.KMATRIX_META, ls.KMATRIX_META),
        ls.phsp_two_body(s, ls.KMATRIX_META, M_ETAP),
    ]
    poles = [
        (sp.Float(m) ** 2, [sp.Float(x) for x in g])
        for m, g in zip(km["pole_masses"], km["pole_couplings"], strict=True)
    ]

    def k(a, b):
        value = sum(g[a] * g[b] / (ps - s) for ps, g in poles)
        if a == 0 or b == 0:
            value += sp.Float(km["f_scatt"][max(a, b)]) * scatt
        return value * adler

    return [int(a == b) - sp.I * k(a, b) * rho[b] for a in range(5) for b in range(5)]


def _swave(s, coefficients: dict[str, complex]) -> sp.Expr:
    r""":math:`[(1 - iK\rho)^{-1} P]_{\pi\pi}` with :math:`P = \sum` coefficient :math:`\times` term."""
    from aman_examples.dynamics import InverseRowProduct  # ruff: ignore[import-outside-top-level]

    vector = [sp.Integer(0)] * 5
    for term, c in coefficients.items():
        vector = [
            v + c * x for v, x in zip(vector, _kmatrix_vector(term, s), strict=True)
        ]
    return InverseRowProduct(0, 5, *_kmatrix_matrix(s), *vector)


def _kmatrix_term(s, term: str) -> sp.Expr:
    """A P-vector term of the pi pi S-wave: ``pole<a>`` or ``prod<channel>``."""
    return _swave(s, {term: 1})


# --- three-body running widths
S3_MIN, S3_MAX, N_KNOTS = (3 * M_PI) ** 2, (M_D0 - M_PI) ** 2 + 0.05, 60


def _a1_vector(pa, p_plus, p_minus, p_bach, rho_sign, f_sign, decays):  # ruff: ignore[too-many-positional-arguments]
    r""":math:`A^\mu(a_1 \to (\rho, f \to p_+ p_-)\, p_\mathrm{bach})` with ``decays``."""
    p_iso = cov.add(p_plus, p_minus)
    s_iso = _mass2(p_iso)
    t_iso = _t1(p_iso, p_plus, p_minus)
    q2a = _q2(pa, p_iso, p_bach)
    rho = _gounaris_sakurai(s_iso, "rho(770)") * _barrier(
        1, ls.q2(s_iso, M_PI**2, M_PI**2), R_RESONANCE
    )
    out = [sp.Integer(0)] * 4
    if rho_sign and "rho_S" in decays:
        term = cov.scale(_p1_times(pa, t_iso), rho_sign * decays["rho_S"] * rho)
        out = cov.add(out, term)
    if rho_sign and "rho_D" in decays:
        t2a, low = _t2(pa, p_iso, p_bach), cov.lower(t_iso)
        vec = [sum(t2a[i][j] * low[j] for j in cov.INDICES) for i in cov.INDICES]
        factor = rho_sign * decays["rho_D"] * rho * _barrier(2, q2a, R_RESONANCE)
        out = cov.add(out, cov.scale(vec, factor))
    if f_sign and "f2_P" in decays:
        vec = _p2_vector(pa, _t1(pa, p_iso, p_bach), _t2(p_iso, p_plus, p_minus))
        f2 = _rbw(
            s_iso, RESONANCES["f2(1270)"][0], _two_body_width(s_iso, "f2(1270)", 2)
        )
        f2 *= _barrier(2, ls.q2(s_iso, M_PI**2, M_PI**2), R_RESONANCE)
        factor = f_sign * decays["f2_P"] * f2 * _barrier(1, q2a, R_RESONANCE)
        out = cov.add(out, cov.scale(vec, factor))
    swave = {k[2:]: v for k, v in decays.items() if k.startswith("S_")}
    if f_sign and swave:
        factor = f_sign * _swave(s_iso, swave) * _barrier(1, q2a, R_RESONANCE)
        out = cov.add(out, cov.scale(_t1(pa, p_iso, p_bach), factor))
    return out


def _pi1300_scalar(pa, p_plus, p_minus, p_bach, rho_sign, f_sign):  # ruff: ignore[too-many-positional-arguments]
    p_iso = cov.add(p_plus, p_minus)
    s_iso = _mass2(p_iso)
    out = sp.Integer(0)
    if rho_sign:
        rho = _gounaris_sakurai(s_iso, "rho(770)") * _barrier(
            1, ls.q2(s_iso, M_PI**2, M_PI**2), R_RESONANCE
        )
        spin = cov.dot(_t1(pa, p_iso, p_bach), _t1(p_iso, p_plus, p_minus))
        out += rho_sign * spin * rho * _barrier(1, _q2(pa, p_iso, p_bach), R_RESONANCE)
    if f_sign:
        swave = {k[2:]: v for k, v in PI1300_DECAYS.items() if k.startswith("S_")}
        out += f_sign * _swave(s_iso, swave)
    return out


def _three_pion_amplitude(name: str, p, *, neutral: bool):
    """The 3pi amplitude of an a1+ or pi(1300)+, summed over identical pions."""
    if name == "pi(1300)":

        def amplitude(pa, plus, minus, bach, rho_sign, f_sign):  # ruff: ignore[too-many-positional-arguments]
            return _pi1300_scalar(pa, plus, minus, bach, rho_sign, f_sign)

    else:
        decays = A1_DECAYS if name == "a1(1260)" else {"rho_S": 1.0}

        def amplitude(pa, plus, minus, bach, rho_sign, f_sign):  # ruff: ignore[too-many-positional-arguments]
            return _a1_vector(pa, plus, minus, bach, rho_sign, f_sign, decays)

    pa = cov.add(cov.add(p[0], p[1]), p[2])
    total = None
    if not neutral:  # (pi+, pi+, pi-): rho0/f (pi+ pi-) pi+
        orders = [(p[0], p[1], p[2]), (p[1], p[0], p[2])]
        parts = [amplitude(pa, a, c, b, 1, 1) for a, b, c in orders]
    else:  # (pi+, pi0, pi0): rho+ (pi+ pi0) pi0 and f (pi0 pi0) pi+, f with 1/2
        orders = [(p[0], p[1], p[2]), (p[0], p[2], p[1])]
        parts = []
        for a, b, c in orders:
            parts.append(amplitude(pa, a, b, c, 1, 0))
            parts.append(amplitude(pa, b, c, a, 0, sp.Rational(1, 2)))
    for part in parts:
        total = (
            part
            if total is None
            else (cov.add(total, part) if isinstance(part, list) else total + part)
        )
    return total


def _dalitz_momenta(m_parent: float, masses, n: int):  # ruff: ignore[too-many-locals]
    """Momenta (x, y, z, E) on an n x n midpoint grid of the Dalitz plot, with weights."""
    m1, m2, m3 = masses
    lo, hi = (m1 + m2) ** 2, (m_parent - m3) ** 2
    s12 = lo + (np.arange(n) + 0.5) * (hi - lo) / n
    e2s = (s12 - m1**2 + m2**2) / (2 * np.sqrt(s12))  # in the 12 frame
    e3s = (m_parent**2 - s12 - m3**2) / (2 * np.sqrt(s12))
    p2s = np.sqrt(np.clip(e2s**2 - m2**2, 0, None))
    p3s = np.sqrt(np.clip(e3s**2 - m3**2, 0, None))
    s23_lo = (e2s + e3s) ** 2 - (p2s + p3s) ** 2
    s23_hi = (e2s + e3s) ** 2 - (p2s - p3s) ** 2
    u = (np.arange(n) + 0.5) / n
    s12g = np.repeat(s12, n)
    s23g = (s23_lo[:, None] + u[None, :] * (s23_hi - s23_lo)[:, None]).ravel()
    weights = np.repeat((s23_hi - s23_lo) * (hi - lo) / n / n, n) / m_parent**2
    s13g = m_parent**2 + m1**2 + m2**2 + m3**2 - s12g - s23g
    e1 = (m_parent**2 + m1**2 - s23g) / (2 * m_parent)
    e3 = (m_parent**2 + m3**2 - s12g) / (2 * m_parent)
    p1 = np.sqrt(np.clip(e1**2 - m1**2, 0, None))
    p3 = np.sqrt(np.clip(e3**2 - m3**2, 0, None))
    cos13 = np.clip(
        (2 * e1 * e3 - (s13g - m1**2 - m3**2)) / np.where(p1 * p3 > 0, 2 * p1 * p3, 1),
        -1,
        1,
    )
    sin13 = np.sqrt(1 - cos13**2)
    zero = np.zeros_like(e1)
    v3 = np.stack([zero, zero, p3, e3], axis=-1)
    v1 = np.stack([p1 * sin13, zero, p1 * cos13, e1], axis=-1)
    v2 = np.stack([-v1[:, 0], zero, -v1[:, 2] - p3, m_parent - e1 - e3], axis=-1)
    return [v1, v2, v3], weights


@functools.cache
def _three_pion_function(name: str, *, neutral: bool) -> Callable:

    symbols = [[sp.Symbol(f"q{i}_{c}", real=True) for c in "xyzE"] for i in range(3)]
    amp = _three_pion_amplitude(name, symbols, neutral=neutral)
    if isinstance(amp, list):
        pa = cov.add(cov.add(symbols[0], symbols[1]), symbols[2])
        exprs = {f"A{i}": a.doit() for i, a in enumerate(amp)}
        exprs.update({f"P{i}": c for i, c in enumerate(pa)})
    else:
        exprs = {"A": amp.doit()}
    functions = {k: create_function(e, backend="numpy") for k, e in exprs.items()}

    def evaluate(data):
        values = {k: np.asarray(f(data)) for k, f in functions.items()}
        if "A" in values:
            return np.abs(values["A"]) ** 2
        a = np.stack([values[f"A{i}"] for i in range(4)], axis=-1)
        p = np.stack(
            [np.broadcast_to(values[f"P{i}"], a.shape[:1]) for i in range(4)], axis=-1
        )
        a_dot = np.einsum("...i,i,...i->...", np.conj(a), np.array(cov.METRIC), a)
        p_dot = np.einsum("...i,i,...i->...", p, np.array(cov.METRIC), p)
        a_p = np.einsum("...i,i,...i->...", np.conj(a), np.array(cov.METRIC), p)
        return np.real(-a_dot + np.abs(a_p) ** 2 / p_dot)  # P1_{mu nu} A^mu* A^nu

    return evaluate


def _three_pion_rate(name: str, mass: float, n_grid: int) -> float:
    r""":math:`\int \sum_\mathrm{spin} |A|^2 d\Phi_3` over both charge configurations."""
    total = 0.0
    for neutral in (False, True):
        masses = (M_PI, M_PI0, M_PI0) if neutral else (M_PI, M_PI, M_PI)
        if mass <= sum(masses):
            continue
        momenta, weights = _dalitz_momenta(mass, masses, n_grid)
        data = {
            f"q{i}_{c}": v[:, j]
            for i, v in enumerate(momenta)
            for j, c in enumerate("xyzE")
        }
        total += 0.5 * np.sum(
            weights * _three_pion_function(name, neutral=neutral)(data)
        )
    return total


@functools.cache
def compute_running_width_knots(name: str, n_grid: int = 80) -> tuple[float, ...]:
    r""":math:`\Gamma(s)/\Gamma_0` of ``a1(1260)``, ``a1(1640)`` or ``pi(1300)`` at the knots.

    :math:`\Gamma(s) \propto (m_0/\sqrt{s}) \int \sum |A_{R \to 3\pi}|^2 d\Phi_3` (Eq.
    (12) of the paper), normalised to :math:`\Gamma_0` at :math:`s = m_0^2`, at
    ``N_KNOTS`` values of :math:`s` from ``S3_MIN`` to ``S3_MAX``; the Dalitz plots are
    integrated on ``n_grid`` x ``n_grid`` grids.
    """
    m0 = RESONANCES[name][0]
    knots = np.linspace(S3_MIN, S3_MAX, N_KNOTS)

    def g(s):
        return m0 / np.sqrt(s) * _three_pion_rate(name, float(np.sqrt(s)), n_grid)

    norm = g(m0**2)
    return tuple(float(g(s) / norm) if s > S3_MIN else 0.0 for s in knots)


RUNNING_WIDTH_KNOTS = {  # compute_running_width_knots(name), stored
    "a1(1260)": (
        0,
        0.00065873813,
        0.0035822317,
        0.0093378401,
        0.018096675,
        0.030065127,
        0.04565975,
        0.065638895,
        0.091264677,
        0.12455237,
        0.16859939,
        0.22776724,
        0.30650225,
        0.404225,
        0.50939879,
        0.60655547,
        0.68818406,
        0.75418761,
        0.80700893,
        0.84933667,
        0.8835256,
        0.91149783,
        0.93480771,
        0.95479263,
        0.97242223,
        0.98815142,
        1.0024107,
        1.0153684,
        1.0273621,
        1.0389995,
        1.0502688,
        1.0611021,
        1.0718065,
        1.0826186,
        1.0937883,
        1.1053759,
        1.1175839,
        1.1305629,
        1.144401,
        1.1589865,
        1.174489,
        1.1907179,
        1.2075971,
        1.2251197,
        1.2430844,
        1.2613592,
        1.2800322,
        1.2989072,
        1.3177383,
        1.3365728,
        1.3553964,
        1.3741025,
        1.3928473,
        1.4113524,
        1.4298457,
        1.4481282,
        1.4667308,
        1.4850251,
        1.5030219,
        1.5208875,
    ),
    "a1(1640)": (
        0,
        0.00024572923,
        0.0014415334,
        0.0040420231,
        0.0083985867,
        0.01490811,
        0.024102851,
        0.036745127,
        0.053961357,
        0.077446517,
        0.10974779,
        0.15446495,
        0.21547514,
        0.29311668,
        0.37926639,
        0.46193174,
        0.53448358,
        0.59595046,
        0.64757314,
        0.69101389,
        0.72781816,
        0.75925741,
        0.78632261,
        0.80977738,
        0.83021522,
        0.84810462,
        0.8638219,
        0.87767538,
        0.88992308,
        0.90078555,
        0.91045437,
        0.91909676,
        0.92685768,
        0.93386076,
        0.94020967,
        0.94599027,
        0.95127345,
        0.95611798,
        0.96057307,
        0.96468051,
        0.96847618,
        0.97199134,
        0.97525342,
        0.97828677,
        0.9811131,
        0.98375191,
        0.9862208,
        0.9885357,
        0.99071108,
        0.99276015,
        0.99469495,
        0.99652647,
        0.99826477,
        0.99991902,
        1.0014976,
        1.0030082,
        1.0044576,
        1.0058523,
        1.0071979,
        1.0084996,
    ),
    "pi(1300)": (
        0,
        0.0033572899,
        0.011106606,
        0.021368746,
        0.033050836,
        0.045563933,
        0.058634982,
        0.07219812,
        0.086340397,
        0.10128698,
        0.11742161,
        0.13533742,
        0.15588604,
        0.18007937,
        0.20863479,
        0.24146364,
        0.27777456,
        0.31651992,
        0.35666621,
        0.3973135,
        0.43767582,
        0.47691193,
        0.5139501,
        0.54819582,
        0.58182597,
        0.61437706,
        0.64536407,
        0.67475196,
        0.70207422,
        0.72753228,
        0.75170925,
        0.77491546,
        0.79679108,
        0.81755345,
        0.83721753,
        0.85590937,
        0.87353985,
        0.89038662,
        0.90631992,
        0.92154643,
        0.93603793,
        0.94989085,
        0.9630363,
        0.97567939,
        0.98774663,
        0.99929592,
        1.0102398,
        1.0207237,
        1.0308543,
        1.0407181,
        1.050381,
        1.0600985,
        1.0697102,
        1.0798275,
        1.0897827,
        1.0996652,
        1.1093889,
        1.1188446,
        1.1283653,
        1.1374199,
    ),
}


class Interpolation(NumPyPrintable):
    r"""Linear interpolation of a table, :math:`f(x)` (``interp`` of NumPy or JAX).

    The arguments are :math:`x`, then the :math:`n` abscissae and the :math:`n` values.
    """

    is_commutative = True

    def __new__(cls, x: sp.Expr, *table: float, **kwargs: Any) -> Self:
        if len(table) % 2:
            msg = "Expected as many abscissae as values"
            raise ValueError(msg)
        args = (sp.sympify(x), *(sp.Float(v) for v in table))
        return sp.Expr.__new__(cls, *args, **kwargs)

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:  # ruff: ignore[unused-method-argument]
        return Rf"f_\mathrm{{table}}\left({printer._print(self.args[0])}\right)"  # ruff: ignore[private-member-access]

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:  # ruff: ignore[unused-method-argument]
        printer.module_imports[printer._module].update({"interp", "array"})  # ruff: ignore[private-member-access]
        n = (len(self.args) - 1) // 2
        xp = ", ".join(repr(float(v)) for v in self.args[1 : 1 + n])
        fp = ", ".join(repr(float(v)) for v in self.args[1 + n :])
        x = printer._print(self.args[0])  # ruff: ignore[private-member-access]
        return f"interp({x}, array([{xp}]), array([{fp}]))"


def _spline_values(
    x: np.ndarray, values: list[float], xmin: float, xmax: float
) -> np.ndarray:
    """The natural cubic spline of :func:`.lineshapes.spline` at ``x`` (NumPy)."""
    n = len(values)
    h = (xmax - xmin) / (n - 1)
    c = ls.spline_curvatures(values, xmin, xmax)
    b = np.clip(((x - xmin) // h).astype(int), 0, n - 2)
    dx = x - (xmin + b * h)
    y0, y1 = np.array(values)[b], np.array(values)[b + 1]
    c0, c1 = np.array(c)[b], np.array(c)[b + 1]
    return (
        y0
        + ((y1 - y0) / h - (c1 + 2 * c0) * h / 6) * dx
        + c0 * dx**2 / 2
        + dx**3 * (c1 - c0) / (6 * h)
    )


N_TABLE = 400


def running_width_table(name: str) -> tuple[np.ndarray, np.ndarray]:
    r""":math:`s` and :math:`\Gamma(s)/\Gamma_0`: the cubic spline through :data:`RUNNING_WIDTH_KNOTS` at ``N_TABLE`` points."""
    s = np.linspace(S3_MIN, S3_MAX, N_TABLE)
    return s, _spline_values(s, list(RUNNING_WIDTH_KNOTS[name]), S3_MIN, S3_MAX)


def _running_propagator(s, name):
    m0, g0 = RESONANCES[name]
    ratio = Interpolation(s, *np.concatenate(running_width_table(name)))
    return 1 / (m0**2 - s - sp.I * sp.sqrt(s) * g0 * ratio)


# --- D0 amplitudes, evaluated in two stages as in AmpGen
@dataclass(frozen=True)
class FourPionModel(AmpGenModel):
    """An :class:`.AmpGenModel` whose chains are written in per-event building blocks.

    ``building_blocks`` maps each block symbol (a lineshape of a pion pair or triplet, a
    K-matrix term, or a spin factor with its barrier factors) to its expression in the
    four-momenta; ``chains`` are short expressions of these blocks. Evaluate them on
    :meth:`data`, the building blocks of a sample of four-momenta: each block is
    lambdified separately and computed once per event, which is much faster to compile
    than the chains written in the four-momenta (:meth:`expanded_chains`).
    """

    stages: tuple[dict[sp.Symbol, sp.Expr], ...] = field(default=(), repr=False)

    @property
    def building_blocks(self) -> dict[sp.Symbol, sp.Expr]:
        """All building blocks, each in terms of the four-momenta and earlier blocks."""
        return {s: e for stage in self.stages for s, e in stage.items()}

    @functools.cached_property
    def transformers(self) -> tuple[_StageTransformer, ...]:
        """One data transformer (NumPy) per stage of building blocks."""
        return tuple(_StageTransformer(stage) for stage in self.stages)

    def data(self, sample: DataSample) -> dict[str, Any]:
        """The building blocks of a sample of four-momenta ``p<i>_x``, ..., ``p<i>_E``."""
        data = dict(sample)
        for transformer in self.transformers:
            data.update(transformer(data))
        return data

    def function(self, expression: sp.Expr, backend: str = "jax") -> Callable:
        """``expression`` (e.g. :attr:`intensity`) as a function of four-momenta."""
        values = {
            s: v
            for s, v in self.parameter_defaults.items()
            if s in expression.free_symbols
        }
        function = create_function(expression.xreplace(values).doit(), backend=backend)
        return _ComposedFunction(self, function)

    def expanded_chains(self, names: Sequence[str] | None = None) -> dict[str, sp.Expr]:
        """The chains (all, or ``names``) written in the four-momenta (slow to lambdify)."""
        chains = {n: self.chains[n] for n in (self.chains if names is None else names)}
        for stage in reversed(self.stages):
            chains = {n: e.xreplace(stage) for n, e in chains.items()}
        return chains


class _StageTransformer:
    """All building blocks of a stage, lambdified as one NumPy function (with CSE)."""

    def __init__(self, blocks: dict[sp.Symbol, sp.Expr]) -> None:
        expressions = [e.doit() for e in blocks.values()]
        self.inputs = sorted(
            set().union(*(e.free_symbols for e in expressions)), key=str
        )
        self.outputs = [s.name for s in blocks]
        self.function = sp.lambdify(self.inputs, expressions, modules="numpy", cse=True)

    def __call__(self, data: DataSample) -> dict[str, Any]:
        # NumPy evaluates both branches of the Piecewise square roots of complex_sqrt
        with np.errstate(invalid="ignore", divide="ignore"):
            values = self.function(*(data[s.name] for s in self.inputs))
        return dict(zip(self.outputs, values, strict=True))


class _ComposedFunction:
    """A TensorWaves-like function of four-momenta: building blocks, then ``function``."""

    def __init__(self, model: FourPionModel, function: Callable) -> None:
        self.model, self.function = model, function

    def __call__(self, data: DataSample) -> Any:
        return self.function(self.model.data(data))


class _Blocks:
    """Building blocks, created once per name."""

    def __init__(self, p: list[list[sp.Symbol]]) -> None:
        self.p = p
        self.stage: dict[sp.Symbol, int] = {}
        self.blocks: dict[sp.Symbol, sp.Expr] = {}

    def _register(self, symbol: sp.Symbol, expr: sp.Expr) -> None:
        if symbol not in self.blocks:
            depends = [
                self.stage[x] for x in sp.sympify(expr).free_symbols if x in self.stage
            ]
            self.stage[symbol] = 1 + max(depends, default=-1)
            self.blocks[symbol] = expr

    def stages(self) -> tuple[dict[sp.Symbol, sp.Expr], ...]:
        n = 1 + max(self.stage.values())
        return tuple(
            {s: e for s, e in self.blocks.items() if self.stage[s] == k}
            for k in range(n)
        )

    def get(
        self, name: str, make: Callable[[], sp.Expr], *, real: bool = False
    ) -> sp.Symbol:
        symbol = sp.Symbol(name, real=True) if real else sp.Symbol(name)
        if symbol not in self.blocks:
            self._register(symbol, make())
        return symbol

    def vector(self, name: str, make: Callable[[], list]) -> list[sp.Symbol]:
        """A four-vector (upper index) as four real columns."""
        symbols = [sp.Symbol(f"{name}_{c}", real=True) for c in "xyzE"]
        if symbols[0] not in self.blocks:
            for symbol, expr in zip(symbols, make(), strict=True):
                self._register(symbol, expr)
        return symbols

    def tensor(self, name: str, make: Callable[[], list]) -> list[list[sp.Symbol]]:
        """A rank-2 tensor (upper indices) as 16 real columns."""
        symbols = [
            [sp.Symbol(f"{name}_{a}{b}", real=True) for b in "xyzE"] for a in "xyzE"
        ]
        if symbols[0][0] not in self.blocks:
            for row_s, row_e in zip(symbols, make(), strict=True):
                for symbol, expr in zip(row_s, row_e, strict=True):
                    self._register(symbol, expr)
        return symbols

    def momentum(self, indices) -> list[sp.Expr]:
        """The summed four-momentum of the pions ``indices``, as symbols of a column."""
        if len(indices) == 1:
            return self.p[indices[0]]
        key = "".join(str(i) for i in sorted(indices))
        return self.vector(
            f"p{key}", lambda: [sum(self.p[i][k] for i in indices) for k in range(4)]
        )

    # pion pairs (pi+ i, pi- j)
    def rho(self, i: int, j: int, name: str = "rho(770)") -> sp.Symbol:
        def make():
            s = _mass2(self.momentum((i, j)))
            return _gounaris_sakurai(s, name) * _barrier(
                1, ls.q2(s, M_PI**2, M_PI**2), R_RESONANCE
            )

        tag = "rho" if name == "rho(770)" else "rho1450"
        return self.get(f"{tag}_{i}{j}", make)

    def f2(self, i: int, j: int) -> sp.Symbol:
        def make():
            s = _mass2(self.momentum((i, j)))
            bw = _rbw(s, RESONANCES["f2(1270)"][0], _two_body_width(s, "f2(1270)", 2))
            return bw * _barrier(2, ls.q2(s, M_PI**2, M_PI**2), R_RESONANCE)

        return self.get(f"f2_{i}{j}", make)

    def f0(self, i: int, j: int) -> sp.Symbol:
        return self.get(f"f0_{i}{j}", lambda: _flatte_f0(_mass2(self.momentum((i, j)))))

    def kmatrix(self, i: int, j: int, term: str) -> sp.Symbol:
        return self.get(
            f"K{term}_{i}{j}",
            lambda: _kmatrix_term(_mass2(self.momentum((i, j))), term),
        )

    def swave(self, i: int, j: int, coefficients: dict[str, complex]) -> sp.Expr:
        return sum(c * self.kmatrix(i, j, t) for t, c in coefficients.items())

    # pion triplets
    def propagator(self, indices, name: str) -> sp.Symbol:
        key = "".join(str(i) for i in sorted(indices))

        def make():
            s = _mass2(self.momentum(indices))
            if name in {"a1(1420)", "a2(1320)"}:
                return _rbw(s, *RESONANCES[name])
            return _running_propagator(s, name)

        return self.get(f"P_{name}_{key}", make)

    def spin(self, name: str, make: Callable[[], sp.Expr]) -> sp.Symbol:
        return self.get(f"S_{name}", make, real=True)


def _three_body_terms(blocks: _Blocks, order, charge: str) -> dict[str, sp.Expr]:  # ruff: ignore[too-many-locals]
    """Chains with a pi+- pi-+ pi+- resonance, for one assignment of the pions."""
    a, b, c, d = order
    plus, minus, bach, d_bach, rho_sign = (
        (a, b, c, d, 1) if charge == "+" else (a, b, d, c, -1)
    )
    p = blocks.momentum
    tag = f"{plus}{minus}{bach}{d_bach}"
    pd, pa, p_iso = p((0, 1, 2, 3)), p((plus, minus, bach)), p((plus, minus))

    triplet_tag = "".join(str(i) for i in sorted((plus, minus, bach)))
    pair_tag = f"{plus}{minus}"
    t_d = blocks.vector(f"tD{triplet_tag}", lambda: _t1(pd, pa, blocks.p[d_bach]))
    t_a = blocks.vector(
        f"tA{triplet_tag}_{pair_tag}", lambda: _t1(pa, p_iso, blocks.p[bach])
    )
    t_iso = blocks.vector(
        f"t{pair_tag}", lambda: _t1(p_iso, blocks.p[plus], blocks.p[minus])
    )
    t2_a = blocks.tensor(
        f"TA{triplet_tag}_{pair_tag}", lambda: _t2(pa, p_iso, blocks.p[bach])
    )
    t2_iso = blocks.tensor(
        f"T{pair_tag}", lambda: _t2(p_iso, blocks.p[plus], blocks.p[minus])
    )
    w = blocks.vector(f"w{triplet_tag}_{pair_tag}", lambda: _p1_times(pa, t_iso))
    b_d1 = blocks.get(
        f"BD1_{triplet_tag}",
        lambda: _barrier(1, _q2(pd, pa, blocks.p[d_bach]), R_D),
        real=True,
    )
    b_d2 = blocks.get(
        f"BD2_{triplet_tag}",
        lambda: _barrier(2, _q2(pd, pa, blocks.p[d_bach]), R_D),
        real=True,
    )
    b_a1 = blocks.get(
        f"BA1_{triplet_tag}_{pair_tag}",
        lambda: _barrier(1, _q2(pa, p_iso, blocks.p[bach]), R_RESONANCE),
        real=True,
    )
    b_a2 = blocks.get(
        f"BA2_{triplet_tag}_{pair_tag}",
        lambda: _barrier(2, _q2(pa, p_iso, blocks.p[bach]), R_RESONANCE),
        real=True,
    )

    s_rho_s = blocks.spin(f"rhoS_{tag}", lambda: cov.dot(t_d, w) * b_d1)

    def rho_d():
        low = cov.lower(t_iso)
        vec = [sum(t2_a[i][j] * low[j] for j in cov.INDICES) for i in cov.INDICES]
        return cov.dot(t_d, vec) * b_a2 * b_d1

    s_rho_d = blocks.spin(f"rhoD_{tag}", rho_d)
    f2_vector = blocks.vector(
        f"f2v{triplet_tag}_{pair_tag}", lambda: _p2_vector(pa, t_a, t2_iso)
    )
    s_f2 = blocks.spin(f"f2P_{tag}", lambda: cov.dot(t_d, f2_vector) * b_a1 * b_d1)
    s_s = blocks.spin(f"SP_{tag}", lambda: cov.dot(t_d, t_a) * b_a1 * b_d1)
    t2_d = blocks.tensor(f"TD{triplet_tag}", lambda: _t2(pd, pa, blocks.p[d_bach]))

    def a2_spin():
        idx, metric = cov.INDICES, cov.METRIC
        t_t = t2_a
        y_low = [
            [
                sum(
                    cov.LEVI_CIVITA_LOWER.get((al, la, si, ro), 0)
                    * pa[ro]
                    * w[si]
                    * t_t[la][be]
                    * metric[be]
                    for la in idx
                    for si in idx
                    for ro in idx
                )
                for be in idx
            ]
            for al in idx
        ]
        # Y is transverse to p_a and traceless, so P2 only symmetrises it
        return (
            sum(t2_d[i][j] * (y_low[i][j] + y_low[j][i]) / 2 for i in idx for j in idx)
            * b_d2
            * b_a2
        )

    s_a2 = blocks.spin(f"a2_{tag}", a2_spin)
    s_pi = blocks.spin(f"piP_{tag}", lambda: cov.dot(t_a, t_iso) * b_a1)
    triplet = (plus, minus, bach)
    rho = blocks.rho(plus, minus)
    other = "-" if charge == "+" else "+"
    a1_swave = {k[2:]: v for k, v in A1_DECAYS.items() if k.startswith("S_")}
    pi_swave = {k[2:]: v for k, v in PI1300_DECAYS.items() if k.startswith("S_")}
    out = {
        f"a1(1260){charge} pi{other}": (
            rho_sign * (s_rho_s + A1_DECAYS["rho_D"] * s_rho_d) * rho
            + A1_DECAYS["f2_P"] * s_f2 * blocks.f2(plus, minus)
            + s_s * blocks.swave(plus, minus, a1_swave)
        )
        * blocks.propagator(triplet, "a1(1260)"),
        f"a1(1640){charge} pi{other}": rho_sign
        * s_rho_s
        * rho
        * blocks.propagator(triplet, "a1(1640)"),
        f"a2(1320){charge} pi{other}": rho_sign
        * s_a2
        * rho
        * blocks.propagator(triplet, "a2(1320)"),
        f"pi(1300){charge} pi{other}": (
            rho_sign * s_pi * rho + blocks.swave(plus, minus, pi_swave)
        )
        * blocks.propagator(triplet, "pi(1300)"),
    }
    if charge == "+":
        out["a1(1420)+ pi-"] = (
            s_s * blocks.f0(plus, minus) * blocks.propagator(triplet, "a1(1420)")
        )
    return out


def _two_body_terms(blocks: _Blocks, order) -> dict[str, sp.Expr]:  # ruff: ignore[too-many-locals]
    """Chains with two pion pairs (a b)(c d), for one assignment of the pions."""
    a, b, c, d = order
    p = blocks.momentum
    tag = f"{a}{b}{c}{d}"
    pd, r1, r2 = p((0, 1, 2, 3)), p((a, b)), p((c, d))

    def t_r(i, j):
        return blocks.vector(
            f"t{i}{j}", lambda: _t1(p((i, j)), blocks.p[i], blocks.p[j])
        )

    t_d = blocks.vector(f"tDD{tag}", lambda: _t1(pd, r1, r2))
    t2_d = blocks.tensor(f"TDD{tag}", lambda: _t2(pd, r1, r2))
    t2_r1 = blocks.tensor(f"T{a}{b}", lambda: _t2(r1, blocks.p[a], blocks.p[b]))
    b_d1 = blocks.get(
        f"BDD1_{tag}", lambda: _barrier(1, _q2(pd, r1, r2), R_D), real=True
    )
    b_d2 = blocks.get(
        f"BDD2_{tag}", lambda: _barrier(2, _q2(pd, r1, r2), R_D), real=True
    )
    s_s = blocks.spin(f"VVS_{tag}", lambda: cov.dot(t_r(a, b), t_r(c, d)))
    s_p = blocks.spin(
        f"VVP_{tag}",
        lambda: cov.levi_civita_contract(t_d, t_r(a, b), t_r(c, d), pd) * b_d1,
    )
    s_d = blocks.spin(
        f"VVD_{tag}", lambda: cov.contract_lower(t2_d, t_r(a, b), t_r(c, d)) * b_d2
    )
    s_vs = blocks.spin(f"VS_{tag}", lambda: cov.dot(t_d, t_r(a, b)) * b_d1)
    s_ts = blocks.spin(f"TS_{tag}", lambda: cov.contract_tensors(t2_d, t2_r1) * b_d2)
    rho1, rho2, rho2h = (
        blocks.rho(a, b),
        blocks.rho(c, d),
        blocks.rho(c, d, "rho(1450)"),
    )

    def k(i, j, term):
        return blocks.kmatrix(i, j, term)

    half = sp.Rational(1, 2)  # rho0 rho0 and S S: each pairing comes twice
    return {
        "rho rho [S]": half * s_s * rho1 * rho2,
        "rho rho [P]": half * s_p * rho1 * rho2,
        "rho rho [D]": half * s_d * rho1 * rho2,
        "rho rho(1450) [P]": s_p * rho1 * rho2h,
        "rho rho(1450) [D]": s_d * rho1 * rho2h,
        "rho S [beta1]": s_vs * rho1 * k(c, d, "pole0"),
        "rho S [f_pipi]": s_vs * rho1 * k(c, d, "prod0"),
        "rho S [f_KK]": s_vs * rho1 * k(c, d, "prod1"),
        "S S [a11]": k(a, b, "pole0") * k(c, d, "pole0"),
        "S S [a12]": half
        * (k(a, b, "pole0") * k(c, d, "pole1") + k(a, b, "pole1") * k(c, d, "pole0")),
        "S S [b2_pipi]": half
        * (k(a, b, "prod0") * k(c, d, "pole1") + k(c, d, "prod0") * k(a, b, "pole1")),
        "S S [c_pipi_pipi]": half * k(a, b, "prod0") * k(c, d, "prod0"),
        "S S [c_pipi_KK]": half
        * (k(a, b, "prod0") * k(c, d, "prod1") + k(a, b, "prod1") * k(c, d, "prod0")),
        "f2 S [f_pipi]": s_ts * blocks.f2(a, b) * k(c, d, "prod0"),
        "f2 S [f_KK]": s_ts * blocks.f2(a, b) * k(c, d, "prod1"),
    }


def coupling_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(f"C_{{{name}}}")


ORDERINGS = ((0, 1, 2, 3), (2, 1, 0, 3), (0, 3, 2, 1), (2, 3, 0, 1))


def formulate_model() -> FourPionModel:
    r"""The BESIII :math:`D^0 \to \pi^+\pi^-\pi^+\pi^-` model, as a :class:`FourPionModel`.

    ``chains`` holds one amplitude per :math:`D^0` coupling (the S-wave chains one per
    production-vector coefficient), summed over the four exchanges of identical pions
    (:data:`ORDERINGS`: pi+ at positions 0 and 2, pi- at 1 and 3); the coupling of
    ``a1(1260)+ pi-`` is fixed.
    """
    momenta = [[sp.Symbol(f"p{i}_{c}", real=True) for c in "xyzE"] for i in range(4)]
    blocks = _Blocks([list(row) for row in momenta])
    chains: dict[str, sp.Expr] = {}
    for order in ORDERINGS:
        terms = _two_body_terms(blocks, order)
        for charge in "+-":
            terms.update(_three_body_terms(blocks, order, charge))
        for name, expr in terms.items():
            chains[name] = chains.get(name, 0) + expr
    chains = {n: chains[n] for n in COUPLINGS}
    couplings = {name: coupling_symbol(name) for name in chains}
    defaults: dict[sp.Symbol, Any] = {
        coupling_symbol(n): complex(v) for n, v in COUPLINGS.items()
    }
    free = [coupling_symbol(n) for n in COUPLINGS if n not in FIXED]
    table = {"D0": {"mass": M_D0}, "pi+": {"mass": M_PI}, "pi-": {"mass": M_PI}}
    return FourPionModel(
        chains,
        couplings,
        defaults,
        free,
        momenta,
        "D0",
        FINAL_STATE,
        table,
        stages=blocks.stages(),
    )


def fit_fraction_groups() -> dict[str, list[str]]:
    """The chains of each fit fraction of the paper (S-wave terms are summed)."""
    groups: dict[str, list[str]] = {}
    for name in COUPLINGS:
        key = name.split(" [")[0] if name.startswith(("rho S", "S S", "f2 S")) else name
        groups.setdefault(key, []).append(name)
    return groups


def phase_space_generator(model: AmpGenModel | None = None) -> PairPhaseSpaceGenerator:
    r"""Four-body phase space as :math:`(\pi^+\pi^-)(\pi^+\pi^-)`."""
    del model
    return PairPhaseSpaceGenerator((M_PI,) * 4, M_D0)
