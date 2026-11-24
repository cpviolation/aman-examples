r"""Lineshapes with the conventions of AmpGen (``src/Lineshapes*.cpp``).

Ported from :mod:`amplitf.ampgen_lineshapes` of cpviolation/AmpliTF. They differ from
:mod:`aman_examples.dynamics` in the normalisation: the Blatt-Weisskopf form factor of
the decay vertex is normalised at :math:`q = 0`, the running width uses the form factor
normalised at the nominal mass, and the amplitudes include AmpGen's ``kFactor``. ``s``
is the invariant mass squared of the resonance, ``s1`` and ``s2`` those of its
daughters, ``L`` the orbital angular momentum and ``radius`` the Blatt-Weisskopf radius
in GeV\ :sup:`-1`.
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from ampform.sympy import argument, unevaluated


def q2(s: sp.Expr, s1: sp.Expr, s2: sp.Expr) -> sp.Expr:
    r"""Squared break-up momentum :math:`q^2(s; s_1, s_2)`, negative below threshold."""
    return (s - 2 * s1 - 2 * s2 + (s1 - s2) ** 2 / s) / 4


def blatt_weisskopf_norm(z: sp.Expr, z0: sp.Expr, L: int) -> sp.Expr:
    r""":math:`B_L^2(z)/B_L^2(z_0)`, :math:`z = q^2 r^2`."""
    if L == 0:
        return sp.Integer(1)
    if L == 1:
        return (1 + z0) / (1 + z)
    if L == 2:  # ruff: ignore[magic-value-comparison]
        return (z0**2 + 3 * z0 + 9) / (z**2 + 3 * z + 9)
    if L == 3:  # ruff: ignore[magic-value-comparison]
        return (z0 * (z0 - 15) ** 2 + 9 * (2 * z0 - 5) ** 2) / (
            z * (z - 15) ** 2 + 9 * (2 * z - 5) ** 2
        )
    msg = f"Blatt-Weisskopf factor for L = {L}"
    raise NotImplementedError(msg)


def kfactor(mass: sp.Expr, width: sp.Expr) -> sp.Expr:
    """AmpGen's normalisation factor of the Breit-Wigner lineshapes."""
    gamma = mass * sp.sqrt(mass**2 + width**2)
    k = 2 * sp.sqrt(2) / sp.pi * mass * width * gamma / sp.sqrt(mass**2 + gamma)
    return sp.sqrt(k)


def running_width(  # ruff: ignore[too-many-positional-arguments]
    s: sp.Expr,
    s1: sp.Expr,
    s2: sp.Expr,
    mass: sp.Expr,
    width: sp.Expr,
    radius: sp.Expr,
    L: int,
) -> sp.Expr:
    r""":math:`\Gamma(s) = \Gamma_0 \frac{B_L^2(q)}{B_L^2(q_0)} \frac{m}{\sqrt{s}} \left(\frac{q}{q_0}\right)^{2L+1}`."""
    q2v = q2(s, s1, s2)
    q2v = sp.Piecewise((q2v, q2v > 0), (0, True))
    q20 = sp.Abs(q2(mass**2, s1, s2))
    bf = blatt_weisskopf_norm(q2v * radius**2, q20 * radius**2, L)
    q2r = q2v / q20
    return width * bf * mass * sp.sqrt(q2r / s) * q2r**L


@unevaluated(real=True)
class FormFactor(sp.Expr):
    r"""Decay-vertex form factor :math:`\sqrt{B_L^2(q)/B_L^2(0)}`. With ``signed``, :math:`q^2` is not taken in absolute value (AmpGen's ``FormFactor`` lineshape of stable particles, here the :math:`D^0`)."""

    s: Any
    s1: Any
    s2: Any
    radius: Any
    L: int = 0
    signed: bool = argument(default=False, sympify=False)
    _latex_repr_ = R"F_{{{L}}}\left({s}\right)"

    def evaluate(self) -> sp.Expr:
        q2v = q2(self.s, self.s1, self.s2)
        z = (q2v if self.signed else sp.Abs(q2v)) * self.radius**2
        return sp.sqrt(blatt_weisskopf_norm(z, 0, int(self.L)))


@unevaluated
class AmpGenBreitWigner(sp.Expr):
    r"""AmpGen ``BW``: :math:`k F_L(q) / (m^2 - s - i m \Gamma(s))`."""

    s: Any
    s1: Any
    s2: Any
    mass: Any
    width: Any
    radius: Any
    L: int = 0
    _latex_repr_ = R"\mathrm{{BW}}_{{{L}}}\left({s}; {mass}, {width}\right)"

    def evaluate(self) -> sp.Expr:
        s, s1, s2, mass, width, radius, L = self.args
        ff = FormFactor(s, s1, s2, radius, L)
        gamma = running_width(s, s1, s2, mass, width, radius, int(L))
        return kfactor(mass, width) * ff / (mass**2 - s - sp.I * mass * gamma)


@unevaluated
class AmpGenGounarisSakurai(sp.Expr):
    """AmpGen ``GounarisSakurai`` (running width without radius, as AmpGen's default)."""

    s: Any
    s1: Any
    s2: Any
    mass: Any
    width: Any
    radius: Any
    mpi: Any
    L: int = 1
    _latex_repr_ = R"\mathrm{{GS}}\left({s}; {mass}, {width}\right)"

    def evaluate(self) -> sp.Expr:  # ruff: ignore[too-many-locals]
        s, s1, s2, mass, width, radius, mpi, L = self.args
        s0 = mass**2

        def k2(x: sp.Expr) -> sp.Expr:
            return x - 4 * mpi**2

        def k(x: sp.Expr) -> sp.Expr:
            return sp.sqrt(sp.Piecewise((k2(x), k2(x) > 0), (0, True)))

        def log_term(x: sp.Expr) -> sp.Expr:
            return sp.log((sp.sqrt(x) + k(x)) / (2 * mpi))

        # at the nominal mass the pions are above threshold: no Piecewise, which SymPy
        # would turn into 1/0 in the (unused) zero branch of the denominators
        k0 = sp.sqrt(k2(s0))
        log0 = sp.log((mass + k0) / (2 * mpi))
        t1 = 2 * s0 * k2(s) * k(s) * log_term(s) / (sp.sqrt(s) * k0**3)
        t2 = log0 * (k2(s) * (s - 3 * s0) + s * (s0 - s)) / (mass * k2(s0))
        t3 = (s0 - s) / k0
        m2 = s0 + width * (t1 + t2 + t3) / sp.pi
        gamma = running_width(s, s1, s2, mass, width, 0, int(L))
        ff = FormFactor(s, s1, s2, radius, L)
        return kfactor(mass, width) * ff / (m2 - s - sp.I * mass * gamma)


def complex_sqrt(x: sp.Expr) -> sp.Expr:
    r"""AmpGen ``complex_sqrt``: :math:`\sqrt{x}` for :math:`x > 0`, :math:`i\sqrt{-x}` otherwise."""
    return sp.Piecewise((sp.sqrt(x), x > 0), (sp.I * sp.sqrt(-x), True))


def phsp_two_body(s: sp.Expr, m0: sp.Expr, m1: sp.Expr) -> sp.Expr:
    r""":math:`\sqrt{1 - (m_0 + m_1)^2/s}` (AmpGen ``phsp_twoBody``)."""
    return complex_sqrt(1 - (m0 + m1) ** 2 / s)


def phsp_focus(s: sp.Expr, m0: sp.Expr, m1: sp.Expr) -> sp.Expr:
    r""":math:`\sqrt{(1 - (m_0 + m_1)^2/s)(1 - (m_0 - m_1)^2/s)}` (AmpGen ``phsp_FOCUS``)."""
    return complex_sqrt((1 - (m0 + m1) ** 2 / s) * (1 - (m0 - m1) ** 2 / s))


def _pol(x: sp.Expr, coefficients: tuple[float, ...]) -> sp.Expr:
    return sum(sp.Float(c) * x**i for i, c in enumerate(coefficients))


@unevaluated
class Flatte(sp.Expr):
    r"""AmpGen ``Flatte`` of the :math:`a_0(980)^0`: :math:`1/(m^2 - s - i m \Gamma(s))`, with :math:`\Gamma(s) = g^2_{\pi\eta}[\rho_{\pi\eta}(s) + (g^2_{KK}/g^2_{\pi\eta})\rho_{KK}(s)]` and :math:`\rho_{KK}` imaginary below the :math:`K^+K^-` threshold. ``kfactor`` (``Flatte.kFactor``) multiplies by AmpGen's ``kFactor`` evaluated with the complex width. No form factor is applied."""

    s: Any
    mass: Any
    g2pieta: Any
    g2kk_by_g2pieta: Any
    mpi: Any
    meta: Any
    mk: Any
    kfactor: bool = argument(default=False, sympify=False)
    _latex_repr_ = R"\mathrm{{Flatte}}\left({s}\right)"

    def evaluate(self) -> sp.Expr:
        s, mass, mpi, meta = self.s, self.mass, self.mpi, self.meta
        rho_pieta = sp.sqrt((1 - (mpi - meta) ** 2 / s) * (1 - (mpi + meta) ** 2 / s))
        width = self.g2pieta * (
            rho_pieta + self.g2kk_by_g2pieta * phsp_two_body(s, self.mk, self.mk)
        )
        bw = 1 / (mass**2 - sp.I * mass * width - s)
        return kfactor(mass, width) * bw if self.kfactor else bw


@unevaluated
class FocusKpi(sp.Expr):
    r"""FOCUS :math:`K\pi` S-wave (AmpGen ``FOCUS``): component ``"Kpi"`` (:math:`I = 1/2`, :math:`K\pi` row of the coupled :math:`K\pi`-:math:`K\eta'` system), ``"KEta"`` or ``"I32"`` (:math:`I = 3/2`)."""

    s: Any
    mk: Any
    mpi: Any
    metap: Any
    component: str = argument(default="Kpi", sympify=False)
    _latex_repr_ = R"\mathrm{{FOCUS}}\left({s}\right)"

    def evaluate(self) -> sp.Expr:  # ruff: ignore[too-many-locals]
        s, mk, mpi = self.s, self.mk, self.mpi
        snorm = mk**2 + mpi**2
        x = s / snorm - 1
        rho1, rho2 = phsp_focus(s, mk, mpi), phsp_focus(s, mk, self.metap)
        if self.component == "I32":
            k32 = (
                (s - sp.Float(0.27))
                / snorm
                * _pol(x, (-0.22147, 0.026637, -0.00092057))
            )
            return 1 / (1 - sp.I * k32 * rho1)
        adler = (s - sp.Float(0.23)) / snorm
        g, sp_ = (sp.Float(0.31072), sp.Float(-0.02323)), sp.Float(1.7919)
        k11 = adler * (g[0] ** 2 / (sp_ - s) + _pol(x, (0.79299, -0.15099, 0.00811)))
        k12 = adler * (
            g[0] * g[1] / (sp_ - s) + _pol(x, (0.15040, -0.038266, 0.0022596))
        )
        k22 = adler * (g[1] ** 2 / (sp_ - s) + _pol(x, (0.17054, -0.0219, 0.00085655)))
        det = k11 * k22 - k12**2
        delta = 1 - rho1 * rho2 * det - sp.I * (rho1 * k11 + rho2 * k22)
        if self.component == "Kpi":
            return (k11 - sp.I * rho2 * det) / delta
        if self.component == "KEta":
            return k12 / delta
        msg = f"Unknown FOCUS component '{self.component}'"
        raise ValueError(msg)


#: Masses hard-coded in AmpGen's kMatrix lineshape
KMATRIX_MPI, KMATRIX_MK, KMATRIX_META, KMATRIX_METAP = (
    0.139570,
    0.493677,
    0.547862,
    0.967780,
)
KMATRIX_CHANNELS = ("pipi", "KK", "4pi", "EtaEta", "EtapEta")


def kmatrix_ampgen(  # ruff: ignore[too-many-positional-arguments]
    s: sp.Expr,
    term: str,
    index: int,
    pole_masses: list[sp.Expr],
    pole_couplings: list[list[sp.Expr]],
    f_scatt: list[sp.Expr],
    s0_scatt: sp.Expr,
    sA0: sp.Expr,
    sA: sp.Expr,
    s0_prod: sp.Expr,
    imaginary_mass: float = 1e-6,
) -> sp.Expr:
    r"""A production term of AmpGen's 5-channel :math:`\pi\pi` S-wave K-matrix.

    The term is ``kMatrix.<term>.<index>``, with propagator :math:`F = (1 -
    iK\rho)^{-1}`:

    - ``pole``/``poleKK``: :math:`\sum_j F_{cj} g_{\alpha j}/(s_\alpha - s)` for pole
      :math:`\alpha` = ``index``;
    - ``prod``/``prodKK``: :math:`F_{cj}(1 - s_0^\mathrm{prod})/(s - s_0^\mathrm{prod})`
      for channel :math:`j` = ``index``;

    with :math:`c = \pi\pi` (``pole``, ``prod``) or :math:`K\bar{K}` (``poleKK``,
    ``prodKK``). Pole positions are :math:`m^2 + i\epsilon`, :math:`\epsilon` =
    ``imaginary_mass`` (``kMatrix::fp``).
    """
    from aman_examples.dynamics import InverseRowProduct  # ruff: ignore[import-outside-top-level]

    n = len(KMATRIX_CHANNELS)
    pole_s = [m**2 + sp.I * sp.Float(imaginary_mass) for m in pole_masses]
    adler = (1 - sA0) * (s - sA * sp.Float(KMATRIX_MPI) ** 2 / 2) / (s - sA0)
    scatt = (1 - s0_scatt) / (s - s0_scatt)
    four_pi = sp.Piecewise(
        (phsp_two_body(s, 2 * KMATRIX_MPI, 2 * KMATRIX_MPI), s > 1),
        (
            _pol(s, (0.00051, -0.01933, 0.13851, -0.20840, -0.29744, 0.13655, 1.07885)),
            True,
        ),
    )
    rho = [
        phsp_two_body(s, KMATRIX_MPI, KMATRIX_MPI),
        phsp_two_body(s, KMATRIX_MK, KMATRIX_MK),
        four_pi,
        phsp_two_body(s, KMATRIX_META, KMATRIX_META),
        phsp_two_body(s, KMATRIX_META, KMATRIX_METAP),
    ]

    def k(a: int, b: int) -> sp.Expr:
        value = sum(
            g[a] * g[b] / (ps - s) for g, ps in zip(pole_couplings, pole_s, strict=True)
        )
        if a == 0 or b == 0:
            value += f_scatt[max(a, b)] * scatt
        return value * adler

    matrix = [int(a == b) - sp.I * k(a, b) * rho[b] for a in range(n) for b in range(n)]
    row = {"pole": 0, "prod": 0, "poleKK": 1, "prodKK": 1}[term]
    if term.startswith("pole"):
        g = pole_couplings[index]
        return InverseRowProduct(row, n, *matrix, *g) / (pole_s[index] - s)
    unit = [int(j == index) for j in range(n)]
    return InverseRowProduct(row, n, *matrix, *unit) * (1 - s0_prod) / (s - s0_prod)


def spline_curvatures(values: list[float], xmin: float, xmax: float) -> list[float]:
    """Second derivatives at the knots of AmpGen's natural cubic spline through ``values`` (uniform knots; zero curvature at both ends)."""
    import numpy as np  # ruff: ignore[import-outside-top-level]

    y = np.asarray(values, dtype=float)
    n = len(y)
    h = (xmax - xmin) / (n - 1)
    m = (
        np.diag(np.full(n - 2, 4.0))
        + np.diag(np.ones(n - 3), 1)
        + np.diag(np.ones(n - 3), -1)
    )
    curvature = np.zeros(n)
    curvature[1:-1] = np.linalg.solve(m, 6.0 * (y[2:] - 2.0 * y[1:-1] + y[:-2]) / h**2)
    return curvature.tolist()


def spline(x: sp.Expr, values: list[float], xmin: float, xmax: float) -> sp.Expr:
    """AmpGen's ``Spline``: natural cubic spline through ``values`` at uniform knots in [``xmin``, ``xmax``]; outside the open interval, the value of the last knot (also below ``xmin``), as in AmpGen."""
    n = len(values)
    h = (xmax - xmin) / (n - 1)
    c = spline_curvatures(values, xmin, xmax)
    # Conditions are single comparisons, evaluated in order: the JAX printer cannot
    # print And/Or (also not the Or that SymPy makes of two pieces with equal values)
    outside = sp.Abs(x - (xmin + xmax) / 2) >= (xmax - xmin) / 2
    pieces = [(sp.Float(values[-1]), outside)]
    for b in range(n - 1):
        dx = x - (xmin + b * h)
        y0, y1, c0, c1 = (
            sp.Float(v) for v in (values[b], values[b + 1], c[b], c[b + 1])
        )
        poly = (
            y0
            + ((y1 - y0) / h - (c1 + 2 * c0) * h / 6) * dx
            + c0 * dx**2 / 2
            + dx**3 * (c1 - c0) / (6 * h)
        )
        pieces.append((poly, x <= xmin + (b + 1) * h if b < n - 2 else True))
    return sp.Piecewise(*pieces)


@unevaluated
class GSpline(sp.Expr):
    r"""AmpGen ``GSpline``: Breit-Wigner with a spline running width, :math:`k F(q) / (m^2 - s - i m \Gamma_0 G(s)/G(m^2))`, with the ``"EFF"`` form factor :math:`e^{-q^2 r^2/2}` or the Blatt-Weisskopf one."""

    s: Any
    s1: Any
    s2: Any
    mass: Any
    width: Any
    radius: Any
    L: int = 0
    knots: tuple = argument(default=(), sympify=False)
    form_factor: str = argument(default="EFF", sympify=False)
    _latex_repr_ = R"\mathrm{{GSpline}}\left({s}; {mass}, {width}\right)"

    def evaluate(self) -> sp.Expr:
        s, mass, width, radius = self.s, self.mass, self.width, self.radius
        values, xmin, xmax = self.knots
        q2v = sp.Abs(q2(s, self.s1, self.s2))
        if self.form_factor == "EFF":
            ff = sp.exp(-q2v * radius**2 / 2)
        else:
            ff = sp.sqrt(blatt_weisskopf_norm(q2v * radius**2, 0, int(self.L)))
        gamma = (
            width
            * spline(s, list(values), xmin, xmax)
            / spline(mass**2, list(values), xmin, xmax)
        )
        return kfactor(mass, width) * ff / (mass**2 - s - sp.I * mass * gamma)
