r"""The Belle 2018 :math:`D^0 \to K^0_S \pi^+ \pi^-` amplitude model.

The model uses the BaBar 2008 parametrisation, with the resonance parameters and
couplings of the Belle 2018 analysis (:file:`data/belle_model.txt`). It is a coherent
sum of quasi-two-body terms:

- :math:`K^*(892)^\mp`, :math:`K_2^*(1430)^\mp`, :math:`K^*(1410)^\mp`,
  :math:`K^*(1680)^\mp` on :math:`m^2(K^0_S\pi^\mp)`;
- :math:`\rho(770)`, :math:`\omega(782)`, :math:`f_2(1270)`, :math:`\rho(1450)` on
  :math:`m^2(\pi^+\pi^-)`;
- the :math:`K\pi` S-wave (LASS) on both :math:`m^2(K^0_S\pi^\mp)`;
- the :math:`\pi\pi` S-wave (K-matrix) on :math:`m^2(\pi^+\pi^-)`.

Particles are labelled :math:`a = \pi^-`, :math:`b = K^0_S`, :math:`c = \pi^+`; the
kinematic variables are :math:`s_{ab} = m^2(K^0_S\pi^-)` (``m2_kspim``) and
:math:`s_{bc} = m^2(K^0_S\pi^+)` (``m2_kspip``). The Cabibbo-favoured :math:`K^*(892)^-`
is on :math:`s_{ab}`.

Spin factors are the Zemach tensors of the Belle code. As there, the spin-1 factors
carry an overall minus sign with respect to :func:`zemach_spin_factor`; this fixes the
relative sign of P-wave and other terms, in which the Belle couplings were fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sympy as sp
from ampform.sympy import unevaluated
from particle.particle import literals as lp

from aman_examples.dynamics import LASS, BelleBreitWigner, kmatrix_amplitude
from aman_examples.io import load_belle_model

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Components of the coherent sum, in the order of the BaBar 2008 model. The
#: constant non-resonant term of BaBar 2008 is not included: its Belle coupling is
#: switched off.
COMPONENTS = (
    "rho770",
    "Kstar892minus",
    "Kstarzero1430minus",
    "Kstartwo1430minus",
    "Kstar1410minus",
    "Kstar1680minus",
    "Kstar892plus",
    "Kstarzero1430plus",
    "Kstartwo1430plus",
    "Kstar1410plus",
    "Kstar1680plus",
    "omega",
    "ftwo1270",
    "rho1450",
    "Kmatrix",
)
#: Reference component: coupling fixed to 1.
REFERENCE = "rho770"
#: Components without a coupling (the K-matrix production is in its P-vector).
UNCOUPLED = ("Kmatrix",)

LATEX = {
    "rho770": R"\rho(770)",
    "Kstar892": R"K^*(892)",
    "Kstarzero1430": R"K_0^*(1430)",
    "Kstartwo1430": R"K_2^*(1430)",
    "Kstar1410": R"K^*(1410)",
    "Kstar1680": R"K^*(1680)",
    "omega": R"\omega(782)",
    "ftwo1270": R"f_2(1270)",
    "rho1450": R"\rho(1450)",
}

#: Radii of the Blatt-Weisskopf form factors (GeV^-1) of the resonances and of the D.
RADIUS_RESONANCE = 1.5
RADIUS_D = 5.0

#: K-matrix channel masses (pi, K, eta, eta') hard-coded in the Belle C++ code
#: (Resonance::amplitude_KMatrix). By default the PDG values are used instead.
CPP_KMATRIX_MASSES = (0.13957, 0.493677, 0.54775, 0.95778)

# Anisovich-Sarantsev K-matrix (fixed in the Belle fit)
KMATRIX_POLE_MASSES = (0.651, 1.2036, 1.55817, 1.21, 1.82206)
KMATRIX_POLE_COUPLINGS = (
    (0.22889, -0.55377, 0, -0.39899, -0.34639),
    (0.94128, 0.55095, 0, 0.39065, 0.31503),
    (0.36856, 0.23888, 0.55639, 0.18340, 0.18681),
    (0.33650, 0.40907, 0.85679, 0.19906, -0.00984),
    (0.18171, -0.17558, -0.79658, -0.00355, 0.22358),
)
KMATRIX_F_SCATT = (0.23399, 0.15044, -0.20545, 0.32825, 0.35412)
KMATRIX_S0_SCATT = -3.92637
KMATRIX_S0_PROD = -0.07
KMATRIX_CHANNELS = (R"\pi\pi", R"K\bar{K}", R"4\pi", R"\eta\eta", R"\eta\eta'")

#: Kinematic variables: m^2(KS pi-) and m^2(KS pi+), in GeV^2
m2_kspim, m2_kspip = sp.symbols("m2_kspim m2_kspip", positive=True)


@unevaluated(real=True)
class ZemachSpinFactor(sp.Expr):
    r"""Zemach spin factor of :math:`D \to R(\to A B) C` for :math:`J_R = 1, 2`.

    .. math::

        J = 1:&\quad s_{AC} - s_{BC} + \frac{(m_D^2 - m_C^2)(m_B^2 - m_A^2)}{s_{AB}} \\
        J = 2:&\quad \left(s_{BC} - s_{AC}
            + \frac{(m_D^2 - m_C^2)(m_A^2 - m_B^2)}{s_{AB}}\right)^2
            - \frac{1}{3}\left(s_{AB} - 2(m_D^2 + m_C^2)
            + \frac{(m_D^2 - m_C^2)^2}{s_{AB}}\right)
            \left(s_{AB} - 2(m_A^2 + m_B^2) + \frac{(m_A^2 - m_B^2)^2}{s_{AB}}\right)
    """

    sAB: Any
    sBC: Any
    sAC: Any
    mA: Any
    mB: Any
    mC: Any
    mD: Any
    J: int = 1
    _latex_repr_ = R"Z_{{{J}}}\left({sAB}, {sBC}, {sAC}\right)"

    def evaluate(self) -> sp.Expr:
        s_ab, s_bc, s_ac, ma, mb, mc, md, j = self.args
        if j == 1:
            return s_ac - s_bc + (md**2 - mc**2) * (mb**2 - ma**2) / s_ab
        if j == 2:  # ruff: ignore[magic-value-comparison]
            return (
                s_bc - s_ac + (md**2 - mc**2) * (ma**2 - mb**2) / s_ab
            ) ** 2 - sp.Rational(1, 3) * (
                s_ab - 2 * (md**2 + mc**2) + (md**2 - mc**2) ** 2 / s_ab
            ) * (s_ab - 2 * (ma**2 + mb**2) + (ma**2 - mb**2) ** 2 / s_ab)
        msg = f"Zemach spin factor for J = {j} is not implemented"
        raise NotImplementedError(msg)


@dataclass(frozen=True)
class DalitzModel:
    r"""A :math:`D^0 \to K^0_S \pi^+ \pi^-` amplitude model.

    ``components`` holds the amplitude of each component without coupling, ``couplings``
    the coupling of each component (a symbol, or 1), ``parameter_defaults`` the values
    of all parameters of the expressions and ``variables`` the kinematic variables
    :math:`s_{ab}` and :math:`s_{bc}`.
    """

    components: dict[str, sp.Expr]
    couplings: dict[str, sp.Expr]
    parameter_defaults: dict[sp.Symbol, complex | float]
    variables: tuple[sp.Symbol, sp.Symbol] = (m2_kspim, m2_kspip)
    masses: dict[str, sp.Symbol] = field(default_factory=dict)

    @property
    def amplitude(self) -> sp.Expr:
        """Coherent sum of all components (unevaluated)."""
        return sum(self.term(name) for name in self.components)

    @property
    def intensity(self) -> sp.Expr:
        r""":math:`|A|^2`."""
        return sp.Abs(self.amplitude) ** 2

    def term(self, name: str) -> sp.Expr:
        """Coupling times amplitude of one component."""
        return self.couplings[name] * self.components[name]

    def free_couplings(self) -> dict[str, sp.Symbol]:
        """Coupling symbols other than the reference and uncoupled components."""
        return {
            k: c
            for k, c in self.couplings.items()
            if isinstance(c, sp.Symbol) and k != REFERENCE
        }


def _complex(model: Mapping[str, list[float]], prefix: str) -> complex:
    return complex(model[f"{prefix}_realpart"][0], model[f"{prefix}_imaginarypart"][0])


def formulate_model(  # ruff: ignore[too-many-locals]
    belle_model: Mapping[str, list[float]] | None = None,
    final_state_masses: tuple[float, float, float] | None = None,
    kmatrix_masses: tuple[float, float, float, float] | None = None,
) -> DalitzModel:
    r"""Build the Belle 2018 model as SymPy expressions.

    Args:
        belle_model: Parameters as read by :func:`.load_belle_model` (default).

        final_state_masses: :math:`(m_\pi, m_{K^0_S}, m_{D^0})` in GeV; PDG values
            by default.

        kmatrix_masses: :math:`(m_\pi, m_K, m_\eta, m_{\eta'})` of the K-matrix
            channels in GeV; PDG values (:math:`K^+` for the :math:`K\bar{K}` channel,
            as in the Belle code) by default, or :data:`CPP_KMATRIX_MASSES`.
    """
    belle_model = load_belle_model() if belle_model is None else belle_model
    if final_state_masses is None:
        final_state_masses = (
            lp.pi_plus.mass / 1e3,
            lp.K_S_0.mass / 1e3,
            lp.D_0.mass / 1e3,
        )
    if kmatrix_masses is None:
        kmatrix_masses = (
            lp.pi_plus.mass / 1e3,
            lp.K_plus.mass / 1e3,
            lp.eta.mass / 1e3,
            lp.etap_958.mass / 1e3,
        )
    defaults: dict[sp.Symbol, Any] = {}

    def symbol(name: str, value: Any, **assumptions: bool) -> sp.Symbol:
        s = sp.Symbol(name, **assumptions)
        defaults[s] = value
        return s

    m_pi = symbol(R"m_{\pi}", final_state_masses[0], positive=True)
    m_ks = symbol(R"m_{K^0_S}", final_state_masses[1], positive=True)
    m_d = symbol(R"m_{D^0}", final_state_masses[2], positive=True)
    s_ab, s_bc = m2_kspim, m2_kspip
    s_ac = m_d**2 + m_ks**2 + 2 * m_pi**2 - s_ab - s_bc

    def mass_width(key: str) -> tuple[sp.Symbol, sp.Symbol]:
        label = LATEX[key]
        return (
            symbol(f"m_{{{label}}}", belle_model[f"{key}_Mass"][0], positive=True),
            symbol(
                Rf"\Gamma_{{{label}}}", belle_model[f"{key}_Width"][0], positive=True
            ),
        )

    d_r = symbol("d_R", RADIUS_RESONANCE, positive=True)
    d_d = symbol("d_D", RADIUS_D, positive=True)

    # Zemach spin factors (a = pi-, b = KS, c = pi+); spin-1 sign as in the Belle code
    def zemach(pair: str, j: int) -> sp.Expr:
        args = {
            "ab": (s_ab, s_bc, s_ac, m_pi, m_ks, m_pi, m_d),
            "bc": (s_bc, s_ab, s_ac, m_pi, m_ks, m_pi, m_d),
            "ac": (s_ac, s_bc, s_ab, m_pi, m_pi, m_ks, m_d),
        }[pair]
        factor = ZemachSpinFactor(*args, J=j)
        return -factor if j == 1 else factor

    # Breit-Wigner lineshapes: (s, daughters (a, b), spectator) per pair
    pairs = {
        "ab": (s_ab, m_pi, m_ks, m_pi),
        "bc": (s_bc, m_pi, m_ks, m_pi),
        "ac": (s_ac, m_pi, m_pi, m_ks),
    }

    def bw(key: str, pair: str, j: int, md0: sp.Expr | None = None) -> sp.Expr:
        m0, gamma0 = resonances[key]
        s, ma, mb, mc = pairs[pair]
        lineshape = BelleBreitWigner(
            s, m0, gamma0, ma, mb, mc, m_d, m_d if md0 is None else md0, d_r, d_d, L=j
        )
        return lineshape * zemach(pair, j)

    resonances = {
        k: mass_width(k)
        for k in (
            "rho770",
            "Kstar892",
            "Kstartwo1430",
            "Kstar1410",
            "Kstar1680",
            "omega",
            "ftwo1270",
            "rho1450",
            "Kstarzero1430",
        )
    }

    # LASS K pi S-wave
    lass_pars = [
        symbol(R"a_\mathrm{LASS}", belle_model["LASS_a"][0], real=True),
        symbol(R"r_\mathrm{LASS}", belle_model["LASS_r"][0], real=True),
        *resonances["Kstarzero1430"],
        m_ks,
        m_pi,
        symbol(R"R_\mathrm{LASS}", belle_model["LASS_R"][0], real=True),
        symbol(R"\phi_R", belle_model["LASS_phi_R"][0], real=True),
        symbol(R"F_\mathrm{LASS}", belle_model["LASS_F"][0], real=True),
        symbol(R"\phi_F", belle_model["LASS_phi_F"][0], real=True),
    ]

    # pi pi S-wave K-matrix
    km_pi, km_k, km_eta, km_etap = (
        symbol(Rf"m_{{{n}}}^\mathrm{{KM}}", v, positive=True)
        for n, v in zip((R"\pi", "K", R"\eta", R"\eta'"), kmatrix_masses, strict=True)
    )
    n = len(KMATRIX_POLE_MASSES)
    pole_masses = [
        symbol(Rf"m_{{{a + 1}}}^\mathrm{{KM}}", v, positive=True)
        for a, v in enumerate(KMATRIX_POLE_MASSES)
    ]
    g = [
        [
            symbol(
                Rf"g^{{{a + 1}}}_{{{KMATRIX_CHANNELS[i]}}}",
                KMATRIX_POLE_COUPLINGS[a][i],
                real=True,
            )
            for i in range(n)
        ]
        for a in range(n)
    ]
    f_scatt_row = [
        symbol(Rf"f_{{{KMATRIX_CHANNELS[i]}}}^\mathrm{{scatt}}", v, real=True)
        for i, v in enumerate(KMATRIX_F_SCATT)
    ]
    f_scatt = [
        [f_scatt_row[max(i, j)] if 0 in {i, j} else 0 for j in range(n)]
        for i in range(n)
    ]
    betas = [
        symbol(Rf"\beta_{{{a + 1}}}", _complex(belle_model, f"Kmatrix_beta{a + 1}"))
        for a in range(n)
    ]
    f_prod = [
        symbol(
            Rf"f_{{{KMATRIX_CHANNELS[j]}}}^\mathrm{{prod}}",
            _complex(belle_model, f"Kmatrix_f_prod_1{j + 1}"),
        )
        for j in range(n)
    ]
    kmatrix = kmatrix_amplitude(
        s_ac,
        pole_masses,
        g,
        f_scatt,
        symbol(R"s_0^\mathrm{scatt}", KMATRIX_S0_SCATT, real=True),
        betas,
        f_prod,
        symbol(R"s_0^\mathrm{prod}", KMATRIX_S0_PROD, real=True),
        [[km_pi, km_pi], [km_k, km_k], [km_pi], [km_eta, km_eta], [km_eta, km_etap]],
    )

    components = {
        "rho770": bw("rho770", "ac", 1),
        "Kstar892minus": bw("Kstar892", "ab", 1),
        "Kstarzero1430minus": LASS(s_ab, *lass_pars),
        "Kstartwo1430minus": bw("Kstartwo1430", "ab", 2),
        "Kstar1410minus": bw("Kstar1410", "ab", 1),
        "Kstar1680minus": bw("Kstar1680", "ab", 1),
        "Kstar892plus": bw("Kstar892", "bc", 1),
        "Kstarzero1430plus": LASS(s_bc, *lass_pars),
        "Kstartwo1430plus": bw("Kstartwo1430", "bc", 2),
        "Kstar1410plus": bw("Kstar1410", "bc", 1),
        "Kstar1680plus": bw("Kstar1680", "bc", 1),
        "omega": bw("omega", "ac", 1),
        "ftwo1270": bw("ftwo1270", "ac", 2),
        # q0 at the nominal mass needs m_D > m0 + m_KS: Belle uses m0 + m_KS instead
        "rho1450": bw("rho1450", "ac", 1, md0=resonances["rho1450"][0] + m_ks),
        "Kmatrix": kmatrix,
    }
    couplings: dict[str, sp.Expr] = {}
    for name in COMPONENTS:
        if name in UNCOUPLED:
            couplings[name] = sp.Integer(1)
        elif name == REFERENCE:
            couplings[name] = symbol(f"C_{{{name}}}", complex(1, 0))
        else:
            couplings[name] = symbol(f"C_{{{name}}}", _complex(belle_model, name))
    return DalitzModel(
        components=components,
        couplings=couplings,
        parameter_defaults=defaults,
        masses={"pi": m_pi, "KS": m_ks, "D": m_d},
    )
