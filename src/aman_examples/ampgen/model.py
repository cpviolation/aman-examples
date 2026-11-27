"""Amplitude models defined with AmpGen options files, as SymPy expressions.

Ported from ``tfaex.ampgen`` of cpviolation/tfa_examples. The conventions follow AmpGen
(https://github.com/GooFit/AmpGen):

- daughters are ordered as in ``Particle::operator<`` (higher spin first, unstable
  before stable, heavier first, then PDG id and charge; antiparticles are ordered as
  their conjugates);
- the default orbital angular momentum is the lowest allowed one, with parity
  conservation for strong decays (all but the head);
- narrow particles (width < 1 keV, e.g. the head) get the ``FormFactor`` lineshape,
  other resonances ``BW`` unless a lineshape is given in brackets;
- pseudo-particles with a single daughter (e.g. ``rhoOmega00``) have no lineshape; their
  decays are expanded with their own couplings.

The data variables are the four-momenta of the final-state particles, in the order of
the event type: ``p<i>_x``, ``p<i>_y``, ``p<i>_z``, ``p<i>_E`` for particle ``i``. Spin
factors are covariant tensors (:mod:`.covariant`), expanded in these components. Only
the lineshapes and spin factors validated against AmpGen are supported; anything else
raises `NotImplementedError`.
"""

from __future__ import annotations

import functools
import itertools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sympy as sp

from aman_examples.ampgen import covariant as cov
from aman_examples.ampgen import lineshapes as ls
from aman_examples.ampgen.options import (
    Decay,
    load_particles,
    parse_descriptor,
    read_options,
)
from aman_examples.dynamics import AbsSquared

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

STABLE_THRESHOLD = 1e-6  # GeV, AmpGen Particle::StableThreshold (1 keV)
SPIN_LETTER = {0: "S", 2: "V", 4: "T"}
ORBITALS = ("S", "P", "D", "F")
TWO_BODY = 2
COMPONENTS = ("x", "y", "z", "E")


class Symbols:
    """Parameter symbols of a model, created on first use with their default values."""

    def __init__(self, table: dict, params: dict) -> None:
        self.table, self.params = table, params
        self.defaults: dict[sp.Symbol, Any] = {}

    def get(
        self, name: str, default: Any, latex: str | None = None, **assumptions: bool
    ) -> sp.Symbol:
        symbol = sp.Symbol(latex or name, **assumptions)
        if symbol not in self.defaults:
            self.defaults[symbol] = self.params.get(name, default)
        return symbol

    def particle(self, name: str, kind: str) -> sp.Symbol:
        """Mass, width or radius of a particle (``<name>_<kind>`` in the options file)."""
        latex = {"mass": "m", "width": R"\Gamma", "radius": "r"}[kind]
        return self.get(
            f"{name}_{kind}",
            self.table[name][kind],
            f"{latex}_{{{name}}}",
            positive=True,
        )


class Particle:
    """A node of an expanded decay chain, with AmpGen's ordering and orbital rules."""

    def __init__(
        self,
        decay,
        table,
        params,
        final_state,
        *,
        is_head=False,
        splines=None,
        slots=None,
    ):
        self.name = decay.name
        self.modifiers = list(decay.modifiers)
        if self.name not in table:
            msg = f"No properties for particle '{self.name}'"
            raise KeyError(msg)
        props = table[self.name]
        self.two_spin, self.parity = props["two_spin"], props["parity"]
        self.pdg_id, self.charge = props["pdg_id"], props["charge"]
        self.mass = params.get(self.name + "_mass", props["mass"])
        self.width = params.get(self.name + "_width", props["width"])
        self.radius = params.get(self.name + "_radius", props["radius"])
        self.is_head = is_head
        self.table, self.params, self.splines = table, params, splines or {}
        if slots is None:  # event positions of each final-state particle, in order
            slots = {
                n: [i for i, f in enumerate(final_state) if f == n] for n in final_state
            }
        self.daughters = [
            Particle(d, table, params, final_state, splines=splines, slots=slots)
            for d in decay.daughters
        ]
        self.index = slots[self.name].pop(0) if not self.daughters else None
        self._sort_daughters()
        self.L = self._orbital()

    # --- AmpGen ordering (Particle::operator< and sortDaughters)
    def _key(self, conj):
        pdg = -self.pdg_id if conj else self.pdg_id
        charge = -self.charge if conj else self.charge
        return self, pdg, charge

    def _sort_daughters(self):
        conj = self.pdg_id < 0

        def less(a, b):  # ruff: ignore[too-many-return-statements]
            (pa, ida, qa), (pb, idb, qb) = a._key(conj), b._key(conj)  # ruff: ignore[private-member-access]
            if pa.two_spin != pb.two_spin:
                return pa.two_spin > pb.two_spin
            if pa.daughters and not pb.daughters:
                return True
            if not pa.daughters and pb.daughters:
                return False
            if pa.mass != pb.mass:
                return pa.mass > pb.mass
            if abs(ida) == abs(idb) and ida != idb:
                return ida > idb
            if qa != qb:
                return qa > qb
            if ida != idb:
                return ida > idb
            return (pa.index or 0) < (pb.index or 0)

        def cmp(a, b):
            return -1 if less(a, b) else (1 if less(b, a) else 0)

        self.daughters = sorted(self.daughters, key=functools.cmp_to_key(cmp))

    # --- orbital angular momentum
    def _orbital(self):
        for m in self.modifiers:
            if m in ORBITALS:
                return ORBITALS.index(m)
        if len(self.daughters) != TWO_BODY:
            return 0
        j, j1, j2 = (
            self.two_spin,
            self.daughters[0].two_spin,
            self.daughters[1].two_spin,
        )
        smin, smax = abs(j1 - j2), j1 + j2
        lmin = min(abs(j - s) for s in range(smin, smax + 1, 2)) // 2
        lmax = (j + smax) // 2
        for L in range(lmin, lmax + 1):
            if (
                self.is_head
                or self.parity
                == self.daughters[0].parity * self.daughters[1].parity * (-1) ** L
            ):
                return L
        msg = f"No allowed orbital angular momentum for {self.name}"
        raise ValueError(msg)

    @property
    def lineshape_name(self):
        named = [m for m in self.modifiers if m not in ORBITALS]
        if named:
            return named[0]
        return "FormFactor" if self.width < STABLE_THRESHOLD else "BW"

    def descriptor(self):
        """AmpGen's unique string: orbital label only if not the default, then the lineshape."""
        mods = (
            ["SPDF"[self.L]]
            if len(self.daughters) == TWO_BODY and self._default_orbital() != self.L
            else []
        )
        mods += [m for m in self.modifiers if m not in ORBITALS]
        m = f"[{';'.join(mods)}]" if mods else ""
        ds = (
            "{" + ",".join(d.descriptor() for d in self.daughters) + "}"
            if self.daughters
            else ""
        )
        return self.name + m + ds

    def _default_orbital(self):
        saved = self.modifiers
        self.modifiers = [m for m in saved if m not in ORBITALS]
        L = self._orbital()
        self.modifiers = saved
        return L

    # --- SymPy expressions
    def momentum(self, p: list[cov.Vector]) -> cov.Vector:
        if not self.daughters:
            return p[self.index]
        total = self.daughters[0].momentum(p)
        for d in self.daughters[1:]:
            total = cov.add(total, d.momentum(p))
        return total

    def mass_squared(self, p: list[cov.Vector], symbols: Symbols) -> sp.Expr:
        if not self.daughters:
            return symbols.particle(self.name, "mass") ** 2
        momentum = self.momentum(p)
        return cov.dot(momentum, momentum)

    def lineshape(self, p: list[cov.Vector], symbols: Symbols) -> sp.Expr:  # ruff: ignore[too-many-locals]
        if not self.daughters:
            return sp.Integer(1)
        if len(self.daughters) == 1:
            return self.daughters[0].lineshape(p, symbols)
        s = self.mass_squared(p, symbols)
        s1, s2 = (d.mass_squared(p, symbols) for d in self.daughters)
        name, _, modifier = self.lineshape_name.partition(".")
        mass, width, radius = (
            symbols.particle(self.name, k) for k in ("mass", "width", "radius")
        )
        if name == "FormFactor" and not modifier:
            own = ls.FormFactor(s, s1, s2, radius, self.L, signed=True)
        elif name == "BW" and not modifier:
            own = ls.AmpGenBreitWigner(s, s1, s2, mass, width, radius, self.L)
        elif name == "GounarisSakurai" and not modifier:
            mpi = symbols.particle("pi+", "mass")
            own = ls.AmpGenGounarisSakurai(s, s1, s2, mass, width, radius, mpi, self.L)
        elif (
            name == "Flatte"
            and self.name == "a(0)(980)0"
            and modifier in {"", "kFactor"}
        ):
            own = ls.Flatte(
                s,
                mass,
                symbols.get("Flatte::g2pieta", 0.175, R"g^2_{\pi\eta}", positive=True),
                symbols.get(
                    "Flatte::g2KK_by_g2pieta",
                    1.20,
                    R"g^2_{KK}/g^2_{\pi\eta}",
                    positive=True,
                ),
                *(symbols.particle(n, "mass") for n in ("pi+", "eta0", "K+")),
                kfactor=modifier == "kFactor",
            )
        elif name == "FOCUS" and modifier in {"Kpi", "KEta", "I32"}:
            masses = (symbols.particle(n, "mass") for n in ("K+", "pi+", "eta'(958)0"))
            own = ls.FocusKpi(s, *masses, component=modifier)
        elif name == "kMatrix" and re.fullmatch(
            r"(pole|poleKK|prod|prodKK)\.\d", modifier
        ):
            term, index = modifier.split(".")
            own = ls.kmatrix_ampgen(
                s, term, int(index), **self.kmatrix_parameters(symbols)
            )
        elif (
            name == "GSpline" and modifier in {"", "EFF"} and self.name in self.splines
        ):
            n, xmin, xmax = self.splines[self.name]
            values = tuple(
                self.params[f"{self.name}::Spline::Gamma::{i}"] for i in range(n)
            )
            own = ls.GSpline(
                s,
                s1,
                s2,
                mass,
                width,
                radius,
                self.L,
                knots=(values, xmin, xmax),
                form_factor="EFF" if modifier == "EFF" else "BlattWeisskopf",
            )
        else:
            msg = f"Lineshape '{self.lineshape_name}' of {self.name} is not implemented"
            raise NotImplementedError(msg)
        return (
            own
            * self.daughters[0].lineshape(p, symbols)
            * self.daughters[1].lineshape(p, symbols)
        )

    def kmatrix_parameters(self, symbols: Symbols) -> dict[str, Any]:
        """Parameters of AmpGen's ``kMatrix`` lineshape for this (pseudo-)particle."""
        poles = range(1, 6)
        channels = ls.KMATRIX_CHANNELS
        return {
            "pole_masses": [
                symbols.get(f"IS_p{n}_mass", None, positive=True) for n in poles
            ],
            "pole_couplings": [
                [symbols.get(f"IS_p{n}_{ch}", None, real=True) for ch in channels]
                for n in poles
            ],
            "f_scatt": [
                symbols.get(f"f_scatt{i}", None, real=True)
                for i in range(len(channels))
            ],
            "s0_scatt": symbols.get("s0_scatt", -3.92637, real=True),
            "sA0": symbols.get("sA0", -0.15, real=True),
            "sA": symbols.get("sA", 1.0, real=True),
            "s0_prod": symbols.get(f"{self.name}_s0_prod", -0.07, real=True),
        }

    def spin_tensor(self, p: list[cov.Vector]) -> Any:
        if not self.daughters:
            return sp.Integer(1)
        if len(self.daughters) == 1:
            return self.daughters[0].spin_tensor(p)
        d0, d1 = self.daughters
        name = f"{SPIN_LETTER[self.two_spin]}_{SPIN_LETTER[d0.two_spin]}{SPIN_LETTER[d1.two_spin]}_{'SPDF'[self.L]}"
        q = cov.add(d0.momentum(p), d1.momentum(p), -1)
        return cov.vertex(
            name, self.momentum(p), q, d0.spin_tensor(p), d1.spin_tensor(p)
        )

    def amplitude(self, p: list[cov.Vector], symbols: Symbols) -> sp.Expr:
        """Lineshapes times spin factor, without couplings (AmpGen ``MatrixElement``)."""
        return self.lineshape(p, symbols) * self.spin_tensor(p)


@dataclass(frozen=True)
class Chain:
    """A fully expanded decay chain and the coupling rules whose product multiplies it."""

    particle: Particle
    rules: list[str]

    @property
    def name(self) -> str:
        return self.particle.descriptor()


@dataclass(frozen=True)
class AmpGenModel:
    """Coherent sum of the decay chains of an AmpGen options file.

    ``chains`` maps each chain name (AmpGen's descriptor) to its amplitude without
    coupling, ``couplings`` to the product of its coupling symbols;
    ``parameter_defaults`` holds the values of all parameters, ``free_couplings`` the
    coupling symbols that are free in the options file, ``momenta`` the data symbols of
    the final-state four-momenta and ``trees`` the decay tree of each chain.
    """

    chains: dict[str, sp.Expr]
    couplings: dict[str, sp.Expr]
    parameter_defaults: dict[sp.Symbol, Any]
    free_couplings: list[sp.Symbol]
    momenta: list[list[sp.Symbol]]
    head: str
    final_state: list[str]
    table: dict = field(repr=False)
    trees: dict[str, Particle] = field(default_factory=dict, repr=False)

    @property
    def amplitude(self) -> sp.Expr:
        return sum(self.couplings[name] * expr for name, expr in self.chains.items())

    @property
    def intensity(self) -> sp.Expr:
        """:math:`|A|^2` of the unfolded amplitude (:class:`.AbsSquared`: ``sp.Abs(A)**2`` takes minutes to unfold and lambdify for this model)."""
        return AbsSquared(self.amplitude.doit())

    def linear_amplitude(
        self,
        free: Sequence[sp.Symbol],
        parameters: Mapping[sp.Symbol, Any] | None = None,
    ) -> tuple[dict[sp.Symbol, sp.Expr], sp.Expr]:
        r"""The amplitude as :math:`A = \sum_k c_k B_k + B_0`, for :class:`.LinearAmplitudeNLL`.

        Returns the basis amplitudes :math:`B_k = \partial A / \partial c_k` for each of
        the ``free`` couplings :math:`c_k`, and the remainder :math:`B_0`. All other
        parameters are fixed to ``parameters`` (default: ``parameter_defaults``).
        Building them chain by chain is much faster than differentiating
        :attr:`amplitude`. Raises `ValueError` if the amplitude is not linear in the
        free couplings, e.g. when two couplings of a product are free.
        """
        values = dict(self.parameter_defaults if parameters is None else parameters)
        fixed = {s: v for s, v in values.items() if s not in set(free)}
        basis: dict[sp.Symbol, sp.Expr] = dict.fromkeys(free, sp.S.Zero)
        constant = sp.S.Zero
        for name, chain in self.chains.items():
            coupling = self.couplings[name]
            used = [c for c in free if c in coupling.free_symbols]
            expression = chain.doit().xreplace(fixed)
            if not used:
                constant += coupling.xreplace(fixed) * expression
            for c in used:
                derivative = sp.diff(coupling, c)
                if derivative.free_symbols & set(free):
                    msg = (
                        f"The amplitude is not linear in the free couplings: {coupling}"
                    )
                    raise ValueError(msg)
                basis[c] += derivative.xreplace(fixed) * expression
        return basis, constant

    def data_from_momenta(self, momenta: Any) -> dict[str, Any]:
        """Data sample from an array of shape ``(N, 4 n)``: (x, y, z, E) of each particle."""
        return {
            symbol.name: momenta[:, 4 * i + j]
            for i, row in enumerate(self.momenta)
            for j, symbol in enumerate(row)
        }


def coupling_symbol(rule: str) -> sp.Symbol:
    """Symbol of the coupling of an options-file rule."""
    return sp.Symbol(f"C_{{{rule}}}")


def _expand(
    decay: Decay, rules_for: Any, final_state: list[str]
) -> list[tuple[Decay, list[str]]]:
    """Expand particles without explicit daughters using their own decay rules."""
    if not decay.daughters:
        rules = rules_for(decay.name) if decay.name not in final_state else []
        if not rules:
            return [(decay, [])]
        out = []
        for rule in rules:
            sub = parse_descriptor(rule)
            sub.modifiers = decay.modifiers + sub.modifiers
            out.extend(
                (expanded, [rule, *used])
                for expanded, used in _expand(sub, rules_for, final_state)
            )
        return out
    combos = [(Decay(decay.name, list(decay.modifiers)), [])]
    for d in decay.daughters:
        combos = [
            (
                Decay(parent.name, parent.modifiers, [*parent.daughters, dexp]),
                used + dused,
            )
            for parent, used in combos
            for dexp, dused in _expand(d, rules_for, final_state)
        ]
    return combos


def _identical_orderings(final_state: list[str], p: list) -> list[list]:
    """The momenta ``p`` permuted in all ways that exchange identical particles."""
    return [
        [p[j] for j in permutation]
        for permutation in itertools.permutations(range(len(final_state)))
        if all(final_state[i] == final_state[j] for i, j in enumerate(permutation))
    ]


def formulate_ampgen_model(  # ruff: ignore[too-many-locals]
    options_file: str | Path,
    particles_file: str | Path,
    event_type: Sequence[str] | None = None,
) -> AmpGenModel:
    """Build the SymPy model of an AmpGen options file.

    As in AmpGen, each chain is symmetrised over identical final-state particles: the
    sum over their exchanges, divided by the square root of the number of exchanges.

    Args:
        options_file: The AmpGen options file.

        particles_file: Particle properties (JSON, as dumped from AmpGen's particle
            table).

        event_type: Head and final-state particles, if not given by the ``EventType``
            of the options file.
    """
    options = read_options(options_file)
    table = load_particles(particles_file)
    head, *final_state = event_type or options.event_type
    params = {k: v for k, (v, _) in options.parameters.items()}
    symbols = Symbols(table, params)
    momenta = [
        [sp.Symbol(f"p{i}_{c}", real=True) for c in COMPONENTS]
        for i in range(len(final_state))
    ]
    p = [list(row) for row in momenta]
    orderings = _identical_orderings(final_state, p)

    descriptors = {rule: parse_descriptor(rule) for rule in options.couplings}

    @functools.cache
    def rules_for(name: str) -> list[str]:
        return [r for r, d in descriptors.items() if d.name == name]

    chains: dict[str, sp.Expr] = {}
    couplings: dict[str, sp.Expr] = {}
    trees: dict[str, Particle] = {}
    for rule, decay in descriptors.items():
        if decay.name != head:
            continue
        for expanded, used in _expand(decay, rules_for, final_state):
            particle = Particle(
                expanded,
                table,
                params,
                final_state,
                is_head=True,
                splines=options.splines,
            )
            name = particle.descriptor()
            if name in chains:
                msg = f"Duplicate decay chain {name}"
                raise ValueError(msg)
            chains[name] = sp.Add(
                *(particle.amplitude(q, symbols) for q in orderings)
            ) / sp.sqrt(len(orderings))
            trees[name] = particle
            couplings[name] = sp.Mul(*(coupling_symbol(r) for r in [rule, *used]))
    if not chains:
        msg = f"No decays of {head} in {options_file}"
        raise ValueError(msg)
    defaults = dict(symbols.defaults)
    for rule, (value, _) in options.couplings.items():
        defaults[coupling_symbol(rule)] = value
    free = [
        coupling_symbol(r) for r, (_, is_free) in options.couplings.items() if is_free
    ]
    return AmpGenModel(
        chains, couplings, defaults, free, momenta, head, final_state, table, trees
    )
