r"""Parametric acceptance in two Dalitz-plot variables and the decay time.

The acceptance (efficiency, not normalised) of :math:`D^0 \to K^0_S \pi^+ \pi^-` events
is a logistic function of a polynomial in Dalitz-plot and time features, times a
decay-time turn-on:

.. math::

    \varepsilon(s_+, s_0, t) = \sigma(L)\,
        \frac{(t - x_0)^a}{\left(b + (t - x_0)^n\right)^m}\, e^{-\beta t}
        \quad (t > x_0), \qquad 0 \quad (t \le x_0),

with :math:`\sigma(L) = 1/(1 + e^{-L})` and :math:`L = \sum_k c_k B_k(q, r, z, u)` a sum
of monomials :math:`B_k` with coefficients :math:`c_k`. The features are

.. math::

    q = X(s_0), \quad r = X(s_+) - X(s_-), \quad z = \frac{t - \mu_t}{\sigma_t},
    \quad u = e^{-(t - t_\mathrm{min})/s},

where :math:`X(s) = (s - \bar{s}) / (\Delta s / 2)` maps the kinematic range of each
invariant mass squared to :math:`[-1, 1]`, :math:`s_\pm = m^2(K^0_S\pi^\pm)` and
:math:`s_0 = m^2(\pi^+\pi^-)`. :math:`\mu_t, \sigma_t, t_\mathrm{min}` are fixed
constants; :math:`s` and the coefficients are parameters.

:func:`formulate_acceptance` builds the expression for any set of monomials;
:data:`V4_TERMS` and :data:`V4_PARAMETERS` define the "v4" model of the d02kspipi_run3
efficiency studies. The expression is used with TensorWaves: to weight or accept-reject
events (:class:`ExpressionAcceptance`), and to fit its parameters to a simulated sample
with :func:`fit_acceptance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import sympy as sp
from tensorwaves.function.sympy import create_parametrized_function

from aman_examples.acceptance import DEFAULT_MASSES, Acceptance

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorwaves.interface import DataSample, FitResult, ParametrizedFunction

#: Data variables
m2_kspim, m2_kspip, t = sp.symbols("m2_kspim m2_kspip t", real=True)
#: Features of the logit (functions of the data variables)
q, r, z, u = sp.symbols("q r z u", real=True)

#: Monomials of the logit of the v4 model, by coefficient name
V4_TERMS: dict[str, sp.Expr] = {
    "a0": sp.Integer(1),
    "a1": q,
    "a2": q**2,
    "a3": r**2,
    "a4": q * r**2,
    "b0": r,
    "b1": q * r,
    "i0": q * z,
    "i1": r * z,
    "i2": q * z**2,
    "k0": u * q,
    "k1": u * r,
    "k2": u * q**2,
    "k3": u * r**2,
    "k4": u * q * r**2,
    "k5": u * q * r,
    "h0": q**3,
    "h1": q**4,
    "h2": r**4,
    "h3": q**2 * r**2,
    "g0": u * q**3,
    "g1": u * r**4,
    "g2": u * q**2 * r**2,
}

#: Fitted parameters of the v4 model (shape only: no absolute normalisation)
V4_PARAMETERS: dict[str, float] = {
    "a0": -1.143587486047769,
    "a1": 0.2728076662935409,
    "a2": 0.3982296963215377,
    "a3": 0.07183537924760787,
    "a4": 0.04272083312008347,
    "b0": 0.0001907117393739772,
    "b1": -0.006141499905599355,
    "x0": 0.336460815366748,
    "a": 3.211496809294131,
    "b": 0.5511458025181769,
    "beta": 0.1556456141588516,
    "n": 2.884106600488068,
    "m": 0.9155812410616199,
    "i0": 0.2001698120151701,
    "i1": -0.001003534913635097,
    "i2": -0.02619969600866154,
    "s": 1.04055176685272,
    "k0": 2.260187085343758,
    "k1": -0.01115764477448246,
    "k2": 0.7534362848566402,
    "k3": 0.0822805242485721,
    "k4": 0.4246248232242467,
    "k5": -0.03845309161008209,
    "h0": 0.0006264273823389177,
    "h1": -0.1309659440945871,
    "h2": -0.06133588556257724,
    "h3": -0.1503050630592253,
    "g0": -0.0435803977730237,
    "g1": -0.09167192161888843,
    "g2": -0.6778695160313601,
}

#: Fixed constants of the v4 model: mean and standard deviation of t in the reference
#: sample of the fit, and the lower edge of the fitted decay-time range
V4_CONSTANTS: dict[str, float] = {
    "t_mean": 1.496841417972478,
    "t_std": 0.9860725713095789,
    "t_min": 0.5,
}

#: Decay-time range in which the v4 model was fitted
V4_TIME_RANGE = (0.5, 8.0)

TIME_SHAPE_PARAMETERS = ("x0", "a", "b", "beta", "n", "m")


@dataclass(frozen=True)
class AcceptanceModel:
    r"""A parametric acceptance as a SymPy expression of ``m2_kspim``, ``m2_kspip`` and ``t``.

    ``parameter_defaults`` holds the coefficients and shape parameters (all can be
    fitted), ``constants`` the fixed :math:`\mu_t, \sigma_t, t_\mathrm{min}`.
    """

    expression: sp.Expr
    parameter_defaults: dict[sp.Symbol, float]
    constants: dict[str, float]
    terms: dict[str, sp.Expr] = field(default_factory=dict)

    def function(self, backend: str = "jax") -> ParametrizedFunction:
        """The acceptance as a TensorWaves function of a `~tensorwaves.interface.DataSample`."""
        return create_parametrized_function(
            self.expression, self.parameter_defaults, backend
        )

    def acceptance(self, backend: str = "jax") -> ExpressionAcceptance:
        """The acceptance as an :class:`~aman_examples.acceptance.Acceptance`, e.g. for :func:`~aman_examples.acceptance.apply_efficiency`."""
        return ExpressionAcceptance(self.function(backend))


def normalised(s: sp.Expr, s_min: float, s_max: float) -> sp.Expr:
    """Linear map of [``s_min``, ``s_max``] to [-1, 1]."""
    return (s - (s_min + s_max) / 2) / ((s_max - s_min) / 2)


def formulate_acceptance(  # ruff: ignore[too-many-locals]
    terms: Mapping[str, sp.Expr] | None = None,
    parameters: Mapping[str, float] | None = None,
    constants: Mapping[str, float] | None = None,
    masses: tuple[float, float, float] = DEFAULT_MASSES,
) -> AcceptanceModel:
    r"""Build a parametric acceptance (see the module docstring).

    Args:
        terms: Coefficient name and monomial in :data:`q`, :data:`r`, :data:`z` and
            :data:`u` for each term of the logit; :data:`V4_TERMS` by default.

        parameters: Values of the coefficients, of ``s`` (the time scale of :math:`u`)
            and of the turn-on parameters ``x0, a, b, beta, n, m``. Missing coefficients
            are 0; by default the values of :data:`V4_PARAMETERS` are used where they
            exist.

        constants: ``t_mean``, ``t_std`` and ``t_min``; :data:`V4_CONSTANTS` by default.

        masses: :math:`(m_\pi, m_{K^0_S}, m_{D^0})` in GeV, for the kinematic ranges.
    """
    terms = dict(V4_TERMS if terms is None else terms)
    values = dict(V4_PARAMETERS if parameters is None else parameters)
    constants = dict(V4_CONSTANTS if constants is None else constants)
    m_pi, m_ks, m_d = masses

    symbols = {
        name: sp.Symbol(name, real=True)
        for name in [*terms, "s", *TIME_SHAPE_PARAMETERS]
    }
    missing = [n for n in ["s", *TIME_SHAPE_PARAMETERS] if n not in values]
    if missing:
        msg = f"No values for the parameters {missing}"
        raise ValueError(msg)
    defaults = {symbols[n]: float(values.get(n, 0.0)) for n in symbols}

    # features
    s_kpi = ((m_ks + m_pi) ** 2, (m_d - m_pi) ** 2)
    s_pipi = ((2 * m_pi) ** 2, (m_d - m_ks) ** 2)
    m2_pipi = m_d**2 + m_ks**2 + 2 * m_pi**2 - m2_kspim - m2_kspip
    features = {
        q: normalised(m2_pipi, *s_pipi),
        r: normalised(m2_kspip, *s_kpi) - normalised(m2_kspim, *s_kpi),
        z: (t - constants["t_mean"]) / constants["t_std"],
        u: sp.exp(-(t - constants["t_min"]) / symbols["s"]),
    }
    logit = sum(symbols[name] * monomial for name, monomial in terms.items())
    logistic = 1 / (1 + sp.exp(-logit.xreplace(features)))

    x0, a, b, beta, n, m = (symbols[p] for p in TIME_SHAPE_PARAMETERS)
    dt = t - x0
    # dt is replaced by 1 where t <= x0, so that the unused branch stays finite
    safe = sp.Piecewise((dt, dt > 0), (1, True))
    turn_on = sp.Piecewise(
        (safe**a / (b + safe**n) ** m * sp.exp(-beta * t), dt > 0), (0, True)
    )
    return AcceptanceModel(logistic * turn_on, defaults, constants, terms)


class ExpressionAcceptance(Acceptance):
    """An :class:`~aman_examples.acceptance.Acceptance` from a TensorWaves function of ``m2_kspim``, ``m2_kspip`` and ``t``."""

    def __init__(self, function: ParametrizedFunction) -> None:
        self.function = function

    def __call__(self, sample: DataSample) -> np.ndarray:
        data = {
            k: np.asarray(sample[k], dtype=float) for k in ("m2_kspim", "m2_kspip", "t")
        }
        return np.asarray(self.function(data), dtype=float)

    def evaluate(self, variables: dict[str, np.ndarray]) -> np.ndarray:
        return self({k: variables[k] for k in ("m2_kspim", "m2_kspip", "t")})


def fit_acceptance(
    model: AcceptanceModel,
    sample: DataSample,
    reference: DataSample,
    *,
    free: list[str] | None = None,
    initial: Mapping[str, float] | None = None,
    limits: Mapping[str, tuple[float | None, float | None]] | None = None,
    backend: str = "jax",
) -> FitResult:
    r"""Fit acceptance parameters to an accepted (simulated) sample.

    Maximises :math:`\sum_i \ln\varepsilon(x_i) - N
    \ln\langle\varepsilon\rangle_\mathrm{ref}` (`~tensorwaves.estimator.UnbinnedNLL`),
    where ``reference`` is a sample of the distribution before the acceptance (e.g.
    uniform in the Dalitz plot, exponential in :math:`t` in the fitted range). The
    fitted acceptance is relative to that reference.

    Args:
        model: The acceptance model.

        sample: Accepted events.

        reference: Events before the acceptance.

        free: Names of the parameters to fit, all by default; the others are fixed to
            the values in ``model.parameter_defaults``.

        initial: Starting values (by name); the model defaults otherwise.

        limits: Bounds (lower, upper; ``None`` for no bound) of free parameters, e.g.
            ``{"x0": (None, 0.499)}`` to keep the turn-on below the smallest decay time.

        backend: Computational backend.
    """
    from tensorwaves.estimator import UnbinnedNLL  # ruff: ignore[import-outside-top-level]
    from tensorwaves.optimizer import Minuit2  # ruff: ignore[import-outside-top-level]

    function = model.function(backend)
    keys = ("m2_kspim", "m2_kspip", "t")
    data = {k: np.asarray(sample[k], dtype=float) for k in keys}
    phsp = {k: np.asarray(reference[k], dtype=float) for k in keys}
    names = [s.name for s in model.parameter_defaults] if free is None else list(free)
    start: dict[str, Any] = {name: function.parameters[name] for name in names}
    start.update(dict(initial or {}))
    estimator = UnbinnedNLL(function, data, phsp, backend=backend)

    def set_limits(minuit: Any) -> None:
        for name, (low, high) in (limits or {}).items():
            minuit.limits[name] = (
                -np.inf if low is None else low,
                np.inf if high is None else high,
            )

    return Minuit2(minuit_modifier=set_limits).optimize(estimator, start)
