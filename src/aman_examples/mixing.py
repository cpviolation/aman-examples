r"""Time-dependent decay rates with :math:`D^0`-:math:`\bar{D}^0` mixing.

Conventions (as in HFLAV and in :mod:`amplitf.mixing`): the mass eigenstates are
:math:`|D_{1,2}\rangle = p|D^0\rangle \pm q|\bar{D}^0\rangle`, with :math:`x = (m_1 -
m_2)/\Gamma`, :math:`y = (\Gamma_1 - \Gamma_2)/(2\Gamma)` and :math:`\Gamma = 1/\tau`.
For a :math:`D^0` produced at :math:`t = 0` and decaying to :math:`f` at time :math:`t`,

.. math::

    |A(t)|^2 = |A_1|^2 \psi_+(t) + |A_2|^2 \psi_-(t)
        + 2\,\mathrm{Re}\left(A_1 A_2^* \psi_i(t)\right),\quad
    A_{1,2} = \frac{1}{2}\left(A_f \pm \frac{q}{p}\bar{A}_f\right),

with :math:`\psi_\pm = e^{-(1 \pm y)t/\tau}` and :math:`\psi_i = e^{-(1 + ix)t/\tau}`.
For a :math:`\bar{D}^0`, :math:`A_f \leftrightarrow \bar{A}_f` and :math:`q/p \to p/q`.

When decay times are sampled from :math:`e^{-t/\tau}` (as by
:class:`.DalitzDecayTimeGenerator`), toy generation and normalisation integrals need the
rate *relative* to that exponential (``relative_to_decay=True``): :math:`e^{\mp y
t/\tau}` and :math:`e^{-ixt/\tau}`. On data, the relative and full rates differ by
:math:`e^{-t/\tau}`, which does not depend on :math:`x`, :math:`y` or :math:`q/p`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from tensorwaves.data.transform import SympyDataTransformer

from aman_examples.d02kspipi import DalitzModel, m2_kspim, m2_kspip

#: Decay time (data variable), in units of the lifetime by default
t = sp.Symbol("t", nonnegative=True)
x_mix, y_mix = sp.symbols("x y", real=True)
qop_abs = sp.Symbol(R"\left|q/p\right|", positive=True)
qop_phase = sp.Symbol(R"\phi_{q/p}", real=True)
tau = sp.Symbol(R"\tau", positive=True)


def time_evolution(
    t: sp.Expr, x: sp.Expr, y: sp.Expr, tau: sp.Expr, *, relative_to_decay: bool = True
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    r""":math:`\psi_+, \psi_-, \psi_i` (divided by :math:`e^{-t/\tau}` if ``relative_to_decay``)."""
    one = 0 if relative_to_decay else 1
    return (
        sp.exp(-(one + y) * t / tau),
        sp.exp(-(one - y) * t / tau),
        sp.exp(-(one + sp.I * x) * t / tau),
    )


def mixing_intensity(
    amplitude: sp.Expr,
    amplitude_bar: sp.Expr,
    q_over_p: sp.Expr,
    psi: tuple[sp.Expr, sp.Expr, sp.Expr],
    *,
    initial_d0: bool = True,
) -> sp.Expr:
    r"""Time-dependent rate of an initial :math:`D^0` (or :math:`\bar{D}^0`).

    Args:
        amplitude: :math:`A_f`, the :math:`D^0 \to f` amplitude.

        amplitude_bar: :math:`\bar{A}_f`, the :math:`\bar{D}^0 \to f` amplitude.

        q_over_p: :math:`q/p`.

        psi: :math:`\psi_+, \psi_-, \psi_i` from :func:`time_evolution`.

        initial_d0: :math:`D^0` (True) or :math:`\bar{D}^0` (False) at :math:`t = 0`.
    """
    if initial_d0:
        direct, mixed = amplitude, q_over_p * amplitude_bar
    else:
        direct, mixed = amplitude_bar, amplitude / q_over_p
    a1, a2 = (direct + mixed) / 2, (direct - mixed) / 2
    psi_p, psi_m, psi_i = psi
    # |a|^2 as Abs(a)**2 of a plain sum: SymPy would expand Abs(conjugate(a))**2 into
    # a * conjugate(a), which evaluates to a complex number
    return (
        sp.Abs(a1) ** 2 * psi_p
        + sp.Abs(a2) ** 2 * psi_m
        + 2 * sp.re(a1 * sp.conjugate(a2) * psi_i)
    )


def conjugate_amplitude(model: DalitzModel) -> sp.Expr:
    r""":math:`\bar{A}_f(s_{ab}, s_{bc}) = A_f(s_{bc}, s_{ab})` for :math:`D^0 \to K^0_S \pi^+ \pi^-` without direct CP violation."""
    return model.amplitude.xreplace({m2_kspim: m2_kspip, m2_kspip: m2_kspim})


#: The D0 and D0bar amplitudes, as data variables of the mixing rate
amplitude_symbol = sp.Symbol("A_f")
amplitude_bar_symbol = sp.Symbol(R"\bar{A}_f")


@dataclass(frozen=True)
class MixingModel:
    r"""Time-dependent rate with mixing, as a function of :math:`A_f`, :math:`\bar{A}_f` and :math:`t`.

    The amplitudes are data variables, computed once per event with
    :func:`amplitude_transformer`: the rate itself is then a small expression and only
    depends on the mixing parameters :math:`x`, :math:`y`, :math:`|q/p|`,
    :math:`\phi_{q/p}` and :math:`\tau` (``parameter_defaults``).
    """

    intensity: sp.Expr
    parameter_defaults: dict[sp.Symbol, float]


def formulate_mixing_model(
    x: float = 0.004,
    y: float = 0.0064,
    q_over_p: float = 1.0,
    phase: float = 0.0,
    *,
    initial_d0: bool = True,
    relative_to_decay: bool = True,
) -> MixingModel:
    r"""Time-dependent rate of :math:`D^0 \to f` with mixing.

    The decay time ``t`` is in units of :math:`\tau` (1 by default). The defaults of
    :math:`x`, :math:`y` are those of the toys in tfa_examples
    (``macros/d02kspipi_toys.py``).
    """
    psi = time_evolution(t, x_mix, y_mix, tau, relative_to_decay=relative_to_decay)
    q_p = qop_abs * sp.exp(sp.I * qop_phase)
    intensity = mixing_intensity(
        amplitude_symbol, amplitude_bar_symbol, q_p, psi, initial_d0=initial_d0
    )
    defaults = {x_mix: x, y_mix: y, qop_abs: q_over_p, qop_phase: phase, tau: 1.0}
    return MixingModel(intensity, defaults)


def amplitude_transformer(
    model: DalitzModel, backend: str = "jax"
) -> SympyDataTransformer:
    r"""Data transformer from (``m2_kspim``, ``m2_kspip``, ``t``) to the variables of :class:`MixingModel`: :math:`A_f`, :math:`\bar{A}_f` (with the parameters of ``model`` fixed to their defaults) and ``t``."""
    defaults = model.parameter_defaults
    return SympyDataTransformer.from_sympy(
        {
            amplitude_symbol: model.amplitude.doit().xreplace(defaults),
            amplitude_bar_symbol: conjugate_amplitude(model).doit().xreplace(defaults),
            t: t,
        },
        backend,
    )
