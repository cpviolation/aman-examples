"""Mixing rate, decay-time sampling and a fit of x and y to toys."""

from __future__ import annotations

import numpy as np
import pytest
from tensorwaves.data import IntensityDistributionGenerator
from tensorwaves.data.rng import NumpyUniformRNG
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.optimizer import Minuit2

from aman_examples.d02kspipi import formulate_model
from aman_examples.mixing import amplitude_transformer, formulate_mixing_model
from aman_examples.phasespace import DalitzDecayTimeGenerator

PARS = {"x": 0.3, "y": 0.2, R"\left|q/p\right|": 0.9, R"\phi_{q/p}": 0.4}


def evaluate(relative_to_decay: bool, initial_d0: bool, data):
    model = formulate_mixing_model(
        relative_to_decay=relative_to_decay, initial_d0=initial_d0
    )
    function = create_parametrized_function(
        model.intensity.doit(), model.parameter_defaults, "numpy"
    )
    function.update_parameters(PARS)
    return function(data)


@pytest.fixture(scope="module")
def random_amplitudes():
    rng = np.random.default_rng(1)
    n = 100
    return {
        "A_f": rng.normal(size=n) + 1j * rng.normal(size=n),
        R"\bar{A}_f": rng.normal(size=n) + 1j * rng.normal(size=n),
        "t": rng.exponential(size=n),
    }


@pytest.mark.parametrize("initial_d0", [True, False])
def test_rate_from_mass_eigenstates(random_amplitudes, initial_d0):
    """Rate from the evolution of the mass eigenstates, A1 exp(-i m1 t - G1 t/2) + ...."""
    a, a_bar, t = random_amplitudes.values()
    x, y, qp = (
        PARS["x"],
        PARS["y"],
        PARS[R"\left|q/p\right|"] * np.exp(1j * PARS[R"\phi_{q/p}"]),
    )
    direct, mixed = (a, qp * a_bar) if initial_d0 else (a_bar, a / qp)
    a1, a2 = (direct + mixed) / 2, (direct - mixed) / 2
    # tau = 1: Gamma_{1,2} = 1 +- y, m1 - m2 = x
    amplitude_t = a1 * np.exp(-(1 + y) * t / 2 - 1j * x * t / 2) + a2 * np.exp(
        -(1 - y) * t / 2 + 1j * x * t / 2
    )
    expected = np.abs(amplitude_t) ** 2
    full = evaluate(
        relative_to_decay=False, initial_d0=initial_d0, data=random_amplitudes
    )
    np.testing.assert_allclose(full, expected, rtol=1e-12)
    relative = evaluate(
        relative_to_decay=True, initial_d0=initial_d0, data=random_amplitudes
    )
    np.testing.assert_allclose(relative * np.exp(-t), full, rtol=1e-12)


@pytest.fixture(scope="module")
def setup():
    model = formulate_model()
    masses = [
        model.parameter_defaults[model.masses[k]] for k in ("pi", "KS", "pi", "D")
    ]
    mixing = formulate_mixing_model(x=0.05, y=0.05)  # exaggerated for a precise check
    function = create_parametrized_function(
        mixing.intensity.doit(), mixing.parameter_defaults, "jax"
    )
    return DalitzDecayTimeGenerator(masses), amplitude_transformer(model), function


def test_no_mixing_is_exponential(setup):
    generator, transformer, function = setup
    function.update_parameters({"x": 0.0, "y": 0.0})
    data = IntensityDistributionGenerator(
        generator, function, domain_transformer=transformer, bunch_size=200_000
    ).generate(50_000, NumpyUniformRNG(seed=3))
    assert data["t"].mean() == pytest.approx(1.0, abs=0.02)
    assert data["t"].std() == pytest.approx(1.0, abs=0.03)


def test_fit_recovers_mixing_parameters(setup):
    generator, transformer, function = setup
    true = {"x": 0.05, "y": 0.05}
    function.update_parameters(true)
    rng = NumpyUniformRNG(seed=4)
    data = IntensityDistributionGenerator(
        generator, function, domain_transformer=transformer, bunch_size=200_000
    ).generate(50_000, rng)
    phsp = generator.generate(1_000_000, rng)
    estimator = UnbinnedNLL(
        function, transformer(data), transformer(phsp), backend="jax"
    )
    result = Minuit2().optimize(estimator, {"x": 0.0, "y": 0.0})
    assert result.minimum_valid
    for name, value in true.items():
        fitted, error = result.parameter_values[name], result.parameter_errors[name]
        assert abs(fitted - value) < 5 * error, (name, fitted, error)


def correlated_acceptance():
    """A 3D acceptance with a correlation between m2(pi+ pi-), |cos theta| and t, binned as from simulation, times a decay-time turn-on."""
    from aman_examples.acceptance import (  # ruff: ignore[import-outside-top-level]
        FunctionAcceptance,
        HistogramAcceptance,
        decay_time_acceptance,
    )

    edges = [
        np.linspace(0.0, 3.0, 31),
        np.linspace(0.0, 10.0, 21),
        np.linspace(0.0, 1.0, 11),
    ]
    centers = np.meshgrid(*((e[1:] + e[:-1]) / 2 for e in edges), indexing="ij")
    m2_pipi, t, abs_cos = centers
    values = (1 - 0.3 * m2_pipi / 3) * (1 - 0.5 * abs_cos**2 * np.exp(-t))
    correlation = HistogramAcceptance(values, edges, ("m2_pipi", "t", "abs_cos_hel"))
    dalitz = FunctionAcceptance(lambda m2_kspim: 1 - 0.1 * m2_kspim, ("m2_kspim",))
    return (
        correlation * dalitz * decay_time_acceptance(t0=0.2, a=2.0, b=0.3, m=1.0, n=2.0)
    )


def test_fit_with_3d_acceptance(setup):
    from aman_examples.acceptance import apply_efficiency  # ruff: ignore[import-outside-top-level]

    generator, transformer, function = setup
    true = {"x": 0.05, "y": 0.05}
    function.update_parameters(true)
    rng = NumpyUniformRNG(seed=5)
    acceptance = correlated_acceptance()
    data = IntensityDistributionGenerator(
        generator, function, domain_transformer=transformer, bunch_size=200_000
    ).generate(80_000, rng)
    data = apply_efficiency(data, acceptance(data), rng=1)
    phsp = generator.generate(1_500_000, rng)
    phsp = apply_efficiency(phsp, acceptance(phsp), rng=2)
    estimator = UnbinnedNLL(
        function, transformer(data), transformer(phsp), backend="jax"
    )
    result = Minuit2().optimize(estimator, {"x": 0.0, "y": 0.0})
    assert result.minimum_valid
    for name, value in true.items():
        fitted, error = result.parameter_values[name], result.parameter_errors[name]
        assert abs(fitted - value) < 5 * error, (name, fitted, error)
