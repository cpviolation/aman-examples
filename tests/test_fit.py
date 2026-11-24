"""Generate toys from the Belle model and fit one coupling with Minuit."""

from __future__ import annotations

from tensorwaves.data import IntensityDistributionGenerator
from tensorwaves.data.rng import NumpyUniformRNG
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.optimizer import Minuit2

from aman_examples.d02kspipi import formulate_model
from aman_examples.phasespace import DalitzPlotGenerator

FLOATED = "Kstar892minus"


def test_fit_recovers_coupling():
    model = formulate_model()
    function = create_parametrized_function(
        model.intensity.doit(), model.parameter_defaults, "jax"
    )
    masses = [
        model.parameter_defaults[model.masses[k]] for k in ("pi", "KS", "pi", "D")
    ]
    phsp_generator = DalitzPlotGenerator(masses)
    rng = NumpyUniformRNG(seed=5)
    phsp = phsp_generator.generate(200_000, rng)
    data = IntensityDistributionGenerator(
        phsp_generator, function, bunch_size=200_000
    ).generate(20_000, rng)

    coupling = model.couplings[FLOATED]
    true = model.parameter_defaults[coupling]
    estimator = UnbinnedNLL(function, data, phsp, backend="jax")
    result = Minuit2().optimize(estimator, {coupling.name: 0.8 * true})
    assert result.minimum_valid
    value, error = (
        result.parameter_values[coupling.name],
        result.parameter_errors[coupling.name],
    )
    assert abs(value.real - true.real) < 5 * error.real
    assert abs(value.imag - true.imag) < 5 * error.imag
