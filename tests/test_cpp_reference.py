"""Term-by-term comparison with the Belle C++ implementation."""

from __future__ import annotations

import numpy as np
import pytest
from tensorwaves.function.sympy import create_parametrized_function

from aman_examples.d02kspipi import COMPONENTS, CPP_KMATRIX_MASSES, formulate_model


def evaluate(model, expr, ref, backend="jax"):
    function = create_parametrized_function(
        expr.doit(), model.parameter_defaults, backend
    )
    return np.asarray(
        function({"m2_kspim": ref["m2_kspim"], "m2_kspip": ref["m2_kspip"]})
    )


def rel_dev(a, b):
    return np.abs(a - b) / np.abs(b)


@pytest.fixture(scope="module")
def models(cpp_reference):
    masses = tuple(float(cpp_reference[k]) for k in ("mpi", "mkz", "md"))
    return {
        "cpp": formulate_model(
            final_state_masses=masses, kmatrix_masses=CPP_KMATRIX_MASSES
        ),
        "pdg": formulate_model(final_state_masses=masses),
    }


def test_reference_contents(cpp_reference):
    # the C++ model has no K*(1680)+ term; its Belle coupling is zero
    terms = {k for k in COMPONENTS if k in cpp_reference}
    assert set(COMPONENTS) - terms == {"Kstar1680plus"}
    np.testing.assert_allclose(
        sum(cpp_reference[k] for k in terms), cpp_reference["total"], rtol=1e-12
    )


@pytest.mark.parametrize(
    "name", [c for c in COMPONENTS if c not in {"Kmatrix", "Kstar1680plus"}]
)
def test_component(models, cpp_reference, name):
    value = evaluate(models["pdg"], models["pdg"].term(name), cpp_reference)
    assert rel_dev(value, cpp_reference[name]).max() < 1e-9


def test_kmatrix_with_cpp_masses(models, cpp_reference):
    """With the channel masses of the C++ code the K-matrix is reproduced exactly (up to a few points where the 5x5 inverse is ill-conditioned)."""
    model = models["cpp"]
    dev = rel_dev(
        evaluate(model, model.term("Kmatrix"), cpp_reference), cpp_reference["Kmatrix"]
    )
    assert np.median(dev) < 1e-12
    assert dev.max() < 1e-4
    total = rel_dev(
        evaluate(model, model.amplitude, cpp_reference), cpp_reference["total"]
    )
    assert np.median(total) < 1e-12
    assert total.max() < 1e-4


def test_kmatrix_with_pdg_masses(models, cpp_reference):
    # PDG eta mass 0.547862 instead of 0.54775: largest effect at the eta-eta threshold
    model = models["pdg"]
    below = cpp_reference["m2_pipi"] <= 1.0
    dev = rel_dev(
        evaluate(model, model.term("Kmatrix"), cpp_reference), cpp_reference["Kmatrix"]
    )
    assert dev[below].max() < 5e-3
    assert dev[~below].max() < 1e-2


# Piecewise evaluates both branches of the complex square roots in NumPy
@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
def test_numpy_backend_agrees(models, cpp_reference):
    model = models["cpp"]
    jax_value = evaluate(model, model.amplitude, cpp_reference, "jax")
    np_value = evaluate(model, model.amplitude, cpp_reference, "numpy")
    np.testing.assert_allclose(np_value, jax_value, rtol=1e-10)


def test_unused_component_is_zero(models, cpp_reference):
    model = models["pdg"]
    assert np.all(evaluate(model, model.term("Kstar1680plus"), cpp_reference) == 0)
