"""LinearAmplitudeNLL against TensorWaves' UnbinnedNLL, on a small linear amplitude."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.function.sympy import create_parametrized_function

from aman_examples.dynamics import AbsSquared
from aman_examples.estimator import LinearAmplitudeNLL

X = sp.Symbol("x", real=True)
C1, C2 = sp.symbols("c1 c2")
BASIS = {C1: sp.exp(sp.I * 3 * X), C2: X**2}
CONSTANT = 1 + sp.I * X
VALUES = {"c1": 0.8 - 0.3j, "c2": -0.5 + 1.1j}


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(1)
    data = {"x": rng.uniform(-1, 1, 500)}
    phsp = {"x": rng.uniform(-1, 1, 3000), "weights": rng.uniform(0.5, 1.5, 3000)}
    return data, phsp


def reference(samples):
    amplitude = sum(c * b for c, b in BASIS.items()) + CONSTANT
    function = create_parametrized_function(
        AbsSquared(amplitude), {C1: 1.0 + 0j, C2: 1.0 + 0j}, backend="jax"
    )
    return UnbinnedNLL(function, *samples, backend="jax")


def test_linear_amplitude_nll(samples):
    nll = LinearAmplitudeNLL(BASIS, CONSTANT, *samples, chunk_size=700)
    ref = reference(samples)
    for values in [VALUES, {"c1": 0.1j, "c2": 2.0 + 0j}]:
        assert nll(values) == pytest.approx(float(ref(values)), abs=1e-9)
    # missing parameters keep their previous values
    assert nll({"c1": 0.1j}) == pytest.approx(
        float(ref({"c1": 0.1j, "c2": 2.0})), abs=1e-9
    )


def test_linear_amplitude_nll_gradient(samples):
    nll = LinearAmplitudeNLL(BASIS, CONSTANT, *samples)
    gradient = nll.gradient(VALUES)
    step = 1e-6
    for name, value in VALUES.items():
        for unit, part in [(1, "real"), (1j, "imag")]:
            up = nll({**VALUES, name: value + step * unit})
            down = nll({**VALUES, name: value - step * unit})
            derivative = getattr(gradient[name], part)
            assert derivative == pytest.approx((up - down) / (2 * step), rel=1e-5)
