"""Parametric acceptance in (s+, s0, t), against the v4 model of d02kspipi_run3."""

from __future__ import annotations

import numpy as np
import pytest
from tensorwaves.data.rng import NumpyUniformRNG

from aman_examples.acceptance import DEFAULT_MASSES, apply_efficiency
from aman_examples.parametric_acceptance import (
    V4_CONSTANTS,
    V4_PARAMETERS,
    V4_TERMS,
    V4_TIME_RANGE,
    fit_acceptance,
    formulate_acceptance,
)
from aman_examples.phasespace import DalitzDecayTimeGenerator

M_PI, M_KS, M_D = DEFAULT_MASSES
# parameter order of the notebook: logit (7), turn-on (6), z terms (3), s, u and higher-order terms
TERMS = list(V4_TERMS)
NAMES = [*TERMS[:7], "x0", "a", "b", "beta", "n", "m", *TERMS[7:10], "s", *TERMS[10:]]


def reference_log_eff_v4(p, m2ab, m2bc, t):  # ruff: ignore[too-many-locals]
    """The v4 model of EfficiencyParametrisation.ipynb (_features and _log_eff_v4).

    With NumPy instead of jax.numpy; m2ab = m2(KS pi+), m2bc = m2(pi+ pi-).
    """
    minab, maxab = (M_KS + M_PI) ** 2, (M_D - M_PI) ** 2
    minbc, maxbc = (2 * M_PI) ** 2, (M_D - M_KS) ** 2
    minac, maxac = minab, maxab
    msqsum = M_D**2 + M_KS**2 + 2 * M_PI**2

    def _normalise_msq(m_sq, m_sq_min, m_sq_max):
        return (m_sq - 0.5 * (m_sq_min + m_sq_max)) / (0.5 * (m_sq_max - m_sq_min))

    q = _normalise_msq(m2bc, minbc, maxbc)
    r = _normalise_msq(m2ab, minab, maxab) - _normalise_msq(
        msqsum - m2ab - m2bc, minac, maxac
    )
    tn = (t - V4_CONSTANTS["t_mean"]) / V4_CONSTANTS["t_std"]
    a0, a1, a2, a3, a4, b0, b1 = p[:7]
    x0, a, b, beta, n, m = p[7:13]
    i0, i1, i2 = p[13:16]
    s = p[16]
    k0, k1, k2, k3, k4, k5 = p[17:23]
    h0, h1, h2, h3 = p[23:27]
    g0, g1, g2 = p[27:30]
    u = np.exp(-(t - V4_CONSTANTS["t_min"]) / s)
    logit = (
        a0 + a1 * q + a2 * q**2 + a3 * r**2 + a4 * q * r**2 + b0 * r + b1 * q * r
        + i0 * q * tn + i1 * r * tn + i2 * q * tn**2
        + u * (k0 * q + k1 * r + k2 * q**2 + k3 * r**2 + k4 * q * r**2 + k5 * q * r)
        + h0 * q**3 + h1 * q**4 + h2 * r**4 + h3 * q**2 * r**2
        + u * (g0 * q**3 + g1 * r**4 + g2 * q**2 * r**2)
    )  # fmt: skip
    dt = t - x0
    pos = dt > 0
    ds = np.where(pos, dt, 1.0)
    log_dt = a * np.log(ds) - m * np.log(b + ds**n) - beta * t
    return -np.log1p(np.exp(-logit)) + np.where(pos, log_dt, -1e30)


@pytest.fixture(scope="module")
def generator():
    return DalitzDecayTimeGenerator((M_PI, M_KS, M_PI, M_D), time_range=V4_TIME_RANGE)


def test_v4_parameters_in_notebook_order():
    assert len(NAMES) == 30
    assert set(NAMES) == set(V4_PARAMETERS)


# NumPy evaluates both branches of the turn-on Piecewise
@pytest.mark.filterwarnings("ignore:invalid value encountered in power:RuntimeWarning")
def test_matches_notebook_v4(generator):
    sample = generator.generate(20_000, NumpyUniformRNG(seed=1))
    model = formulate_acceptance()
    ours = model.acceptance(backend="numpy")(sample)
    p = np.array([V4_PARAMETERS[n] for n in NAMES])
    m2_pipi = M_D**2 + M_KS**2 + 2 * M_PI**2 - sample["m2_kspim"] - sample["m2_kspip"]
    expected = np.exp(reference_log_eff_v4(p, sample["m2_kspip"], m2_pipi, sample["t"]))
    np.testing.assert_allclose(ours, expected, rtol=1e-12)
    # JAX backend; zero below the turn-on
    np.testing.assert_allclose(model.acceptance("jax")(sample), expected, rtol=1e-10)
    below = {k: v[:5] for k, v in sample.items()} | {"t": np.full(5, 0.2)}
    assert np.all(model.acceptance("numpy")(below) == 0)


def test_time_range(generator):
    t = generator.generate(200_000, NumpyUniformRNG(seed=2))["t"]
    t_min, t_max = V4_TIME_RANGE
    assert t.min() >= t_min
    assert t.max() < t_max
    # mean of the exponential truncated to [t_min, t_max), tau = 1
    expected = (
        t_min
        + 1
        - (t_max - t_min) * np.exp(-(t_max - t_min)) / -np.expm1(-(t_max - t_min))
    )
    assert t.mean() == pytest.approx(expected, abs=0.01)


def test_custom_terms():
    from aman_examples.parametric_acceptance import q, u  # ruff: ignore[import-outside-top-level]

    model = formulate_acceptance(
        terms={"c0": 1, "c1": q, "c2": u * q},
        parameters={
            "c0": 0.5,
            "c1": -0.3,
            "s": 1.0,
            "x0": 0.0,
            "a": 1.0,
            "b": 1.0,
            "beta": 0.0,
            "n": 1.0,
            "m": 1.0,
        },
    )
    assert {s.name for s in model.parameter_defaults} == {
        "c0",
        "c1",
        "c2",
        "s",
        "x0",
        "a",
        "b",
        "beta",
        "n",
        "m",
    }
    with pytest.raises(ValueError, match="No values"):
        formulate_acceptance(terms={"c0": 1}, parameters={"c0": 0.5})


def test_fit_recovers_parameters(generator):
    """Accept-reject with v4 on a reference sample, then fit some parameters back."""
    rng = NumpyUniformRNG(seed=3)
    model = formulate_acceptance()
    reference = generator.generate(1_000_000, rng)
    sample = generator.generate(1_500_000, rng)
    accepted = apply_efficiency(sample, model.acceptance("jax")(sample), rng=4)
    free = ["a1", "a2", "k0", "s", "beta"]
    initial = {name: 1.2 * V4_PARAMETERS[name] for name in free}
    result = fit_acceptance(model, accepted, reference, free=free, initial=initial)
    assert result.minimum_valid
    for name in free:
        value, error = result.parameter_values[name], result.parameter_errors[name]
        assert abs(value - V4_PARAMETERS[name]) < 5 * error, (name, value, error)


def test_fit_limits(generator):
    """Limits reach Minuit: a parameter bounded away from its true value ends at the bound."""
    rng = NumpyUniformRNG(seed=5)
    model = formulate_acceptance()
    reference = generator.generate(200_000, rng)
    sample = generator.generate(300_000, rng)
    accepted = apply_efficiency(sample, model.acceptance("jax")(sample), rng=6)
    result = fit_acceptance(
        model,
        accepted,
        reference,
        free=["a2"],
        initial={"a2": 0.1},
        limits={"a2": (None, 0.2)},
    )
    assert result.parameter_values["a2"] == pytest.approx(0.2, abs=1e-3)
