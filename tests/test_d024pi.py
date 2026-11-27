"""The BESIII D0 -> pi+ pi- pi+ pi- model, against an independent NumPy implementation and the paper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from tensorwaves.data.rng import NumpyUniformRNG
from tensorwaves.function.sympy import create_function

from aman_examples import d024pi


def _load_reference():
    path = Path(__file__).parent / "d024pi_reference.py"
    spec = importlib.util.spec_from_file_location("d024pi_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference_chains = _load_reference().chains

PARITY = np.array([-1, -1, -1, 1])


def momenta(sample):
    return np.stack(
        [np.column_stack([sample[f"p{i}_{c}"] for c in "xyzE"]) for i in range(4)],
        axis=1,
    )


def as_data(p):
    return {f"p{i}_{c}": p[:, i, j] for i in range(4) for j, c in enumerate("xyzE")}


def test_interpolation():
    x = sp.Symbol("x", real=True)
    expr = d024pi.Interpolation(x, 0.0, 1.0, 3.0, 2.0, 4.0, 0.0)
    function = create_function(expr, backend="numpy")
    values = function({"x": np.array([-1.0, 0.5, 2.0, 5.0])})
    np.testing.assert_allclose(values, [2.0, 3.0, 2.0, 0.0])
    with pytest.raises(ValueError, match="abscissae"):
        d024pi.Interpolation(x, 0.0, 1.0, 3.0)


def test_running_width_tables():
    for name in ("a1(1260)", "a1(1640)", "pi(1300)"):
        s, ratio = d024pi.running_width_table(name)
        m0 = d024pi.RESONANCES[name][0]
        assert np.interp(m0**2, s, ratio) == pytest.approx(1, abs=2e-3)
        assert ratio[0] == pytest.approx(0, abs=1e-3)
        assert np.all(ratio >= -1e-3)


@pytest.mark.slow
def test_running_width_knots():
    """The stored knots are those computed from the 3pi amplitudes."""
    computed = d024pi.compute_running_width_knots("pi(1300)")
    np.testing.assert_allclose(
        computed, d024pi.RUNNING_WIDTH_KNOTS["pi(1300)"], rtol=1e-6, atol=1e-8
    )


def test_phase_space():
    sample = d024pi.phase_space_generator().generate(1000, NumpyUniformRNG(seed=1))
    p = momenta(sample)
    mass = np.sqrt(p[..., 3] ** 2 - np.sum(p[..., :3] ** 2, axis=-1))
    np.testing.assert_allclose(mass, d024pi.M_PI, rtol=1e-9)
    np.testing.assert_allclose(p.sum(axis=1)[:, 3], d024pi.M_D0, rtol=1e-12)


@pytest.mark.slow
def test_amplitudes():  # ruff: ignore[too-many-locals]
    """Chains against the NumPy reference; fit fractions and F+ against the paper."""
    model = d024pi.formulate_model()
    assert sorted(model.chains) == sorted(d024pi.COUPLINGS)
    assert len(model.free_couplings) == len(d024pi.COUPLINGS) - 1
    sample = d024pi.phase_space_generator().generate(40_000, NumpyUniformRNG(seed=2))
    p = momenta(sample)
    p_bar = np.stack([p[:, j] * PARITY for j in (1, 0, 3, 2)], axis=1)
    reference = reference_chains(p[:500])
    amps, amps_bar = {}, {}
    for name, expr in model.chains.items():
        function = create_function(expr.doit(), backend="jax")
        values = np.asarray(function(as_data(p)))
        ref = reference[name]
        rms = np.sqrt(np.mean(np.abs(ref) ** 2))
        assert np.max(np.abs(values[:500] - ref)) < 1e-8 * rms, name
        coupling = model.parameter_defaults[model.couplings[name]]
        amps[name] = coupling * values
        amps_bar[name] = coupling * np.asarray(function(as_data(p_bar)))
    total, total_bar = sum(amps.values()), sum(amps_bar.values())
    norm = np.mean(np.abs(total) ** 2)
    for group, names in d024pi.fit_fraction_groups().items():
        fraction = 100 * np.mean(np.abs(sum(amps[n] for n in names)) ** 2) / norm
        value, error = d024pi.FIT_FRACTIONS[group]
        assert abs(fraction - value) < 2 * error, group
    plus = np.mean(np.abs(total + total_bar) ** 2)
    minus = np.mean(np.abs(total - total_bar) ** 2)
    value, error = d024pi.F_PLUS
    assert 100 * plus / (plus + minus) == pytest.approx(value, abs=error)
