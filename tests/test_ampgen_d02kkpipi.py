"""D0 -> K+ K- pi+ pi- model defined with AmpGen, against amplitudes computed with AmpGen.

Reference: data/ampgen_d02kkpipi_v2_reference.npz, computed with AmpGen fc651a7 from
DtoKKpipi_v2.opt (2000 phase-space events; amplitude of every decay chain without
couplings, couplings, total amplitude), as in cpviolation/tfa_examples.
"""

from __future__ import annotations

import cmath
from pathlib import Path

import numpy as np
import pytest
from tensorwaves.data.rng import NumpyUniformRNG
from tensorwaves.data.transform import SympyDataTransformer
from tensorwaves.function.sympy import create_parametrized_function

from aman_examples import d02kkpipi
from aman_examples.ampgen.model import coupling_symbol
from aman_examples.ampgen.options import parse_descriptor, read_options
from aman_examples.estimator import LinearAmplitudeNLL
from aman_examples.phasespace import ImportanceSampledFunction

REF = np.load(Path(__file__).parent / "data" / "ampgen_d02kkpipi_v2_reference.npz")
REF_NAMES = list(REF["names"])


@pytest.fixture(scope="module")
def model():
    return d02kkpipi.formulate_model()


@pytest.fixture(scope="module")
def data(model):
    return model.data_from_momenta(REF["events"])


@pytest.fixture(scope="module")
def d0_couplings(model):
    return [c for c in model.free_couplings if c.name.startswith("C_{D0")]


@pytest.fixture(scope="module")
def linear(model, d0_couplings):
    return model.linear_amplitude(d0_couplings)


def rms(x):
    return np.sqrt(np.mean(np.abs(x) ** 2))


def final_state_indices(particle):
    if not particle.daughters:
        return [particle.index]
    return [i for d in particle.daughters for i in final_state_indices(d)]


def near_spline_knot(particle, events, tolerance=5e-4):
    """Events whose mass squared, for a resonance with a spline lineshape, is within ``tolerance`` knot spacings of a knot. There AmpGen's spline can jump to the next interval (its interval index and its offset in the interval are computed separately, and do not always agree at a knot within rounding)."""
    mask = np.zeros(len(events), dtype=bool)
    if particle.daughters and particle.lineshape_name.startswith("GSpline"):
        n, xmin, xmax = particle.splines[particle.name]
        p = sum(events[:, 4 * i : 4 * i + 4] for i in final_state_indices(particle))
        s = p[:, 3] ** 2 - np.sum(p[:, :3] ** 2, axis=1)
        u = (s - xmin) / ((xmax - xmin) / (n - 1))
        mask |= (np.abs(u - np.round(u)) < tolerance) & (u > 0) & (u < n - 1)
    for d in particle.daughters:
        mask |= near_spline_knot(d, events, tolerance)
    return mask


def test_read_options():
    options = read_options(d02kkpipi.OPTIONS)
    assert options.event_type == ["D0", "K+", "K-", "pi+", "pi-"]
    params = {k: v for k, (v, _) in options.parameters.items()}
    assert params["D0_radius"] == 1.21
    assert params["K*(892)bar0_radius"] == params["K*(892)0_radius"] == 1.13  # alias
    c, free = options.couplings["D0[D]{K*(892)0{K+,pi-},K*(892)bar0{K-,pi+}}"]
    assert c == pytest.approx(cmath.rect(0.693083, 2.85743))
    assert free
    c_plus = options.couplings["K(1)(1270)+[GSpline.EFF]{rho(1450)0{pi+,pi-},K+}"][0]
    c_minus, free = options.couplings[
        "K(1)(1270)bar-[GSpline.EFF]{rho(1450)0{pi+,pi-},K-}"
    ]
    assert c_minus == pytest.approx(-c_plus)
    assert not free
    assert options.splines["a(1)(1260)+"] == (40, 0.1764, 3.0)


def test_descriptor_round_trip():
    text = "D0[D]{K*(1680)0{K+,pi-},K*(892)bar0{K-,pi+}}"
    assert parse_descriptor(text).descriptor() == text


def test_chains_match_ampgen(model):
    """Expansion of pseudo-particles, daughter ordering and orbital labels as in AmpGen."""
    assert sorted(model.chains) == sorted(REF_NAMES)


@pytest.mark.parametrize("name", REF_NAMES)
def test_chain_amplitude(model, data, name):
    function = create_parametrized_function(
        model.chains[name].doit(), model.parameter_defaults, "jax"
    )
    amplitude = np.asarray(function(data))
    reference = REF["amplitudes"][REF_NAMES.index(name)]
    use = ~near_spline_knot(model.trees[name], REF["events"])
    assert np.mean(~use) < 0.01
    # AmpGen evaluates its amplitudes with fast-math: agreement ~1e-6 - 1e-5 of the rms
    assert np.max(np.abs(amplitude - reference)[use]) < 1e-4 * rms(reference)
    coupling = complex(model.couplings[name].subs(model.parameter_defaults))
    assert coupling == pytest.approx(REF["couplings"][REF_NAMES.index(name)], abs=1e-12)


def test_total_amplitude(model, data):
    function = create_parametrized_function(
        model.amplitude.doit(), model.parameter_defaults, "jax"
    )
    total = np.asarray(function(data))
    use = ~np.any(
        [near_spline_knot(t, REF["events"]) for t in model.trees.values()], axis=0
    )
    assert np.max(np.abs(total - REF["total"])[use]) < 1e-4 * rms(REF["total"])


def test_free_couplings(model):
    assert (
        coupling_symbol("D0{phi(1020)0{K+,K-},rhoOmega00}") not in model.free_couplings
    )
    assert len([c for c in model.free_couplings if c.name.startswith("C_{D0")]) == 25


def test_phase_space(model):
    sample = d02kkpipi.phase_space_generator(model).generate(
        1000, NumpyUniformRNG(seed=1)
    )
    p = [np.column_stack([sample[f"p{i}_{c}"] for c in "xyzE"]) for i in range(4)]
    for momentum, name in zip(p, model.final_state, strict=True):
        mass = np.sqrt(momentum[:, 3] ** 2 - np.sum(momentum[:, :3] ** 2, axis=1))
        np.testing.assert_allclose(mass, model.table[name]["mass"], rtol=1e-9)
    total = sum(p)
    np.testing.assert_allclose(total[:, :3], 0, atol=1e-12)
    np.testing.assert_allclose(total[:, 3], model.table["D0"]["mass"], rtol=1e-12)


def test_linear_amplitude(model, data, d0_couplings, linear):
    """The basis amplitudes and couplings give the total amplitude of AmpGen."""
    basis, constant = linear
    expressions = {f"B{k}": e for k, e in enumerate([*basis.values(), constant])}
    transformer = SympyDataTransformer.from_sympy(expressions, backend="jax")
    columns = transformer(data)
    couplings = [model.parameter_defaults[c] for c in d0_couplings] + [1]
    total = sum(
        c * np.asarray(columns[name])
        for c, name in zip(couplings, expressions, strict=True)
    )
    use = ~np.any(
        [near_spline_knot(t, REF["events"]) for t in model.trees.values()], axis=0
    )
    assert np.max(np.abs(total - REF["total"])[use]) < 1e-4 * rms(REF["total"])


def test_linear_amplitude_not_linear(model):
    rho_omega = [c for c in model.free_couplings if c.name.startswith("C_{rhoOmega")]
    phi = coupling_symbol("D0{phi(1020)0{K+,K-},rhoOmega00}")
    assert rho_omega
    with pytest.raises(ValueError, match="not linear"):
        model.linear_amplitude([phi, *rho_omega])


@pytest.mark.slow
def test_small_fit(model, d0_couplings, linear):
    """Toys from the model; fit the 25 D0 couplings, the rhoOmega couplings fixed."""
    from tensorwaves.data import IntensityDistributionGenerator  # ruff: ignore[import-outside-top-level]
    from tensorwaves.optimizer import Minuit2  # ruff: ignore[import-outside-top-level]

    intensity = create_parametrized_function(
        model.intensity, model.parameter_defaults, "jax"
    )
    peaked = d02kkpipi.peaked_phase_space_generator(model)
    rng = NumpyUniformRNG(seed=3)
    data = IntensityDistributionGenerator(
        peaked, ImportanceSampledFunction(intensity, peaked), bunch_size=100_000
    ).generate(5000, rng)
    phsp = peaked.generate(200_000, rng)
    phsp["weights"] = 1 / peaked.density_ratio(phsp)
    estimator = LinearAmplitudeNLL(*linear, data, phsp)
    initial = {c.name: 1.2 * model.parameter_defaults[c] for c in d0_couplings}
    result = Minuit2().optimize(estimator, initial)
    assert result.minimum_valid
    for c in d0_couplings:
        true = model.parameter_defaults[c]
        value, error = result.parameter_values[c.name], result.parameter_errors[c.name]
        assert abs(value.real - true.real) < 5 * error.real
        assert abs(value.imag - true.imag) < 5 * error.imag
