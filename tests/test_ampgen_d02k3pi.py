"""D0 -> K-+ pi+- pi+- pi-+ models defined with AmpGen, against amplitudes computed with AmpGen.

References: data/ampgen_d02kpipipi_reference.npz (CF, D02Kpipipi.opt) and
data/ampgen_d02pikpipi_reference.npz (DCS, D02piKpipi.opt), computed with AmpGen fc651a7
(2000 phase-space events each; amplitude of every decay chain without couplings,
couplings, total amplitude), as described in macros/ampgen/README.md of
cpviolation/tfa_examples.
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import numpy as np
import pytest
from tensorwaves.data.rng import NumpyUniformRNG
from tensorwaves.function.sympy import create_parametrized_function

from aman_examples import d02k3pi
from aman_examples.ampgen.options import read_options

DATA = Path(__file__).parent / "data"
REF = {
    "cf": np.load(DATA / "ampgen_d02kpipipi_reference.npz"),
    "dcs": np.load(DATA / "ampgen_d02pikpipi_reference.npz"),
}
FORMULATE = {"cf": d02k3pi.formulate_cf_model, "dcs": d02k3pi.formulate_dcs_model}
CHAINS = [(k, str(n)) for k, ref in REF.items() for n in ref["names"]]


@pytest.fixture(scope="module")
def models():
    return {k: f() for k, f in FORMULATE.items()}


def rms(x):
    return np.sqrt(np.mean(np.abs(x) ** 2))


def test_read_options():
    options = read_options(d02k3pi.CF_OPTIONS)
    assert options.event_type == ["D0", "K-", "pi+", "pi+", "pi-"]
    c, free = options.couplings["D0{K*(892)bar0{K-,pi+},rho(770)0{pi+,pi-}}"]
    assert c == pytest.approx(cmath.rect(0.196, math.radians(-22.4)))  # degrees
    assert free
    params = {k: v for k, (v, _) in options.parameters.items()}
    assert params["IS_p1_mass"] == 0.651  # imported from kMatrix.opt
    dcs = read_options(d02k3pi.DCS_OPTIONS)
    assert dcs.event_type is None
    # sub-decays of the K1(1270)+ imported from the Dbar0 model
    assert "K(1)(1270)+[GSpline.EFF]{rho(770)0{pi+,pi-},K+}" in dcs.couplings


@pytest.mark.parametrize("key", ["cf", "dcs"])
def test_chains_match_ampgen(models, key):
    assert sorted(models[key].chains) == sorted(str(n) for n in REF[key]["names"])
    assert models[key].final_state == list(REF[key]["event_type"])


@pytest.mark.parametrize(("key", "name"), CHAINS)
def test_chain_amplitude(models, key, name):
    model, ref = models[key], REF[key]
    function = create_parametrized_function(
        model.chains[name].doit(), model.parameter_defaults, "jax"
    )
    amplitude = np.asarray(function(model.data_from_momenta(ref["events"])))
    reference = ref["amplitudes"][list(ref["names"]).index(name)]
    # AmpGen evaluates its amplitudes with fast-math: agreement ~1e-7 - 1e-5 of the rms
    assert np.max(np.abs(amplitude - reference)) < 1e-4 * rms(reference)
    coupling = complex(model.couplings[name].subs(model.parameter_defaults))
    expected = ref["couplings"][list(ref["names"]).index(name)]
    assert coupling == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("key", ["cf", "dcs"])
def test_identical_pions(models, key):
    """The amplitude is symmetric under the exchange of the two identical pions."""
    model = models[key]
    function = create_parametrized_function(
        model.amplitude.doit(), model.parameter_defaults, "jax"
    )
    events = REF[key]["events"][:200]
    swapped = events[:, [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]]
    a = np.asarray(function(model.data_from_momenta(events)))
    b = np.asarray(function(model.data_from_momenta(swapped)))
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * rms(a))
    np.testing.assert_allclose(a, REF[key]["total"][:200], atol=1e-4 * rms(a))


def test_phase_space(models):
    model = models["cf"]
    sample = d02k3pi.phase_space_generator(model).generate(
        1000, NumpyUniformRNG(seed=1)
    )
    p = [np.column_stack([sample[f"p{i}_{c}"] for c in "xyzE"]) for i in range(4)]
    for momentum, name in zip(p, model.final_state, strict=True):
        mass = np.sqrt(momentum[:, 3] ** 2 - np.sum(momentum[:, :3] ** 2, axis=1))
        np.testing.assert_allclose(mass, model.table[name]["mass"], rtol=1e-9)
    total = sum(p)
    np.testing.assert_allclose(total[:, :3], 0, atol=1e-12)
    np.testing.assert_allclose(total[:, 3], model.table["D0"]["mass"], rtol=1e-12)
