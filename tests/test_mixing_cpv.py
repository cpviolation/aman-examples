"""Mixing and CP violation with the bin-flip parameters x_CP, y_CP, Delta x, Delta y."""

from __future__ import annotations

import numpy as np
import pytest
from tensorwaves.data import IntensityDistributionGenerator
from tensorwaves.data.rng import NumpyUniformRNG
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.optimizer import Minuit2

from aman_examples.binflip import (
    binflip_bin,
    binflip_ratio,
    fit_binflip,
    strong_phase_parameters,
    time_binned_ratios,
)
from aman_examples.d02kspipi import formulate_model
from aman_examples.estimator import SimultaneousEstimator
from aman_examples.mixing import (
    amplitude_transformer,
    formulate_mixing_model,
    formulate_mixing_model_zcp,
    mixing_from_zcp,
    zcp_from_mixing,
)
from aman_examples.phasespace import DalitzDecayTimeGenerator, DalitzPlotGenerator

TRUTH = {
    "x": 0.03,
    "y": 0.025,
    "q_over_p": 0.8,
    "phase": -0.3,
}  # exaggerated mixing and CPV
NAMES = ("x_CP", "y_CP", "Delta_x", "Delta_y")


def test_conversions():
    z = zcp_from_mixing(**TRUTH)
    back = mixing_from_zcp(*z.values())
    for key, value in TRUTH.items():
        assert back[key] == pytest.approx(value, abs=1e-12)
    # without CP violation in mixing: x_CP = x, y_CP = y, Delta x = Delta y = 0
    no_cpv = zcp_from_mixing(0.004, 0.0064)
    assert no_cpv == pytest.approx({
        "x_CP": 0.004,
        "y_CP": 0.0064,
        "Delta_x": 0,
        "Delta_y": 0,
    })
    # negative y: the branch still reduces to y_CP + i x_CP
    back = mixing_from_zcp(*zcp_from_mixing(0.004, -0.0064, 0.9, 0.1).values())
    assert back == pytest.approx({
        "x": 0.004,
        "y": -0.0064,
        "q_over_p": 0.9,
        "phase": 0.1,
    })


@pytest.mark.parametrize("initial_d0", [True, False])
def test_zcp_rate_equals_mixing_rate(initial_d0):
    rng = np.random.default_rng(1)
    n = 100
    data = {
        "A_f": rng.normal(size=n) + 1j * rng.normal(size=n),
        R"\bar{A}_f": rng.normal(size=n) + 1j * rng.normal(size=n),
        "t": 3 * rng.exponential(size=n),
    }
    reference = formulate_mixing_model(**TRUTH, initial_d0=initial_d0)
    zcp = formulate_mixing_model_zcp(
        **dict(
            zip(
                ("x_cp_value", "y_cp_value", "delta_x_value", "delta_y_value"),
                zcp_from_mixing(**TRUTH).values(),
                strict=True,
            )
        ),
        initial_d0=initial_d0,
    )
    f_ref = create_parametrized_function(
        reference.intensity.doit(), reference.parameter_defaults, "numpy"
    )
    f_zcp = create_parametrized_function(
        zcp.intensity.doit(), zcp.parameter_defaults, "numpy"
    )
    np.testing.assert_allclose(f_zcp(data), f_ref(data), rtol=1e-12)
    assert {s.name for s in zcp.parameter_defaults} == {*NAMES, R"\tau"}


@pytest.fixture(scope="module")
def setup():
    model = formulate_model()
    masses = [
        model.parameter_defaults[model.masses[k]] for k in ("pi", "KS", "pi", "D")
    ]
    return model, masses


def strong_phases(masses, transformer, size):
    """c_b, s_b and r_b = F+b/F-b on a mirror-symmetrised uniform sample."""
    half = DalitzPlotGenerator(masses).generate(size // 2, NumpyUniformRNG(seed=2))
    sample = {
        "m2_kspim": np.concatenate([half["m2_kspim"], half["m2_kspip"]]),
        "m2_kspip": np.concatenate([half["m2_kspip"], half["m2_kspim"]]),
    }
    sample["t"] = np.zeros(len(sample["m2_kspim"]))
    amplitudes = transformer(sample)
    bins = binflip_bin(sample["m2_kspim"], sample["m2_kspip"])
    s = strong_phase_parameters(
        np.asarray(amplitudes["A_f"]), np.asarray(amplitudes[R"\bar{A}_f"]), bins
    )
    return sample, amplitudes, bins, s["c"] - 1j * s["s"], s["F_plus"] / s["F_minus"]


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
@pytest.mark.parametrize("initial_d0", [True, False])
def test_binflip_expansion(setup, initial_d0):  # ruff: ignore[too-many-locals]
    """R_b(t) of the second-order expansion against exact integrals of the rate."""
    model, masses = setup
    transformer = amplitude_transformer(model, backend="numpy")
    sample, amplitudes, bins, x_b, r = strong_phases(masses, transformer, 200_000)
    if not initial_d0:  # Dalitz-plot coordinates swapped for D0bar
        bins = binflip_bin(sample["m2_kspip"], sample["m2_kspim"])
    z = zcp_from_mixing(**TRUTH)
    z_cp, dz = -(z["y_CP"] + 1j * z["x_CP"]), -(z["Delta_y"] + 1j * z["Delta_x"])
    mixing = formulate_mixing_model(
        **TRUTH, initial_d0=initial_d0, relative_to_decay=False
    )
    rate = create_parametrized_function(
        mixing.intensity.doit(), mixing.parameter_defaults, "numpy"
    )
    for t in (0.5, 2.0, 4.0):
        w = rate({**amplitudes, "t": np.full(len(bins), t)})
        exact = np.array([
            w[bins == b].sum() / w[bins == -b].sum() for b in range(1, 9)
        ])
        zz = z_cp + (dz if initial_d0 else -dz)
        approx = binflip_ratio(
            np.array([t]), np.array([t**2]), r, x_b, zz, z_cp**2 - dz**2
        )[:, 0]
        effect = np.abs(exact - r).max()
        assert effect > 0.01
        assert np.abs(approx - exact).max() < 0.01 * effect


def test_simultaneous_estimator():
    class Quadratic:
        def __init__(self, centre):
            self.centre = centre

        def __call__(self, p):
            return (p["a"] - self.centre) ** 2

        def gradient(self, p):
            return {"a": 2 * (p["a"] - self.centre)}

    combined = SimultaneousEstimator(Quadratic(1.0), Quadratic(3.0))
    assert combined({"a": 2.0}) == pytest.approx(2.0)
    assert combined.gradient({"a": 2.0}) == {"a": 0.0}
    result = Minuit2().optimize(combined, {"a": 0.0})
    assert result.parameter_values["a"] == pytest.approx(2.0, abs=1e-6)
    with pytest.raises(ValueError, match="At least one"):
        SimultaneousEstimator()


@pytest.mark.slow
def test_fits_recover_cpv_parameters(setup):  # ruff: ignore[too-many-locals]
    """Toys of D0 and D0bar; amplitude fit and bin-flip fit of x_CP, y_CP, Delta x, Delta y."""
    model, masses = setup
    transformer = amplitude_transformer(model)
    truth = zcp_from_mixing(**TRUTH)
    generator = DalitzDecayTimeGenerator(masses)
    rng = NumpyUniformRNG(seed=11)
    phsp = transformer(generator.generate(1_000_000, rng))
    functions, data = {}, {}
    for flavour, initial_d0 in (("D0", True), ("D0bar", False)):
        rate = formulate_mixing_model_zcp(*truth.values(), initial_d0=initial_d0)
        functions[flavour] = create_parametrized_function(
            rate.intensity.doit(), rate.parameter_defaults, "jax"
        )
        data[flavour] = IntensityDistributionGenerator(
            generator,
            functions[flavour],
            domain_transformer=transformer,
            bunch_size=200_000,
        ).generate(100_000, rng)
    estimator = SimultaneousEstimator(
        *(
            UnbinnedNLL(functions[f], transformer(data[f]), phsp, backend="jax")
            for f in functions
        )
    )
    result = Minuit2().optimize(
        estimator, {"x_CP": 0.01, "y_CP": 0.01, "Delta_x": 0.0, "Delta_y": 0.0}
    )
    assert result.minimum_valid
    for name in NAMES:
        value, error = result.parameter_values[name], result.parameter_errors[name]
        assert abs(value - truth[name]) < 5 * error, (name, value, error)

    _, _, _, x_b, r = strong_phases(
        masses, amplitude_transformer(model, backend="numpy"), 400_000
    )
    edges = np.quantile(
        np.concatenate([data[f]["t"] for f in data]), np.linspace(0, 1, 11)
    )
    edges[-1] += 1e-9
    ratios = {
        "D0": time_binned_ratios(
            data["D0"]["t"],
            binflip_bin(data["D0"]["m2_kspim"], data["D0"]["m2_kspip"]),
            edges,
        ),
        "D0bar": time_binned_ratios(
            data["D0bar"]["t"],
            binflip_bin(data["D0bar"]["m2_kspip"], data["D0bar"]["m2_kspim"]),
            edges,
        ),
    }
    minuit = fit_binflip(ratios, x_b, r)
    assert minuit.valid
    for name in NAMES:
        assert abs(minuit.values[name] - truth[name]) < 5 * minuit.errors[name], name
