from __future__ import annotations

import numpy as np
import pytest
from tensorwaves.data.rng import NumpyUniformRNG

from aman_examples.binflip import (
    bin_flip_ratios,
    binflip_bin,
    load_binning,
    strong_phase_parameters,
)
from aman_examples.d02kspipi import formulate_model
from aman_examples.mixing import amplitude_transformer
from aman_examples.phasespace import DalitzPlotGenerator


def test_binning_is_antisymmetric():
    _, x_edges, _ = load_binning()
    centers = (x_edges[:-1] + x_edges[1:]) / 2
    x, y = (v.ravel() for v in np.meshgrid(centers, centers, indexing="ij"))
    off_diagonal = x != y
    bins = binflip_bin(x[off_diagonal], y[off_diagonal])
    np.testing.assert_array_equal(
        binflip_bin(y[off_diagonal], x[off_diagonal]), -bins * (bins != 0)
    )
    assert set(np.unique(np.abs(bins))) == set(range(9))
    assert binflip_bin(np.array([0.1]), np.array([5.0]))[0] == 0  # outside the grid


def test_ratios():
    t = np.arange(8.0)
    bins = np.array([1, -1, -1, 2, 1, -1, 2, -2])
    centers, ratio, error = bin_flip_ratios(t, bins, n_time_bins=2)
    np.testing.assert_allclose(centers, [1.5, 5.5])
    np.testing.assert_allclose(ratio[0], [1 / 2, 1 / 1])
    np.testing.assert_allclose(ratio[1], [0, 1])  # no -2 in the first time bin
    np.testing.assert_allclose(error[0, 0], np.sqrt(1 / 4 + 1 / 8))


@pytest.fixture(scope="module")
def strong_phases():
    model = formulate_model()
    masses = [
        model.parameter_defaults[model.masses[k]] for k in ("pi", "KS", "pi", "D")
    ]
    sample = DalitzPlotGenerator(masses).generate(500_000, NumpyUniformRNG(seed=9))
    sample["t"] = np.zeros(len(sample["m2_kspim"]))
    amplitudes = amplitude_transformer(model)(sample)
    bins = binflip_bin(sample["m2_kspim"], sample["m2_kspip"])
    return strong_phase_parameters(
        np.asarray(amplitudes["A_f"]), np.asarray(amplitudes[R"\bar{A}_f"]), bins
    )


def test_strong_phases_follow_equal_delta_delta_binning(strong_phases):
    """The binning was built from the BaBar 2008 model with equal 45 degree ranges of the strong-phase difference: arg(c + i s) changes by about -45 degrees per bin."""
    c, s = strong_phases["c"], strong_phases["s"]
    assert np.all(c**2 + s**2 <= 1)
    assert np.all(c**2 + s**2 > 0.3)  # coherence ~0.6 in bin 8
    phase = np.degrees(np.arctan2(s, c))
    step = (np.diff(phase) + 180) % 360 - 180
    assert np.all((step > -60) & (step < -25))
    assert abs(phase[0]) < 15


def test_fractions_sum_to_one(strong_phases):
    total = strong_phases["F_plus"].sum() + strong_phases["F_minus"].sum()
    assert total == pytest.approx(
        1, abs=0.01
    )  # 0.2% of the plot is outside the binning
    # the K*(892)- band (low m2(KS pi-)) is in the -b bins
    assert strong_phases["F_minus"].sum() > strong_phases["F_plus"].sum()
