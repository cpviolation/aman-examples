from __future__ import annotations

import numpy as np
import pytest
from tensorwaves.data import IntensityDistributionGenerator
from tensorwaves.data.rng import NumpyUniformRNG

from aman_examples.phasespace import (
    DalitzPlotGenerator,
    ImportanceSampledFunction,
    NBodyPhaseSpaceGenerator,
    PairPhaseSpaceGenerator,
    is_inside,
)

MASSES = (0.13957, 0.497611, 0.13957, 1.86484)  # pi, KS, pi, D0


def test_generator_inside_and_uniform():
    generator = DalitzPlotGenerator(MASSES)
    sample = generator.generate(100_000, NumpyUniformRNG(seed=1))
    x, y = sample["m2_kspim"], sample["m2_kspip"]
    assert len(x) == len(y) == 100_000
    assert np.all(is_inside(x, y, *MASSES))
    # symmetric masses: the two variables have the same distribution
    assert abs(np.mean(x) - np.mean(y)) < 0.01


def test_boundary():
    _, _, mc, md = MASSES
    assert not is_inside(0.1, 1.0, *MASSES)  # below the KS pi threshold
    assert not is_inside(2.0, 2.0, *MASSES)  # s_ab + s_bc too large
    assert is_inside(1.0, 1.0, *MASSES)
    # at the edge of the s_ab range there is a single s_bc value
    s_ab = (md - mc) ** 2 - 1e-9
    assert not np.any(is_inside(s_ab, np.linspace(0.3, 3.0, 100), *MASSES))


def test_area():
    """The Dalitz plot area is the phase-space integral (up to constants)."""
    generator = DalitzPlotGenerator(MASSES)
    ma, mb, mc, md = MASSES
    s_ab = np.linspace(*generator.range_ab, 200_001)[1:-1]
    m_ab = np.sqrt(s_ab)
    e_b = (s_ab - ma**2 + mb**2) / (2 * m_ab)
    e_c = (md**2 - s_ab - mc**2) / (2 * m_ab)
    width = 4 * np.sqrt(e_b**2 - mb**2) * np.sqrt(e_c**2 - mc**2)
    exact = np.trapezoid(width, s_ab)
    assert abs(generator.area / exact - 1) < 5e-3


def _mass(sample, indices):
    p = sum(
        np.column_stack([sample[f"p{i}_{c}"] for c in ("x", "y", "z", "E")])
        for i in indices
    )
    return np.sqrt(p[:, 3] ** 2 - np.sum(p[:, :3] ** 2, axis=1))


# K+ K- pi+ pi- from D0, with peaks at the phi(1020), rho(770), omega(782)
KKPIPI = (0.493677, 0.493677, 0.13957, 0.13957)
PEAKS = (((1.0195, 0.00426, 0.3),), ((0.7755, 0.149, 0.2), (0.7827, 0.0085, 0.1)))


@pytest.mark.parametrize("peaks", [((), ()), PEAKS])
def test_pair_generator_kinematics(peaks):
    sample = PairPhaseSpaceGenerator(KKPIPI, 1.86484, peaks=peaks).generate(
        10_000, NumpyUniformRNG(seed=1)
    )
    for i, m in enumerate(KKPIPI):
        np.testing.assert_allclose(_mass(sample, (i,)), m, rtol=1e-9)
    for c in ("x", "y", "z"):
        np.testing.assert_allclose(
            sum(sample[f"p{i}_{c}"] for i in range(4)), 0, atol=1e-12
        )
    np.testing.assert_allclose(
        sum(sample[f"p{i}_E"] for i in range(4)), 1.86484, rtol=1e-12
    )


@pytest.mark.parametrize("peaks", [((), ()), PEAKS])
def test_pair_generator_is_phase_space(peaks):
    """Weighted by 1/r, the events are distributed as GENBOD's four-body phase space."""
    reference = NBodyPhaseSpaceGenerator(KKPIPI, 1.86484).generate(
        200_000, NumpyUniformRNG(seed=2)
    )
    generator = PairPhaseSpaceGenerator(KKPIPI, 1.86484, peaks=peaks)
    sample = generator.generate(200_000, NumpyUniformRNG(seed=3))
    weights = 1 / generator.density_ratio(sample)
    if peaks == ((), ()):
        np.testing.assert_allclose(weights, 1)
    for indices in ((0, 1), (2, 3), (0, 3), (1, 2), (0, 1, 2)):
        a, b = _mass(reference, indices), _mass(sample, indices)
        edges = np.quantile(a, np.linspace(0, 1, 21))
        edges[[0, -1]] += (-1e-9, 1e-9)
        n_a, _ = np.histogram(a, edges)
        n_b, _ = np.histogram(b, edges, weights=weights)
        n_b2, _ = np.histogram(b, edges, weights=weights**2)
        scale = len(a) / weights.sum()
        chi2 = np.sum((n_a - scale * n_b) ** 2 / (n_a + scale**2 * n_b2))
        assert chi2 < 45, indices  # 20 degrees of freedom; p ~ 1e-3


def test_pair_generator_peaks():
    generator = PairPhaseSpaceGenerator(KKPIPI, 1.86484, peaks=PEAKS)
    sample = generator.generate(100_000, NumpyUniformRNG(seed=4))
    m_kk = _mass(sample, (0, 1))
    near_phi = np.abs(m_kk - 1.0195) < 0.00426 / 2
    # about half of the Cauchy component lies within one half-width of the peak
    assert near_phi.mean() > 0.3 * 0.45
    ratio = generator.density_ratio(sample)
    assert ratio[near_phi].min() > 10 * np.median(ratio)


def test_pair_generator_errors():
    with pytest.raises(ValueError, match="partition"):
        PairPhaseSpaceGenerator(KKPIPI, 1.86484, pairs=((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="forbidden"):
        PairPhaseSpaceGenerator(KKPIPI, 1.0)
    with pytest.raises(ValueError, match="sum to 1"):
        PairPhaseSpaceGenerator(
            KKPIPI, 1.86484, peaks=(((1.0195, 0.004, 0.6), (1.0195, 0.004, 0.5)), ())
        )


def test_importance_sampled_toys():
    """Toys from a narrow peak: same distribution as from uniform phase space."""

    def intensity(sample):  # a phi(1020) Breit-Wigner in m(K+K-)
        return 1 / ((_mass(sample, (0, 1)) - 1.0195) ** 2 + 0.00213**2)

    uniform = PairPhaseSpaceGenerator(KKPIPI, 1.86484)
    peaked = PairPhaseSpaceGenerator(KKPIPI, 1.86484, peaks=PEAKS)
    reference = IntensityDistributionGenerator(uniform, intensity).generate(
        5000, NumpyUniformRNG(seed=5)
    )
    toys = IntensityDistributionGenerator(
        peaked, ImportanceSampledFunction(intensity, peaked)
    ).generate(5000, NumpyUniformRNG(seed=6))
    for indices in ((0, 1), (2, 3), (0, 2)):
        a, b = _mass(reference, indices), _mass(toys, indices)
        edges = np.quantile(np.concatenate([a, b]), np.linspace(0, 1, 21))
        edges[-1] += 1e-9
        n_a, _ = np.histogram(a, edges)
        n_b, _ = np.histogram(b, edges)
        assert np.sum((n_a - n_b) ** 2 / (n_a + n_b)) < 45, indices


def test_density_ratio_normalisation():
    """1/r averages to 1 over generated events: phase-space integrals are unbiased."""
    generator = PairPhaseSpaceGenerator(KKPIPI, 1.86484, peaks=PEAKS)
    weights = 1 / generator.density_ratio(
        generator.generate(500_000, NumpyUniformRNG(seed=7))
    )
    error = weights.std() / np.sqrt(len(weights))
    assert abs(weights.mean() - 1) < 5 * error
