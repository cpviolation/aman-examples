from __future__ import annotations

import numpy as np
from tensorwaves.data.rng import NumpyUniformRNG

from aman_examples.phasespace import DalitzPlotGenerator, is_inside

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
