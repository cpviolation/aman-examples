from __future__ import annotations

import numpy as np
import pytest

from aman_examples.acceptance import apply_efficiency, decay_time_efficiency

PARS = {"t0": 0.3, "a": 1.5, "b": 2.0, "m": 1.0, "n": 1.0}


def test_efficiency_shape():
    t = np.array([0.0, 0.3, 0.5, 1.0, 5.0])
    efficiency = decay_time_efficiency(t, **PARS)
    assert np.all(efficiency[:2] == 0)
    dt = t[2:] - PARS["t0"]
    np.testing.assert_allclose(efficiency[2:], dt**1.5 / (2.0 + dt))


def test_apply_efficiency_fraction():
    rng = np.random.default_rng(0)
    sample = {"t": rng.exponential(1.0, 200_000), "other": np.zeros(200_000)}
    efficiency = decay_time_efficiency(sample["t"], **PARS)
    kept = apply_efficiency(sample, efficiency, rng=1)
    assert len(kept["t"]) == len(kept["other"])
    assert len(kept["t"]) / 200_000 == pytest.approx(
        efficiency.mean() / efficiency.max(), rel=0.02
    )
    assert np.all(kept["t"] > PARS["t0"])
    with pytest.raises(ValueError, match="zero"):
        apply_efficiency(sample, np.zeros(200_000))
    with pytest.raises(ValueError, match="One efficiency"):
        apply_efficiency(sample, efficiency[:10])
