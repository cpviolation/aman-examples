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


def test_helicity_cosine_from_four_vectors():
    """Angle between KS and pi+ in the pi+ pi- frame, from explicit boosts."""
    from tensorwaves.data.rng import NumpyUniformRNG  # ruff: ignore[import-outside-top-level]

    from aman_examples.acceptance import DEFAULT_MASSES, helicity_cosine  # ruff: ignore[import-outside-top-level]
    from aman_examples.phasespace import NBodyPhaseSpaceGenerator, _boost  # ruff: ignore[import-outside-top-level]

    m_pi, m_ks, m_d = DEFAULT_MASSES
    sample = NBodyPhaseSpaceGenerator((m_pi, m_ks, m_pi), m_d).generate(
        1000, NumpyUniformRNG(seed=1)
    )
    pim, ks, pip = (
        np.column_stack([sample[f"p{i}_{c}"] for c in "xyzE"]) for i in range(3)
    )

    def m2(p):
        return p[:, 3] ** 2 - np.sum(p[:, :3] ** 2, axis=1)

    pipi = pim + pip
    beta = -pipi[:, :3] / pipi[:, 3:4]
    ks_r, pip_r = _boost(ks, beta), _boost(pip, beta)
    expected = np.sum(ks_r[:, :3] * pip_r[:, :3], axis=1) / (
        np.linalg.norm(ks_r[:, :3], axis=1) * np.linalg.norm(pip_r[:, :3], axis=1)
    )
    np.testing.assert_allclose(
        helicity_cosine(m2(pipi), m2(ks + pip)), expected, atol=1e-9
    )


def test_histogram_acceptance(tmp_path):
    import uproot  # ruff: ignore[import-outside-top-level]

    from aman_examples.acceptance import HistogramAcceptance, decay_time_acceptance  # ruff: ignore[import-outside-top-level]

    rng = np.random.default_rng(2)
    edges = [np.linspace(0, 2, 5), np.linspace(0, 5, 6), np.linspace(0, 1, 3)]
    values = rng.uniform(0.2, 2.0, size=(4, 5, 2))
    acceptance = HistogramAcceptance(
        values, edges, ("m2_pipi", "t", "abs_cos_hel"), normalise=True
    )
    variables = {
        "m2_pipi": np.array([0.1, 1.9, 2.5, 0.6]),
        "t": np.array([0.5, 4.9, 1.0, 2.2]),
        "abs_cos_hel": np.array([0.2, 0.9, 0.5, 0.6]),
    }
    expected = (
        np.array([values[0, 0, 0], values[3, 4, 1], 0.0, values[1, 2, 1]])
        / values.max()
    )
    np.testing.assert_allclose(acceptance.evaluate(variables), expected)
    # a TH3 written to and read from a ROOT file
    file_name = tmp_path / "corr.root"
    with uproot.recreate(file_name) as f:
        f["corr"] = (values, *edges)
    from_root = HistogramAcceptance.from_root(
        file_name, "corr", ("m2_pipi", "t", "abs_cos_hel"), normalise=True
    )
    np.testing.assert_allclose(from_root.evaluate(variables), expected)
    # products of acceptances
    product = acceptance * decay_time_acceptance(**PARS)
    t_eff = decay_time_efficiency(variables["t"], **PARS)
    np.testing.assert_allclose(product.evaluate(variables), expected * t_eff)
    with pytest.raises(ValueError, match="do not match"):
        HistogramAcceptance(values[:-1], edges, ("m2_pipi", "t", "abs_cos_hel"))


def test_efficiency_model_from_root(tmp_path):
    """The EfficiencyModel files of d02kspipi_toys: Dal (TH2D), DT (TH1D), corr (TH3D)."""
    import uproot  # ruff: ignore[import-outside-top-level]

    from aman_examples.acceptance import efficiency_model_from_root  # ruff: ignore[import-outside-top-level]

    rng = np.random.default_rng(3)
    dal = (
        rng.uniform(0.5, 1, (10, 10)),
        np.linspace(0.3, 3.1, 11),
        np.linspace(0.3, 3.1, 11),
    )
    dt = (np.array([0.2, 0.6, 1.0, 1.0]), np.array([0.0, 1.0, 2.0, 3.0, 10.0]))
    corr = (
        rng.uniform(1, 4, (6, 5, 2)),
        np.linspace(0, 3, 7),
        np.linspace(0, 10, 6),
        np.linspace(0, 1, 3),
    )
    for name, key, hist in (
        ("dal", "Dal", dal),
        ("dt", "DT", dt),
        ("corr", "corr", corr),
    ):
        with uproot.recreate(tmp_path / f"{name}.root") as f:
            f[key] = hist
    model = efficiency_model_from_root(
        dalitz=tmp_path / "dal.root",
        decay_time=tmp_path / "dt.root",
        correlations=tmp_path / "corr.root",
        min_decay_time=0.3,
    )
    sample = {
        "m2_kspim": np.array([0.8, 0.8]),
        "m2_kspip": np.array([1.5, 1.5]),
        "t": np.array([0.2, 1.5]),
    }
    efficiency = model(sample)
    assert efficiency[0] == 0  # below the minimum decay time
    from aman_examples.acceptance import acceptance_variables  # ruff: ignore[import-outside-top-level]

    v = acceptance_variables(sample)
    ix, iy = (
        np.digitize(1.5, dal[1]) - 1,
        np.digitize(0.8, dal[2]) - 1,
    )  # x = m2(KS pi+)
    t_eff = np.interp(1.5, [0.5, 1.5, 2.5, 6.5], dt[0])
    ic = (
        np.digitize(v["m2_pipi"][1], corr[1]) - 1,
        np.digitize(1.5, corr[2]) - 1,
        np.digitize(v["abs_cos_hel"][1], corr[3]) - 1,
    )
    expected = dal[0][ix, iy] * t_eff * corr[0][ic] / corr[0].max()
    assert efficiency[1] == pytest.approx(expected)
