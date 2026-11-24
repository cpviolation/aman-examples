"""Comparison with the binned Belle amplitude used for the toys of tfa_examples.

``data/belle_cache.root`` holds the Belle amplitude computed by
``make_amplitude_hists.cpp`` of d02kspipi_toys on a 500x500 grid in x = m2(KS pi-), y =
m2(KS pi+).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import uproot
from tensorwaves.function.sympy import create_parametrized_function

from aman_examples.d02kspipi import formulate_model
from aman_examples.phasespace import is_inside

CACHE = Path(__file__).parent / "data" / "belle_cache.root"


def test_model_matches_cpp_cache():
    with uproot.open(CACHE) as f:
        amp, xbins, ybins = f["histo_amplitude"].to_numpy()
        phase, _, _ = f["histo_phase"].to_numpy()
    xc = 0.5 * (xbins[:-1] + xbins[1:])
    yc = 0.5 * (ybins[:-1] + ybins[1:])
    x, y = (v.ravel() for v in np.meshgrid(xc[::5], yc[::5], indexing="ij"))
    ref = (amp * np.exp(1j * phase))[::5, ::5].ravel()

    model = formulate_model()
    masses = [
        model.parameter_defaults[model.masses[k]] for k in ("pi", "KS", "pi", "D")
    ]
    use = is_inside(x, y, *masses) & (np.abs(ref) > 0)
    function = create_parametrized_function(
        model.amplitude.doit(), model.parameter_defaults, "jax"
    )
    ratio = np.asarray(function({"m2_kspim": x[use], "m2_kspip": y[use]})) / ref[use]
    # residual differences: K-matrix channel masses (PDG here) and final-state masses
    assert np.median(np.abs(ratio - 1)) < 5e-3
    assert abs(np.median(ratio.real) - 1) < 5e-3
    assert abs(np.median(ratio.imag)) < 5e-3
