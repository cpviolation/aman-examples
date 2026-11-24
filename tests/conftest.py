from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def cpp_reference() -> dict[str, np.ndarray]:
    """Belle amplitude terms from the C++ code at 2000 Dalitz points.

    Produced by ``macros/make_cpp_reference.py`` of cpviolation/tfa_examples from
    ``Resonance.cpp`` / ``d02kspipi_amplitudes.cpp`` of d02kspipi_toys (commit in
    ``commit``); every term includes its coupling.
    """
    with np.load(DATA_DIR / "d02kspipi_cpp_reference.npz") as f:
        ref = {k: f[k] for k in f.files}
    msqsum = ref["md"] ** 2 + 2 * ref["mpi"] ** 2 + ref["mkz"] ** 2
    ref["m2_pipi"] = msqsum - ref["m2_kspim"] - ref["m2_kspip"]
    return ref
