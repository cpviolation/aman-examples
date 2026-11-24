r"""The LHCb :math:`D^0 \to K^+ K^- \pi^+ \pi^-` model as defined in AmpGen.

The model of JHEP 02 (2019) 126, from AmpGen's ``options/DtoKKpipi_v2.opt`` (copied
unmodified to :file:`data/ampgen/DtoKKpipi_v2.opt`, GPL-2.0-only as AmpGen), built with
:func:`.formulate_ampgen_model`: 55 decay chains with Breit-Wigner, Gounaris-Sakurai,
:math:`\rho`-:math:`\omega`, :math:`a_0(980)` Flatté, FOCUS :math:`K\pi` S-wave,
AmpGen's :math:`\pi\pi`/:math:`K\bar{K}` K-matrix and spline running-width lineshapes.
The final state is ordered as :math:`(K^+, K^-, \pi^+, \pi^-)`.
"""

from __future__ import annotations

from aman_examples.ampgen.model import AmpGenModel, formulate_ampgen_model
from aman_examples.io import DATA_DIR
from aman_examples.phasespace import NBodyPhaseSpaceGenerator

OPTIONS = DATA_DIR / "ampgen" / "DtoKKpipi_v2.opt"
PARTICLES = DATA_DIR / "ampgen" / "particles.json"
FINAL_STATE = ["K+", "K-", "pi+", "pi-"]


def formulate_model(
    options: str = str(OPTIONS), particles: str = str(PARTICLES)
) -> AmpGenModel:
    r"""The :math:`D^0 \to K^+ K^- \pi^+ \pi^-` model of an AmpGen options file."""
    model = formulate_ampgen_model(options, particles)
    if model.final_state != FINAL_STATE:
        msg = f"Expected final state {FINAL_STATE}, got {model.final_state}"
        raise ValueError(msg)
    return model


def phase_space_generator(model: AmpGenModel) -> NBodyPhaseSpaceGenerator:
    """Four-body phase space with AmpGen's masses."""
    masses = tuple(model.table[name]["mass"] for name in model.final_state)
    return NBodyPhaseSpaceGenerator(masses, model.table[model.head]["mass"])
