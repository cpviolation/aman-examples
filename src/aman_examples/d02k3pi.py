r"""The LHCb :math:`D^0 \to K^\mp \pi^\pm \pi^\pm \pi^\mp` models as defined in AmpGen.

The models of Eur. Phys. J. C 78 (2018) 443, from AmpGen's options files (copied
unmodified to :file:`data/ampgen/`, GPL-2.0-only as AmpGen) and built with
:func:`.formulate_ampgen_model`:

- the Cabibbo-favoured (CF) :math:`D^0 \to K^- \pi^+ \pi^+ \pi^-` model,
  ``D02Kpipipi.opt``: 33 decay chains, dominated by :math:`a_1(1260)^+ K^-`;
- the doubly Cabibbo-suppressed (DCS) :math:`D^0 \to K^+ \pi^- \pi^- \pi^+` model,
  ``D02piKpipi.opt``: 15 decay chains, dominated by :math:`K_1(1270)^+ \pi^-` and
  :math:`K_1(1400)^+ \pi^-`. Its :math:`K_1(1270)^+` decays and lineshape parameters are
  those of the CF model for :math:`\bar{D}^0`, imported from ``Dbar02piKpipi.opt``.

Both use Breit-Wigner, Gounaris-Sakurai, the FOCUS :math:`K\pi` S-wave, AmpGen's
:math:`\pi\pi` K-matrix (``kMatrix.opt``) and spline running widths for the
:math:`a_1(1260)`, :math:`K_1(1270)` and :math:`K(1460)`. The amplitudes are symmetric
under the exchange of the two identical pions, as in AmpGen. The final states are
ordered as in the options files: :math:`(K^-, \pi^+, \pi^+, \pi^-)` and :math:`(K^+,
\pi^-, \pi^-, \pi^+)`.
"""

from __future__ import annotations

from aman_examples.ampgen.model import AmpGenModel, formulate_ampgen_model
from aman_examples.io import DATA_DIR
from aman_examples.phasespace import PairPhaseSpaceGenerator

CF_OPTIONS = DATA_DIR / "ampgen" / "D02Kpipipi.opt"
DCS_OPTIONS = DATA_DIR / "ampgen" / "D02piKpipi.opt"
PARTICLES = DATA_DIR / "ampgen" / "particles.json"
CF_FINAL_STATE = ["K-", "pi+", "pi+", "pi-"]
DCS_FINAL_STATE = ["K+", "pi-", "pi-", "pi+"]


def formulate_cf_model(
    options: str = str(CF_OPTIONS), particles: str = str(PARTICLES)
) -> AmpGenModel:
    r"""The Cabibbo-favoured :math:`D^0 \to K^- \pi^+ \pi^+ \pi^-` model."""
    model = formulate_ampgen_model(options, particles)
    if model.final_state != CF_FINAL_STATE:
        msg = f"Expected final state {CF_FINAL_STATE}, got {model.final_state}"
        raise ValueError(msg)
    return model


def formulate_dcs_model(
    options: str = str(DCS_OPTIONS), particles: str = str(PARTICLES)
) -> AmpGenModel:
    r"""The doubly Cabibbo-suppressed :math:`D^0 \to K^+ \pi^- \pi^- \pi^+` model.

    The options file has no ``EventType``; it is that of AmpGen's documentation.
    """
    return formulate_ampgen_model(
        options, particles, event_type=["D0", *DCS_FINAL_STATE]
    )


def phase_space_generator(model: AmpGenModel) -> PairPhaseSpaceGenerator:
    r"""Four-body phase space with AmpGen's masses, as :math:`(K\pi)(\pi\pi)`."""
    masses = tuple(model.table[name]["mass"] for name in model.final_state)
    return PairPhaseSpaceGenerator(
        masses, model.table[model.head]["mass"], pairs=((0, 1), (2, 3))
    )
