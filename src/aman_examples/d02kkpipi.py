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
from aman_examples.phasespace import PairPhaseSpaceGenerator

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


def phase_space_generator(model: AmpGenModel) -> PairPhaseSpaceGenerator:
    r"""Four-body phase space with AmpGen's masses, as :math:`(K^+K^-)(\pi^+\pi^-)`."""
    masses = tuple(model.table[name]["mass"] for name in model.final_state)
    return PairPhaseSpaceGenerator(masses, model.table[model.head]["mass"])


def peaked_phase_space_generator(
    model: AmpGenModel, fractions: tuple[float, float, float] = (0.2, 0.2, 0.1)
) -> PairPhaseSpaceGenerator:
    r"""Phase space enhanced at the :math:`\phi(1020)`, :math:`\rho(770)` and :math:`\omega(782)`.

    For toy generation with :class:`.ImportanceSampledFunction`: the fractions of
    :math:`m(K^+K^-)` drawn around the :math:`\phi(1020)`, and of :math:`m(\pi^+\pi^-)`
    around the :math:`\rho(770)` and :math:`\omega(782)`. The default makes the
    generation of toys from the full model about 10 times faster than from uniform phase
    space.
    """
    masses = tuple(model.table[name]["mass"] for name in model.final_state)

    def peak(name: str, fraction: float) -> tuple[float, float, float]:
        return model.table[name]["mass"], model.table[name]["width"], fraction

    f_phi, f_rho, f_omega = fractions
    return PairPhaseSpaceGenerator(
        masses,
        model.table[model.head]["mass"],
        peaks=(
            (peak("phi(1020)0", f_phi),),
            (peak("rho(770)0", f_rho), peak("omega(782)0", f_omega)),
        ),
    )
