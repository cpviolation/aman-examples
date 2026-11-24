"""Decay-time efficiency and its application to toy samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from particle.particle import literals as lp

if TYPE_CHECKING:
    from tensorwaves.interface import DataSample


def decay_time_efficiency(  # ruff: ignore[too-many-positional-arguments]
    t: np.ndarray, t0: float, a: float, b: float, m: float, n: float
) -> np.ndarray:
    r"""Decay-time efficiency (not normalised).

    .. math::

        \varepsilon(t) = \left(\frac{(t-t_0)^a}{b + (t-t_0)^n}\right)^m
        \quad (t > t_0), \qquad \varepsilon(t) = 0 \quad (t \le t_0)

    For :math:`a > n` the function grows without bound, so with accept-reject the
    efficiency is relative to its largest value in the sample.
    """
    dt = np.asarray(t, dtype=float) - t0
    dtp = np.clip(dt, 0.0, None)
    return np.where(dt > 0, (dtp**a / (b + dtp**n)) ** m, 0.0)


def apply_efficiency(
    sample: DataSample,
    efficiency: np.ndarray,
    rng: np.random.Generator | int | None = None,
) -> DataSample:
    """Keep each event with probability efficiency / max(efficiency)."""
    efficiency = np.asarray(efficiency, dtype=float)
    size = len(next(iter(sample.values())))
    if efficiency.shape != (size,):
        msg = "One efficiency value per event is needed"
        raise ValueError(msg)
    maximum = efficiency.max()
    if maximum <= 0:
        msg = "The efficiency is zero for all events"
        raise ValueError(msg)
    keep = np.random.default_rng(rng).uniform(0.0, maximum, size) < efficiency
    return {k: np.asarray(v)[keep] for k, v in sample.items()}


# ----------------------------------------------------------------------------
# Acceptance models in (m2(KS pi-), m2(KS pi+), t), as the EfficiencyModel of the
# C++ toy generator in d02kspipi_toys (generator/src/EfficiencyModel.cpp)

#: Masses (pi+, KS, D0) in GeV, PDG values as in :func:`.d02kspipi.formulate_model`
DEFAULT_MASSES = (lp.pi_plus.mass / 1e3, lp.K_S_0.mass / 1e3, lp.D_0.mass / 1e3)


def helicity_cosine(
    m2_pipi: np.ndarray,
    m2_kspip: np.ndarray,
    masses: tuple[float, float, float] = DEFAULT_MASSES,
) -> np.ndarray:
    r"""Cosine of the helicity angle of the :math:`\pi^+\pi^-` system.

    The angle between the :math:`K^0_S` and the :math:`\pi^+` in the :math:`\pi^+\pi^-`
    rest frame, from :math:`m^2(\pi^+\pi^-)` and :math:`m^2(K^0_S\pi^+)`, as ``CosHel``
    of d02kspipi_toys (``common/src/common_funcs.cpp``); clipped to :math:`[-1, 1]`.
    """
    m_pi, m_ks, m_d = masses
    m2_pipi, m2_kspip = (
        np.asarray(m2_pipi, dtype=float),
        np.asarray(m2_kspip, dtype=float),
    )
    q2 = np.clip((m2_pipi - 2 * m_pi**2) ** 2 - 4 * m_pi**4, 0, None) / (
        4 * m2_pipi
    )  # pi in pipi frame
    p2 = np.clip((m_d**2 - m_ks**2 - m2_pipi) ** 2 - 4 * m_ks**2 * m2_pipi, 0, None) / (
        4 * m_d**2
    )
    # KS energy and momentum in the pipi frame
    e_ks = (np.sqrt((m_ks**2 + p2) * (m2_pipi + p2)) + p2) / np.sqrt(m2_pipi)
    p_ks = np.sqrt(np.clip(e_ks**2 - m_ks**2, 0, None))
    e_pi = np.sqrt(m_pi**2 + q2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (e_ks * e_pi - (m2_kspip - m_ks**2 - m_pi**2) / 2) / (p_ks * np.sqrt(q2))
    return np.clip(cos, -1.0, 1.0)


def acceptance_variables(
    sample: DataSample, masses: tuple[float, float, float] = DEFAULT_MASSES
) -> dict[str, np.ndarray]:
    r"""The variables an acceptance can depend on, from ``m2_kspim``, ``m2_kspip`` (and ``t``): also ``m2_pipi``, ``cos_hel`` (:func:`helicity_cosine`) and ``abs_cos_hel``."""
    m_pi, m_ks, m_d = masses
    m2_kspim, m2_kspip = np.asarray(sample["m2_kspim"]), np.asarray(sample["m2_kspip"])
    m2_pipi = m_d**2 + m_ks**2 + 2 * m_pi**2 - m2_kspim - m2_kspip
    cos_hel = helicity_cosine(m2_pipi, m2_kspip, masses)
    variables = {
        "m2_kspim": m2_kspim,
        "m2_kspip": m2_kspip,
        "m2_pipi": m2_pipi,
        "cos_hel": cos_hel,
        "abs_cos_hel": np.abs(cos_hel),
    }
    if "t" in sample:
        variables["t"] = np.asarray(sample["t"])
    return variables


class Acceptance:
    """An efficiency (not normalised) per event of a `~tensorwaves.interface.DataSample`.

    Acceptances can be multiplied (``acceptance_1 * acceptance_2``), as the components
    of ``EfficiencyModel`` in d02kspipi_toys.
    """

    def __call__(self, sample: DataSample) -> np.ndarray:
        return self.evaluate(acceptance_variables(sample, self.masses))

    masses: tuple[float, float, float] = DEFAULT_MASSES

    def evaluate(self, variables: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def __mul__(self, other: Acceptance) -> ProductAcceptance:
        return ProductAcceptance(self, other)


class ProductAcceptance(Acceptance):
    """Product of acceptances."""

    def __init__(self, *components: Acceptance) -> None:
        self.components = [
            c
            for a in components
            for c in (a.components if isinstance(a, ProductAcceptance) else [a])
        ]
        self.masses = self.components[0].masses

    def evaluate(self, variables: dict[str, np.ndarray]) -> np.ndarray:
        result = np.ones_like(next(iter(variables.values())), dtype=float)
        for component in self.components:
            result *= component.evaluate(variables)
        return result


class FunctionAcceptance(Acceptance):
    """An acceptance given by a function of some of the variables of :func:`acceptance_variables`, passed as keyword arguments."""

    def __init__(
        self,
        function,
        variables: tuple[str, ...],
        masses: tuple[float, float, float] = DEFAULT_MASSES,
    ) -> None:
        self.function, self.variables, self.masses = function, tuple(variables), masses

    def evaluate(self, variables: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(
            self.function(**{v: variables[v] for v in self.variables}), dtype=float
        )


class HistogramAcceptance(Acceptance):
    """A binned acceptance in one to three variables of :func:`acceptance_variables`.

    The efficiency of an event is the content of its bin (``TH1::GetBinContent`` of
    ``FindBin``, as ``EfficiencyModel`` in d02kspipi_toys), and zero outside the
    histogram. For example, the correlation model of d02kspipi_toys is a histogram in
    ``("m2_pipi", "t", "abs_cos_hel")``.

    Args:
        values: Bin contents, of shape ``(n_1, ..., n_d)``.

        edges: Bin edges in each variable.

        variables: The variable of each axis.

        normalise: Divide by the largest bin content, as ``EfficiencyModel`` does for
            its correlation histogram.

        interpolate: For a one-dimensional histogram, interpolate linearly between bin
            centres (constant beyond the first and last centre), as
            ``TH1::Interpolate``.
    """

    def __init__(
        self,
        values: np.ndarray,
        edges: list[np.ndarray],
        variables: tuple[str, ...],
        *,
        normalise: bool = False,
        interpolate: bool = False,
        masses: tuple[float, float, float] = DEFAULT_MASSES,
    ) -> None:
        self.values = np.array(
            values, dtype=float
        )  # a copy: normalise divides in place
        self.interpolate = interpolate
        self.edges = [np.asarray(e, dtype=float) for e in edges]
        self.variables = tuple(variables)
        self.masses = masses
        if self.values.ndim != len(self.edges) or self.values.ndim != len(
            self.variables
        ):
            msg = "One set of edges and one variable per histogram axis are needed"
            raise ValueError(msg)
        if self.values.shape != tuple(len(e) - 1 for e in self.edges):
            msg = f"Bin contents of shape {self.values.shape} do not match the edges"
            raise ValueError(msg)
        if interpolate and self.values.ndim != 1:
            msg = "Interpolation is only implemented for one-dimensional histograms"
            raise ValueError(msg)
        if normalise:
            self.values /= self.values.max()

    @classmethod
    def from_root(
        cls,
        file_name: str,
        histogram: str,
        variables: tuple[str, ...],
        *,
        normalise: bool = False,
        interpolate: bool = False,
        masses: tuple[float, float, float] = DEFAULT_MASSES,
    ) -> HistogramAcceptance:
        """Read a ``TH1``, ``TH2`` or ``TH3`` (e.g. ``"corr"`` of d02kspipi_toys)."""
        import uproot  # ruff: ignore[import-outside-top-level]

        with uproot.open(file_name) as f:
            values, *edges = f[histogram].to_numpy()
        return cls(
            values,
            edges,
            variables,
            normalise=normalise,
            interpolate=interpolate,
            masses=masses,
        )

    def evaluate(self, variables: dict[str, np.ndarray]) -> np.ndarray:
        if self.interpolate:
            centres = (self.edges[0][1:] + self.edges[0][:-1]) / 2
            return np.interp(variables[self.variables[0]], centres, self.values)
        indices, inside = [], None
        for name, edges in zip(self.variables, self.edges, strict=True):
            x = variables[name]
            i = np.searchsorted(edges, x, side="right") - 1
            ok = (i >= 0) & (i < len(edges) - 1)
            inside = ok if inside is None else inside & ok
            indices.append(np.clip(i, 0, len(edges) - 2))
        return np.where(inside, self.values[tuple(indices)], 0.0)


def decay_time_acceptance(
    t0: float, a: float, b: float, m: float, n: float
) -> FunctionAcceptance:
    """:func:`decay_time_efficiency` as an :class:`Acceptance`."""
    return FunctionAcceptance(
        lambda t: decay_time_efficiency(t, t0, a, b, m, n), ("t",)
    )


def efficiency_model_from_root(
    dalitz: str | None = None,
    decay_time: str | None = None,
    correlations: str | None = None,
    min_decay_time: float = 0.0,
    masses: tuple[float, float, float] = DEFAULT_MASSES,
) -> Acceptance:
    r"""The acceptance of ``EfficiencyModel`` in d02kspipi_toys, from its ROOT files.

    The product of the models given (each argument is the name of a ROOT file):

    - ``dalitz``: ``TH2D`` ``"Dal"`` with :math:`m^2(K^0_S\pi^+)` on the x axis and
      :math:`m^2(K^0_S\pi^-)` on the y axis, bin content;
    - ``decay_time``: ``TH1D`` ``"DT"`` in :math:`t/\tau`, linearly interpolated;
    - ``correlations``: ``TH3D`` ``"corr"`` in :math:`(m^2(\pi^+\pi^-), t/\tau,
      |\cos\theta_\mathrm{hel}|)`, normalised to its maximum, bin content;

    and zero for :math:`t/\tau` below ``min_decay_time`` if the decay-time or the
    correlation model is given. The detection asymmetry of ``EfficiencyModel`` is not
    included: it depends on the flavour of the :math:`D` meson.
    """
    components: list[Acceptance] = []
    if dalitz:
        components.append(
            HistogramAcceptance.from_root(
                dalitz, "Dal", ("m2_kspip", "m2_kspim"), masses=masses
            )
        )
    if decay_time:
        components.append(
            HistogramAcceptance.from_root(
                decay_time, "DT", ("t",), interpolate=True, masses=masses
            )
        )
    if correlations:
        components.append(
            HistogramAcceptance.from_root(
                correlations,
                "corr",
                ("m2_pipi", "t", "abs_cos_hel"),
                normalise=True,
                masses=masses,
            )
        )
    if not components:
        msg = "No acceptance model given"
        raise ValueError(msg)
    if decay_time or correlations:
        components.append(
            FunctionAcceptance(
                lambda t: (t >= min_decay_time).astype(float), ("t",), masses
            )
        )
    return ProductAcceptance(*components)
