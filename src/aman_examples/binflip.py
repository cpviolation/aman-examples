r"""Bin-flip method: Dalitz-plot binning, ratios :math:`R_b(t)` and strong phases.

The binning is the BaBar 2008 "equal :math:`\Delta\delta_D`" scheme with 8 bins on each
side of the :math:`m^2(K^0_S\pi^-) = m^2(K^0_S\pi^+)` diagonal
(:file:`data/BinningLUT_K0Spipi_BABAR2008_EqualDeltadeltaD.root`). Bin :math:`+b` is the
part of bin :math:`b` with :math:`m^2(K^0_S\pi^-) > m^2(K^0_S\pi^+)` and :math:`-b` its
mirror image; the Cabibbo-favoured :math:`K^*(892)^-` band, at low
:math:`m^2(K^0_S\pi^-)`, is in the :math:`-b` bins.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from aman_examples.io import DATA_DIR

BINNING_LUT = DATA_DIR / "BinningLUT_K0Spipi_BABAR2008_EqualDeltadeltaD.root"
BINNING_LUT_HISTOGRAM = "h_BinningLUT_K0Spipi_BABAR2008_EqualDeltadeltaD"
N_BINS = 8


@cache
def load_binning() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Bin numbers on a grid, with the grid edges in :math:`m^2(K^0_S\pi^-)` and :math:`m^2(K^0_S\pi^+)`."""
    import uproot  # ruff: ignore[import-outside-top-level]

    with uproot.open(BINNING_LUT) as f:
        values, x_edges, y_edges = f[BINNING_LUT_HISTOGRAM].to_numpy()
    return values.astype(int), x_edges, y_edges


def binflip_bin(m2_kspim: np.ndarray, m2_kspip: np.ndarray) -> np.ndarray:
    r"""Signed bin number (:math:`\pm 1, \ldots, \pm 8`; 0 outside the binning).

    The sign is positive where the grid index in :math:`m^2(K^0_S\pi^-)` is larger than
    that in :math:`m^2(K^0_S\pi^+)`, as in tfa_examples (``tfaex.io``).
    """
    values, x_edges, y_edges = load_binning()
    ix = np.digitize(m2_kspim, x_edges) - 1
    iy = np.digitize(m2_kspip, y_edges) - 1
    inside = (ix >= 0) & (ix < len(x_edges) - 1) & (iy >= 0) & (iy < len(y_edges) - 1)
    ix, iy = np.where(inside, ix, 0), np.where(inside, iy, 0)
    sign = np.where(ix > iy, 1, -1)
    return np.where(inside, values[ix, iy] * sign, 0)


def bin_flip_ratios(
    t: np.ndarray, bins: np.ndarray, n_time_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Bin-flip ratios :math:`R_b(t)` in decay-time bins.

    The events are split into ``n_time_bins`` decay-time bins with equal numbers of
    events, and :math:`R_b = N(+b) / N(-b)` is computed in each, with the uncertainty
    :math:`\sqrt{N_+/N_-^2 + N_+^2/N_-^3}`.

    Returns:
        The mean decay time of each time bin (shape ``(n_time_bins,)``), and the ratios
        and their uncertainties (shape ``(8, n_time_bins)``, bins 1 to 8).
    """
    order = np.argsort(t)
    t_split = np.array_split(np.asarray(t)[order], n_time_bins)
    bin_split = np.array_split(np.asarray(bins)[order], n_time_bins)
    centers = np.array([s.mean() for s in t_split])
    n_plus = np.array([
        [np.sum(s == b) for s in bin_split] for b in range(1, N_BINS + 1)
    ])
    n_minus = np.array([
        [np.sum(s == -b) for s in bin_split] for b in range(1, N_BINS + 1)
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_minus > 0, n_plus / n_minus, 0.0)
        error = np.where(
            n_minus > 0, np.sqrt(n_plus / n_minus**2 + n_plus**2 / n_minus**3), 0.0
        )
    return centers, ratio, error


def strong_phase_parameters(
    amplitude: np.ndarray, amplitude_bar: np.ndarray, bins: np.ndarray
) -> dict[str, np.ndarray]:
    r"""Strong-phase parameters :math:`c_b, s_b` and fractions :math:`F_{\pm b}`.

    From the :math:`D^0` and :math:`\bar{D}^0` amplitudes at events uniform in the
    Dalitz plot (Monte Carlo integration):

    .. math::

        c_b + i s_b = \frac{\int_b A_f \bar{A}_f^*}
            {\sqrt{\int_b |A_f|^2 \int_b |\bar{A}_f|^2}}, \qquad
        F_{\pm b} = \frac{\int_{\pm b} |A_f|^2}{\int |A_f|^2},

    with the integrals over bin :math:`+b`; then :math:`c_b` is the amplitude-weighted
    average of :math:`\cos(\delta_D - \bar{\delta}_D)` and :math:`s_b` that of its sine,
    with :math:`\delta_D = \arg A_f`, :math:`\bar\delta_D = \arg\bar{A}_f`.

    Returns:
        Arrays over bins 1 to 8: ``c``, ``s``, ``F_plus``, ``F_minus``.
    """
    amplitude, amplitude_bar = np.asarray(amplitude), np.asarray(amplitude_bar)
    total = np.sum(np.abs(amplitude) ** 2)
    out: dict[str, list[float]] = {"c": [], "s": [], "F_plus": [], "F_minus": []}
    for b in range(1, N_BINS + 1):
        a, a_bar = amplitude[bins == b], amplitude_bar[bins == b]
        cross = np.sum(a * np.conj(a_bar))
        norm = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(a_bar) ** 2))
        out["c"].append(cross.real / norm)
        out["s"].append(cross.imag / norm)
        out["F_plus"].append(np.sum(np.abs(a) ** 2) / total)
        out["F_minus"].append(np.sum(np.abs(amplitude[bins == -b]) ** 2) / total)
    return {k: np.array(v) for k, v in out.items()}


def binflip_ratio(  # ruff: ignore[too-many-positional-arguments]
    t_mean: np.ndarray,
    t2_mean: np.ndarray,
    r: np.ndarray,
    x_b: np.ndarray,
    z: complex,
    w2: complex,
) -> np.ndarray:
    r"""Bin-flip ratio :math:`R_b(t)` to second order in the mixing parameters.

    For an initial :math:`D^0`, with :math:`z = z_{CP} + \Delta z` and :math:`w^2 =
    z_{CP}^2 - \Delta z^2 = (q/p)(p/q)(y + ix)^2`:

    .. math::

        R_b = \frac{r_b\left(1 + \frac{\langle t^2\rangle}{4}\mathrm{Re}\,w^2\right)
              + \sqrt{r_b}\langle t\rangle \mathrm{Re}(X_b z)
              + \frac{\langle t^2\rangle}{4}|z|^2}
             {1 + \frac{\langle t^2\rangle}{4}\mathrm{Re}\,w^2
              + \sqrt{r_b}\langle t\rangle \mathrm{Re}(X_b^* z)
              + r_b\frac{\langle t^2\rangle}{4}|z|^2},

    with :math:`t` in units of :math:`\tau`, :math:`r_b = F_{+b}/F_{-b}` and :math:`X_b
    = c_b - i s_b` (:func:`strong_phase_parameters`). For an initial :math:`\bar{D}^0`,
    with the Dalitz-plot coordinates swapped before binning, the same expression holds
    with :math:`z = z_{CP} - \Delta z`.

    It follows from :math:`A(t) \propto \cosh\frac{(y+ix)t}{2} A_f + \frac{z\,t}{2}
    \bar{A}_f + O(t^2)` integrated over bin :math:`+b` (numerator) and :math:`-b`
    (denominator).
    """
    t_mean, t2_mean = np.asarray(t_mean), np.asarray(t2_mean)
    r, x_b = np.asarray(r)[:, None], np.asarray(x_b)[:, None]
    common = 1 + t2_mean / 4 * np.real(w2)
    numerator = (
        r * common + np.sqrt(r) * t_mean * np.real(x_b * z) + t2_mean / 4 * abs(z) ** 2
    )
    denominator = (
        common
        + np.sqrt(r) * t_mean * np.real(np.conj(x_b) * z)
        + r * t2_mean / 4 * abs(z) ** 2
    )
    return numerator / denominator


def time_binned_ratios(
    t: np.ndarray, bins: np.ndarray, edges: np.ndarray
) -> dict[str, np.ndarray]:
    r"""Ratios :math:`R_b = N(+b)/N(-b)` in the decay-time bins ``edges``, with their uncertainties and :math:`\langle t\rangle, \langle t^2\rangle` of each time bin."""
    t, bins = np.asarray(t), np.asarray(bins)
    index = np.digitize(t, edges) - 1
    n = len(edges) - 1
    t_mean = np.array([t[index == i].mean() for i in range(n)])
    t2_mean = np.array([np.mean(t[index == i] ** 2) for i in range(n)])
    n_plus = np.array([
        [np.sum((bins == b) & (index == i)) for i in range(n)]
        for b in range(1, N_BINS + 1)
    ])
    n_minus = np.array([
        [np.sum((bins == -b) & (index == i)) for i in range(n)]
        for b in range(1, N_BINS + 1)
    ])
    ratio = n_plus / n_minus
    error = np.sqrt(n_plus / n_minus**2 + n_plus**2 / n_minus**3)
    return {"t_mean": t_mean, "t2_mean": t2_mean, "ratio": ratio, "error": error}


def fit_binflip(
    ratios: dict[str, dict[str, np.ndarray]],
    x_b: np.ndarray,
    r_initial: np.ndarray,
    initial: dict[str, float] | None = None,
) -> Any:
    r"""Least-squares fit of :math:`x_{CP}, y_{CP}, \Delta x, \Delta y` and the :math:`r_b` to bin-flip ratios of :math:`D^0` and :math:`\bar{D}^0` decays.

    Args:
        ratios: :func:`time_binned_ratios` of each flavour, keyed ``"D0"`` and
            ``"D0bar"`` (the latter binned with swapped Dalitz-plot coordinates).

        x_b: :math:`X_b = c_b - i s_b` of the 8 bins (fixed, e.g. from a model or
            from measurements at charm threshold).

        r_initial: Starting values of :math:`r_b`.

        initial: Starting values of ``x_CP``, ``y_CP``, ``Delta_x``, ``Delta_y`` (0 by
            default).

    Returns:
        The `iminuit.Minuit` object after ``migrad`` and ``hesse``.
    """
    from iminuit import Minuit  # ruff: ignore[import-outside-top-level]

    names = ["x_CP", "y_CP", "Delta_x", "Delta_y"]
    r_names = [f"r_{b}" for b in range(1, N_BINS + 1)]
    sign = {"D0": 1, "D0bar": -1}

    def chi2(*p: float) -> float:
        xcp, ycp, dx, dy = p[:4]
        r = np.array(p[4:])
        z_cp, dz = -(ycp + 1j * xcp), -(dy + 1j * dx)
        total = 0.0
        for flavour, data in ratios.items():
            z = z_cp + sign[flavour] * dz
            model = binflip_ratio(
                data["t_mean"], data["t2_mean"], r, x_b, z, z_cp**2 - dz**2
            )
            total += np.sum(((data["ratio"] - model) / data["error"]) ** 2)
        return total

    start = dict.fromkeys(names, 0.0) | dict(initial or {})
    start |= {n: float(v) for n, v in zip(r_names, r_initial, strict=True)}
    minuit = Minuit(chi2, *start.values(), name=list(start))
    minuit.errordef = Minuit.LEAST_SQUARES
    for name in r_names:
        minuit.limits[name] = (0, None)
    minuit.migrad()
    minuit.hesse()
    return minuit
