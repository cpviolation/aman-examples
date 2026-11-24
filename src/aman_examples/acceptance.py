"""Decay-time efficiency and its application to toy samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
