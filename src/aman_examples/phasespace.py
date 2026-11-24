r"""Uniform sampling of a three-body Dalitz plot, as a TensorWaves domain generator.

A uniform distribution in the Dalitz variables :math:`(s_{ab}, s_{bc})` is uniform in
three-body phase space, so no event weights are needed. This replaces TensorWaves'
:class:`~tensorwaves.data.phasespace.TFPhaseSpaceGenerator`, which needs TensorFlow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tensorwaves.interface import DataGenerator

if TYPE_CHECKING:
    from tensorwaves.interface import DataSample, RealNumberGenerator


def is_inside(  # ruff: ignore[too-many-positional-arguments]
    s_ab: np.ndarray | float,
    s_bc: np.ndarray | float,
    ma: float,
    mb: float,
    mc: float,
    md: float,
) -> np.ndarray:
    r"""Whether :math:`(s_{ab}, s_{bc})` is inside the Dalitz plot of :math:`D \to abc`.

    Uses the range of :math:`s_{bc}` at fixed :math:`s_{ab}`, from the energies of
    :math:`b` and :math:`c` in the :math:`ab` rest frame.
    """
    s_ab, s_bc = np.asarray(s_ab), np.asarray(s_bc)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_ab = np.sqrt(s_ab)
        e_b = (s_ab - ma**2 + mb**2) / (2 * m_ab)
        e_c = (md**2 - s_ab - mc**2) / (2 * m_ab)
        p_b = np.sqrt(e_b**2 - mb**2)
        p_c = np.sqrt(e_c**2 - mc**2)
        lo = (e_b + e_c) ** 2 - (p_b + p_c) ** 2
        hi = (e_b + e_c) ** 2 - (p_b - p_c) ** 2
        inside = (
            (s_ab > (ma + mb) ** 2)
            & (s_ab < (md - mc) ** 2)
            & (s_bc > lo)
            & (s_bc < hi)
        )
    return inside & np.isfinite(lo) & np.isfinite(hi)


class DalitzPlotGenerator(DataGenerator):
    r"""Events uniform in the Dalitz plot of :math:`D \to a b c`.

    Args:
        masses: :math:`(m_a, m_b, m_c, m_D)` in GeV.

        variables: Names of the two variables, :math:`s_{ab}` and :math:`s_{bc}`, in the
            generated `~tensorwaves.interface.DataSample`.
    """

    def __init__(
        self,
        masses: tuple[float, float, float, float],
        variables: tuple[str, str] = ("m2_kspim", "m2_kspip"),
    ) -> None:
        self.masses = tuple(float(m) for m in masses)
        self.variables = variables
        ma, mb, mc, md = self.masses
        self.range_ab = ((ma + mb) ** 2, (md - mc) ** 2)
        self.range_bc = ((mb + mc) ** 2, (md - ma) ** 2)

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        s_ab, s_bc = np.empty(0), np.empty(0)
        while len(s_ab) < size:
            n = 2 * (size - len(s_ab)) + 1000
            x = rng(size=n, min_value=self.range_ab[0], max_value=self.range_ab[1])
            y = rng(size=n, min_value=self.range_bc[0], max_value=self.range_bc[1])
            keep = is_inside(x, y, *self.masses)
            s_ab = np.concatenate([s_ab, x[keep]])
            s_bc = np.concatenate([s_bc, y[keep]])
        return {self.variables[0]: s_ab[:size], self.variables[1]: s_bc[:size]}

    @property
    def area(self) -> float:
        """Area of the Dalitz plot in GeV^4 (Monte Carlo estimate, 1e6 points)."""
        rng = np.random.default_rng(0)
        n = 1_000_000
        x = rng.uniform(*self.range_ab, n)
        y = rng.uniform(*self.range_bc, n)
        box = (self.range_ab[1] - self.range_ab[0]) * (
            self.range_bc[1] - self.range_bc[0]
        )
        return box * float(np.mean(is_inside(x, y, *self.masses)))
