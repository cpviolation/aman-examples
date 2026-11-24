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


class DalitzDecayTimeGenerator(DataGenerator):
    r"""Events uniform in the Dalitz plot, with decay times from :math:`e^{-t/\tau}/\tau`.

    The decay times are *not* uniform: a rate used with these events, for toy generation
    or for normalisation integrals, has to be divided by :math:`e^{-t/\tau}/\tau` (see
    :func:`.time_evolution` with ``relative_to_decay=True``).

    Args:
        masses: :math:`(m_a, m_b, m_c, m_D)` in GeV.

        tau: Lifetime, in the units of the decay time.

        variables: Names of :math:`s_{ab}`, :math:`s_{bc}` and the decay time.
    """

    def __init__(
        self,
        masses: tuple[float, float, float, float],
        tau: float = 1.0,
        variables: tuple[str, str, str] = ("m2_kspim", "m2_kspip", "t"),
    ) -> None:
        self.dalitz = DalitzPlotGenerator(masses, variables[:2])
        self.tau = float(tau)
        self.variables = variables

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        sample = dict(self.dalitz.generate(size, rng))
        u = rng(size=size)
        sample[self.variables[2]] = -self.tau * np.log1p(-u)
        return sample


def _breakup_momentum(m: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.clip((m**2 - (m1 + m2) ** 2) * (m**2 - (m1 - m2) ** 2), 0, None)
    ) / (2 * m)


def _boost(p: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Boost four-vectors ``p`` (N, 4) with components (x, y, z, E) by velocity ``beta`` (N, 3)."""
    b2 = np.sum(beta**2, axis=1)
    gamma = 1 / np.sqrt(1 - b2)
    bp = np.sum(beta * p[:, :3], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(b2 > 0, (gamma - 1) * bp / b2, 0.0)
    space = p[:, :3] + (factor + gamma * p[:, 3])[:, None] * beta
    return np.column_stack([space, gamma * (p[:, 3] + bp)])


class NBodyPhaseSpaceGenerator(DataGenerator):
    r"""Unweighted events uniform in :math:`n`-body phase space, in the parent rest frame.

    Raubold-Lynch (GENBOD) algorithm, as ROOT's ``TGenPhaseSpace``, followed by
    accept-reject on its weights, so that no TensorFlow (TensorWaves'
    ``TFPhaseSpaceGenerator``) is needed. The sample has the four-momentum components
    ``p<i>_x``, ``p<i>_y``, ``p<i>_z``, ``p<i>_E`` of each particle ``i``.

    Args:
        masses: Masses of the final-state particles in GeV.

        parent_mass: Mass of the decaying particle in GeV.
    """

    def __init__(self, masses: tuple[float, ...], parent_mass: float) -> None:
        self.masses = np.array(masses, dtype=float)
        self.parent_mass = float(parent_mass)
        n = len(self.masses)
        kinetic = self.parent_mass - self.masses.sum()
        if kinetic <= 0:
            msg = "The decay is kinematically forbidden"
            raise ValueError(msg)
        # maximum weight (TGenPhaseSpace::SetDecay)
        mmin, mmax, self.max_weight = 0.0, kinetic + self.masses[0], 1.0
        for k in range(1, n):
            mmin += self.masses[k - 1]
            mmax += self.masses[k]
            self.max_weight *= float(
                _breakup_momentum(
                    np.array(mmax), np.array(mmin), np.array(self.masses[k])
                )
            )

    def _weighted(
        self, size: int, rng: RealNumberGenerator
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.masses)
        kinetic = self.parent_mass - self.masses.sum()
        r = np.sort(np.column_stack([rng(size=size) for _ in range(n - 2)]), axis=1)
        cumulative = np.cumsum(self.masses)
        invariant = np.empty((size, n))  # mass of the system of particles 0..k
        invariant[:, 0] = self.masses[0]
        invariant[:, 1:-1] = cumulative[1:-1] + r * kinetic
        invariant[:, -1] = self.parent_mass
        momenta = [
            _breakup_momentum(
                invariant[:, k], invariant[:, k - 1], np.full(size, self.masses[k])
            )
            for k in range(1, n)
        ]
        weight = np.prod(momenta, axis=0)

        def isotropic(p: np.ndarray) -> np.ndarray:
            cos_theta = 2 * rng(size=size) - 1
            phi = 2 * np.pi * rng(size=size)
            sin_theta = np.sqrt(1 - cos_theta**2)
            return p[:, None] * np.column_stack([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta,
            ])

        # particles 0 and 1 back to back in the rest frame of system (0, 1), then add
        # particle k recoiling against system (0..k-1), boosting the latter
        three = isotropic(momenta[0])
        four = [
            np.column_stack([three, np.sqrt(momenta[0] ** 2 + self.masses[0] ** 2)]),
            np.column_stack([-three, np.sqrt(momenta[0] ** 2 + self.masses[1] ** 2)]),
        ]
        for k in range(2, n):
            three = isotropic(momenta[k - 1])
            system_energy = np.sqrt(momenta[k - 1] ** 2 + invariant[:, k - 1] ** 2)
            beta = -three / system_energy[:, None]
            four = [_boost(p, beta) for p in four]
            four.append(
                np.column_stack([
                    three,
                    np.sqrt(momenta[k - 1] ** 2 + self.masses[k] ** 2),
                ])
            )
        return np.concatenate(four, axis=1), weight

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        accepted: list[np.ndarray] = []
        count = 0
        while count < size:
            n = 2 * (size - count) + 1000
            events, weight = self._weighted(n, rng)
            keep = rng(size=n, max_value=self.max_weight) < weight
            accepted.append(events[keep])
            count += int(keep.sum())
        events = np.concatenate(accepted)[:size]
        return {
            f"p{i}_{c}": events[:, 4 * i + j]
            for i in range(len(self.masses))
            for j, c in enumerate(("x", "y", "z", "E"))
        }
