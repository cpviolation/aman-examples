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

        time_range: Restrict the decay times to [``t_min``, ``t_max``) (truncated
            exponential); unrestricted by default.
    """

    def __init__(
        self,
        masses: tuple[float, float, float, float],
        tau: float = 1.0,
        variables: tuple[str, str, str] = ("m2_kspim", "m2_kspip", "t"),
        time_range: tuple[float, float] = (0.0, np.inf),
    ) -> None:
        self.dalitz = DalitzPlotGenerator(masses, variables[:2])
        self.tau = float(tau)
        self.variables = variables
        self.time_range = (float(time_range[0]), float(time_range[1]))
        if not 0 <= self.time_range[0] < self.time_range[1]:
            msg = f"Invalid decay-time range {time_range}"
            raise ValueError(msg)

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        sample = dict(self.dalitz.generate(size, rng))
        t_min, t_max = self.time_range
        # inverse of the cumulative distribution of the truncated exponential
        fraction = -np.expm1(-(t_max - t_min) / self.tau)
        u = rng(size=size)
        sample[self.variables[2]] = t_min - self.tau * np.log1p(-u * fraction)
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


class PairPhaseSpaceGenerator(DataGenerator):
    r"""Four-body phase space as :math:`D \to (12)(34)`, importance-sampled in the pair masses.

    Uses :math:`d\Phi_4 \propto p^*\, p_{12}\, p_{34}\; dm_{12}\, dm_{34}\, d\Omega^*\,
    d\Omega_{12}\, d\Omega_{34}`, with :math:`p^*` the momentum of the pairs in the
    :math:`D` rest frame and :math:`p_{12}`, :math:`p_{34}` the momenta in the pair rest
    frames. The pair masses are drawn from mixtures of a flat density and of Cauchy
    (non-relativistic Breit-Wigner) densities centred on known narrow resonances, then
    accepted with probability :math:`\propto p^*\, p_{12}\, p_{34}`, and the pairs decay
    isotropically. The events are therefore distributed as phase space times
    :meth:`density_ratio`, which is known for each event: to generate events from an
    intensity :math:`I`, use :class:`ImportanceSampledFunction`, :math:`I/r`, which
    peaks much less than :math:`I` when the resonances dominate the peaks of :math:`I`.

    The sample has the same columns as :class:`NBodyPhaseSpaceGenerator` (``p<i>_x``,
    ``p<i>_y``, ``p<i>_z``, ``p<i>_E``), in the order of ``masses``.

    Args:
        masses: Masses of the four final-state particles in GeV.

        parent_mass: Mass of the decaying particle in GeV.

        pairs: Indices of the particles of the two pairs.

        peaks: For each pair, a sequence of ``(mass, width, fraction)``: a Cauchy density
            of that mass and width (GeV) is mixed into the pair-mass density with that
            fraction. The fractions of a pair sum to less than 1, the rest is flat.
    """

    def __init__(
        self,
        masses: tuple[float, float, float, float],
        parent_mass: float,
        pairs: tuple[tuple[int, int], tuple[int, int]] = ((0, 1), (2, 3)),
        peaks: tuple[
            tuple[tuple[float, float, float], ...],
            tuple[tuple[float, float, float], ...],
        ] = ((), ()),
    ) -> None:
        self.masses = np.array(masses, dtype=float)
        self.parent_mass = float(parent_mass)
        self.pairs = pairs
        if sorted(i for pair in pairs for i in pair) != [0, 1, 2, 3]:
            msg = f"The pairs {pairs} are not a partition of the four particles"
            raise ValueError(msg)
        pair_masses = [self.masses[list(pair)].sum() for pair in pairs]
        if self.parent_mass <= sum(pair_masses):
            msg = "The decay is kinematically forbidden"
            raise ValueError(msg)
        self.ranges = [
            (pair_masses[k], self.parent_mass - pair_masses[1 - k]) for k in range(2)
        ]
        self.peaks = [tuple(tuple(map(float, p)) for p in peaks[k]) for k in range(2)]
        for k in range(2):
            if sum(f for _, _, f in self.peaks[k]) >= 1:
                msg = f"The peak fractions of pair {k} sum to 1 or more"
                raise ValueError(msg)
        # maximum of p* p12 p34 on a grid (smooth, zero on the boundary)
        grid = [np.linspace(*r, 401) for r in self.ranges]
        m12, m34 = np.meshgrid(*grid, indexing="ij")
        self.max_weight = 1.02 * float(self._phase_space_weight(m12, m34).max())
        # the accept-reject on p* p12 p34 changes the normalisation of the product of
        # the pair-mass densities: phase-space average of that product, by quadrature
        nodes = [self._quadrature_nodes(k) for k in range(2)]
        m12, m34 = np.meshgrid(*nodes, indexing="ij")
        weight = self._phase_space_weight(m12, m34)
        product = self._pair_density(0, m12) * self._pair_density(1, m34)
        self._normalisation = 1 / (
            np.trapezoid(np.trapezoid(weight * product, nodes[1]), nodes[0])
            / np.trapezoid(np.trapezoid(weight, nodes[1]), nodes[0])
        )

    def _quadrature_nodes(self, k: int) -> np.ndarray:
        lo, hi = self.ranges[k]
        nodes = [np.linspace(lo, hi, 2001)]
        for mass, width, _ in self.peaks[k]:
            nodes.append(np.linspace(mass - 30 * width, mass + 30 * width, 1201))
        nodes_array = np.unique(np.concatenate(nodes))
        return nodes_array[(nodes_array >= lo) & (nodes_array <= hi)]

    def _phase_space_weight(self, m12: np.ndarray, m34: np.ndarray) -> np.ndarray:
        (a, b), (c, d) = self.pairs
        p_star = _breakup_momentum(
            np.broadcast_to(self.parent_mass, np.shape(m12)), m12, m34
        )
        p12 = _breakup_momentum(
            m12, np.full(np.shape(m12), self.masses[a]), self.masses[b]
        )
        p34 = _breakup_momentum(
            m34, np.full(np.shape(m34), self.masses[c]), self.masses[d]
        )
        return np.where(m12 + m34 < self.parent_mass, p_star * p12 * p34, 0.0)

    def _cauchy_bounds(self, k: int, mass: float, width: float) -> tuple[float, float]:
        lo, hi = self.ranges[k]
        return np.arctan((lo - mass) / (width / 2)), np.arctan(
            (hi - mass) / (width / 2)
        )

    def _pair_density(self, k: int, m: np.ndarray) -> np.ndarray:
        """Density of the proposal for the mass of pair ``k``, relative to flat."""
        lo, hi = self.ranges[k]
        ratio = 1 - sum(f for _, _, f in self.peaks[k]) + 0 * m
        for mass, width, fraction in self.peaks[k]:
            a, b = self._cauchy_bounds(k, mass, width)
            cauchy = (width / 2) / ((m - mass) ** 2 + (width / 2) ** 2) / (b - a)
            ratio += fraction * (hi - lo) * cauchy
        return ratio

    def _sample_pair_mass(
        self, k: int, size: int, rng: RealNumberGenerator
    ) -> np.ndarray:
        lo, hi = self.ranges[k]
        m = rng(size=size, min_value=lo, max_value=hi)
        choice = rng(size=size)
        u = rng(size=size)
        edge = 0.0
        for mass, width, fraction in self.peaks[k]:
            a, b = self._cauchy_bounds(k, mass, width)
            selected = (choice >= edge) & (choice < edge + fraction)
            m = np.where(selected, mass + width / 2 * np.tan(a + u * (b - a)), m)
            edge += fraction
        return m

    def density_ratio(self, sample: DataSample) -> np.ndarray:
        """Density of the generated events relative to phase space, per event.

        Normalised so that its average over phase space is 1: the average of :math:`f/r`
        over generated events estimates the phase-space average of :math:`f`.
        """
        return (
            self._normalisation
            * self._pair_density(0, self._pair_mass(sample, 0))
            * self._pair_density(1, self._pair_mass(sample, 1))
        )

    def _pair_mass(self, sample: DataSample, k: int) -> np.ndarray:
        p = sum(
            np.column_stack([sample[f"p{i}_{c}"] for c in ("x", "y", "z", "E")])
            for i in self.pairs[k]
        )
        return np.sqrt(np.clip(p[:, 3] ** 2 - np.sum(p[:, :3] ** 2, axis=1), 0, None))

    def _decay(
        self, m12: np.ndarray, m34: np.ndarray, rng: RealNumberGenerator
    ) -> np.ndarray:
        """Four-momenta (N, 4 particles, 4 components) for given pair masses."""
        size = len(m12)

        def isotropic(p: np.ndarray) -> np.ndarray:
            cos_theta = 2 * rng(size=size) - 1
            phi = 2 * np.pi * rng(size=size)
            sin_theta = np.sqrt(1 - cos_theta**2)
            return p[:, None] * np.column_stack([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta,
            ])

        momenta = np.empty((size, 4, 4))
        p_star = isotropic(_breakup_momentum(np.full(size, self.parent_mass), m12, m34))
        for (i, j), m, sign in ((self.pairs[0], m12, 1), (self.pairs[1], m34, -1)):
            mi, mj = np.full(size, self.masses[i]), np.full(size, self.masses[j])
            q = isotropic(_breakup_momentum(m, mi, mj))
            beta = sign * p_star / np.sqrt(np.sum(p_star**2, axis=1) + m**2)[:, None]
            momenta[:, i] = _boost(
                np.column_stack([q, np.sqrt(np.sum(q**2, 1) + mi**2)]), beta
            )
            momenta[:, j] = _boost(
                np.column_stack([-q, np.sqrt(np.sum(q**2, 1) + mj**2)]), beta
            )
        return momenta

    def generate(self, size: int, rng: RealNumberGenerator) -> DataSample:
        m12_list: list[np.ndarray] = []
        m34_list: list[np.ndarray] = []
        count = 0
        while count < size:
            n = 2 * (size - count) + 1000
            m12 = self._sample_pair_mass(0, n, rng)
            m34 = self._sample_pair_mass(1, n, rng)
            keep = rng(size=n, max_value=self.max_weight) < self._phase_space_weight(
                m12, m34
            )
            m12_list.append(m12[keep])
            m34_list.append(m34[keep])
            count += int(keep.sum())
        m12 = np.concatenate(m12_list)[:size]
        m34 = np.concatenate(m34_list)[:size]
        momenta = self._decay(m12, m34, rng)
        return {
            f"p{i}_{c}": momenta[:, i, j]
            for i in range(4)
            for j, c in enumerate(("x", "y", "z", "E"))
        }


class ImportanceSampledFunction:
    r"""Intensity divided by the density ratio of an importance-sampled domain generator.

    For use with :class:`~tensorwaves.data.IntensityDistributionGenerator` and a
    generator such as :class:`PairPhaseSpaceGenerator`, whose events are distributed as
    phase space times :math:`r(x)`: accepting them with probability :math:`\propto
    I(x)/r(x)` gives events distributed as :math:`I(x)` in phase space. The transformer,
    if any, is applied here, so that :math:`r` is computed from the generated
    four-momenta: pass no ``domain_transformer`` to the
    `~tensorwaves.data.IntensityDistributionGenerator`.

    Args:
        function: The intensity.

        generator: The domain generator, with a ``density_ratio(sample)`` method.

        transformer: Optional transformation of the generated sample into the input of
            ``function``.
    """

    def __init__(self, function, generator, transformer=None) -> None:
        self.function = function
        self.generator = generator
        self.transformer = transformer

    def __call__(self, data: DataSample) -> np.ndarray:
        inputs = data if self.transformer is None else self.transformer(data)
        return np.asarray(self.function(inputs)) / self.generator.density_ratio(data)
