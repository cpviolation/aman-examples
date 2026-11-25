"""TensorWaves estimators: combinations, and a fast likelihood for linear amplitudes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tensorwaves.data.transform import SympyDataTransformer
from tensorwaves.interface import Estimator

if TYPE_CHECKING:
    from collections.abc import Mapping

    import sympy as sp
    from tensorwaves.interface import DataSample, ParameterValue


class SimultaneousEstimator(Estimator):
    r"""Sum of estimators, e.g. the negative log-likelihoods of several samples (:math:`D^0` and :math:`\bar{D}^0` decays) with shared parameters.

    Each estimator receives all parameters; those it does not depend on are ignored.
    """

    def __init__(self, *estimators: Estimator) -> None:
        if not estimators:
            msg = "At least one estimator is needed"
            raise ValueError(msg)
        self.estimators = estimators

    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float:
        return sum(float(e(parameters)) for e in self.estimators)

    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
        total: dict[str, ParameterValue] = {}
        for estimator in self.estimators:
            for name, value in estimator.gradient(parameters).items():
                total[name] = total.get(name, 0) + value
        return total


class LinearAmplitudeNLL(Estimator):
    r"""Unbinned negative log-likelihood of :math:`|A|^2` with :math:`A` linear in the free parameters.

    With :math:`A = \sum_k c_k B_k + B_0`, where the basis amplitudes :math:`B_k` do not
    depend on the free complex parameters :math:`c_k` (e.g. from
    :meth:`.AmpGenModel.linear_amplitude`), the normalisation integral is a quadratic
    form,

    .. math::

        \int |A|^2 = \tilde{c}^\dagger M \tilde{c}, \qquad
        M_{kl} = \frac{1}{n} \sum_{j=1}^n w_j B_k^*(x_j) B_l(x_j), \qquad
        \tilde{c} = (c_1, \ldots, c_K, 1),

    computed once from the ``phsp`` sample (with its ``"weights"`` column :math:`w_j`,
    if any, as in :class:`~tensorwaves.estimator.UnbinnedNLL`). Each call then costs
    :math:`K^2` operations for the normalisation and :math:`K N` for the data, whatever
    the size of the ``phsp`` sample, instead of evaluating the intensity on all ``phsp``
    events. The value is that of :class:`~tensorwaves.estimator.UnbinnedNLL` for the
    same samples, up to rounding.

    Each call takes the values of the free parameters from ``parameters`` (keyed by
    symbol name), or keeps the previous ones (initially ``initial``, default 1); other
    names are ignored. ``chunk_size`` events of ``phsp`` are evaluated at a time.
    """

    def __init__(
        self,
        basis: Mapping[sp.Symbol, sp.Expr],
        constant: sp.Expr,
        data: DataSample,
        phsp: DataSample,
        *,
        initial: Mapping[str, complex] | None = None,
        chunk_size: int = 100_000,
    ) -> None:
        import jax  # ruff: ignore[import-outside-top-level]
        import jax.numpy as jnp  # ruff: ignore[import-outside-top-level]

        jax.config.update("jax_enable_x64", True)  # ruff: ignore[boolean-positional-value-in-call]
        self.parameter_names = [c.name for c in basis]
        expressions = {f"B{k}": e for k, e in enumerate([*basis.values(), constant])}
        transformer = SympyDataTransformer.from_sympy(expressions, backend="jax")

        def evaluate(sample: DataSample) -> np.ndarray:
            size = len(next(iter(sample.values())))
            columns = transformer({k: v for k, v in sample.items() if k != "weights"})
            return np.stack([
                np.broadcast_to(np.asarray(columns[name], dtype=complex), size)
                for name in expressions
            ])

        size = len(next(iter(phsp.values())))
        weights = np.asarray(phsp.get("weights", np.ones(size)), dtype=float)
        gram = np.zeros((len(expressions),) * 2, dtype=complex)
        for start in range(0, size, chunk_size):
            chunk = slice(start, start + chunk_size)
            b = evaluate({k: np.asarray(v)[chunk] for k, v in phsp.items()})
            gram += (b.conj() * weights[chunk]) @ b.T
        self.gram = gram / size
        data_basis = jnp.asarray(evaluate(data))
        gram_j = jnp.asarray(self.gram)
        n_data = data_basis.shape[1]

        def nll(c: jax.Array) -> jax.Array:
            ct = jnp.concatenate([c, jnp.ones(1, dtype=c.dtype)])
            norm = jnp.real(ct.conj() @ gram_j @ ct)
            return n_data * jnp.log(norm) - jnp.sum(
                jnp.log(jnp.abs(ct @ data_basis) ** 2)
            )

        self.__nll = jax.jit(nll)
        self.__grad = jax.jit(jax.grad(nll))
        initial = initial or {}
        self.__values = np.array([
            complex(initial.get(name, 1)) for name in self.parameter_names
        ])

    def __update(self, parameters: Mapping[str, ParameterValue]) -> np.ndarray:
        for k, name in enumerate(self.parameter_names):
            if name in parameters:
                self.__values[k] = complex(parameters[name])  # ty:ignore[invalid-argument-type]
        return self.__values

    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float:
        return float(self.__nll(self.__update(parameters)))

    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
        r"""Derivatives :math:`\partial/\partial\mathrm{Re}\,c + i\,\partial/\partial\mathrm{Im}\,c`, as in TensorWaves."""
        # jax.grad gives the conjugated Wirtinger derivative, as in TensorWaves
        grad = np.asarray(self.__grad(self.__update(parameters))).conj()
        return dict(zip(self.parameter_names, grad.tolist(), strict=True))
