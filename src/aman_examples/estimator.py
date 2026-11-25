"""Combination of TensorWaves estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorwaves.interface import Estimator

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorwaves.interface import ParameterValue


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
