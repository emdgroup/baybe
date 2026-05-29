"""Mean factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from attrs import define
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentFactoryProtocol,
    PlainGPComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.means import Mean as GPyTorchMean

    MeanFactoryProtocol = GPComponentFactoryProtocol[GPyTorchMean]
    PlainMeanFactory = PlainGPComponentFactory[GPyTorchMean]
else:
    # At runtime, we avoid loading GPyTorch eagerly for performance reasons
    MeanFactoryProtocol = GPComponentFactoryProtocol[Any]
    """A factory protocol for Gaussian process mean functions."""

    PlainMeanFactory = PlainGPComponentFactory[Any]
    """A trivial factory returning a fixed, pre-defined mean function."""


@define
class LazyConstantMeanFactory(MeanFactoryProtocol):
    """A factory providing constant mean functions using lazy loading."""

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> GPyTorchMean:
        from gpytorch.means import ConstantMean

        return ConstantMean()
