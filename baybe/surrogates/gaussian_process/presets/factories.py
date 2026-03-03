"""Preset component factories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define
from typing_extensions import override

from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.mean import MeanFactoryProtocol

if TYPE_CHECKING:
    from gpytorch.means import Mean as GPyTorchMean
    from torch import Tensor


@define
class LazyConstantMeanFactory(MeanFactoryProtocol):
    """A factory providing constant mean functions using lazy loading."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchMean:
        from gpytorch.means import ConstantMean

        return ConstantMean()
