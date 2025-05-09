"""BoTorch target transformations."""

from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors import GPyTorchPosterior
from torch import Tensor


class AffinePosteriorTransform(PosteriorTransform):
    """An affine posterior transformation."""

    def __init__(self, factor: float, shift: float) -> None:
        super().__init__()
        self.factor = factor
        self.shift = shift

    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:  # noqa: D102
        return GPyTorchPosterior(posterior.distribution * self.factor + self.shift)

    def evaluate(self, Y: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()
