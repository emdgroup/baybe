"""BoTorch objectives."""

from botorch.acquisition.objective import MCAcquisitionObjective
from torch import Tensor

from baybe.utils.basic import compose


class ChainedMCObjective(MCAcquisitionObjective):
    """A chained Monte Carlo objective."""

    def __init__(self, *objectives: MCAcquisitionObjective) -> None:
        super().__init__()
        self.objectives = objectives

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:  # noqa: D102
        return compose(*(o.forward for o in self.objectives))(samples, X)
