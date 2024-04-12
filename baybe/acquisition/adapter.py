"""Adapter for making BoTorch's acquisition functions work with BayBE models."""

from typing import Any, Callable, Optional

import gpytorch.distributions
from botorch.models.gpytorch import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from torch import Tensor

from baybe.surrogates.base import Surrogate


class AdapterModel(Model):
    """A BoTorch model that uses a BayBE surrogate model for posterior computation.

    Can be used, for example, as an adapter layer for making a BayBE
    surrogate model usable in conjunction with BoTorch acquisition functions.

    Args:
        surrogate: The internal surrogate model
    """

    def __init__(self, surrogate: Surrogate):
        super().__init__()
        self._surrogate = surrogate

    @property
    def num_outputs(self) -> int:  # noqa: D102
        # See base class.
        # TODO: So far, the usage is limited to single-output models.
        return 1

    def posterior(  # noqa: D102
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        # See base class.
        mean, var = self._surrogate.posterior(X)
        mvn = gpytorch.distributions.MultivariateNormal(mean, var)
        return GPyTorchPosterior(mvn)
