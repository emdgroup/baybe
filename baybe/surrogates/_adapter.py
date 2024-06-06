"""Adapter functionality for making BayBE surrogates BoTorch-ready."""

from collections.abc import Callable
from typing import Any

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
        surrogate: The internal surrogate model.
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
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Callable[[Posterior], Posterior] | None = None,
        **kwargs: Any,
    ) -> Posterior:
        # See base class.
        mean, var = self._surrogate.posterior(X)
        mvn = gpytorch.distributions.MultivariateNormal(mean, var)
        return GPyTorchPosterior(mvn)
