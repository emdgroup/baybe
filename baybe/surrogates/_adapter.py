"""Adapter functionality for making BayBE surrogates BoTorch-ready."""

from collections.abc import Callable
from typing import Any

from botorch.models.gpytorch import Model
from botorch.posteriors import Posterior
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
    def num_outputs(self) -> int:
        if (objective := self._surrogate._objective) is None:
            raise RuntimeError(
                "The model needs to be fitted to a specific context first."
            )
        return objective.n_outputs

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Callable[[Posterior], Posterior] | None = None,
        **kwargs: Any,
    ) -> Posterior:
        if (
            (output_indices is not None)
            or observation_noise
            or (posterior_transform is not None)
        ):
            raise NotImplementedError(
                "The optional model posterior arguments are not yet implemented."
            )
        return self._surrogate._posterior_comp(X)
