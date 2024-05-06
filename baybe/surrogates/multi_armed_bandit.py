"""Multi-armed bandit surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
from attrs import define, field
from scipy.stats import beta

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.parameters import CategoricalParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate

if TYPE_CHECKING:
    from torch import Tensor


@define
class BernoulliMultiArmedBanditSurrogate(Surrogate):
    """A multi-armed bandit model with Bernoulli likelihood and beta prior."""

    joint_posterior: ClassVar[bool] = False
    # See base class.

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    prior: np.ndarray[float] = field(
        factory=lambda: np.ones(2), converter=lambda x: np.asarray(x, float)
    )
    """Beta prior parameters. By default, configured to produce a uniform prior."""

    _win_lose_counts: Optional[np.ndarray[int]] = field(init=False, default=None)
    """Sufficient statistics of the trained model (i.e., win and lose counts)."""

    @prior.validator
    def _validate_prior(self, attribute, value) -> None:
        if value.shape != (2,):
            raise ValueError(f"The shape of '{attribute.name}' must be (2,).")
        if not np.all(value > 0.0):
            raise ValueError(
                f"Both values in '{attribute.name}' must be strictly positive."
            )

    @property
    def _posterior_beta_parameters(self) -> np.ndarray[float]:
        """The parameters of the posterior beta distribution."""
        return self._win_lose_counts + self.prior

    @property
    def means(self) -> np.ndarray[float]:
        """Posterior means of the bandit arms."""
        return beta(*self._posterior_beta_parameters.T).mean()

    @property
    def variances(self) -> np.ndarray[float]:
        """Posterior variances of the bandit arms."""
        return beta(*self._posterior_beta_parameters.T).var()

    def _posterior(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        # See base class.

        import torch

        candidate_arms = candidates.argmax(dim=-1)
        posterior_mean = self.means[candidate_arms]
        posterior_variance = self.variances[candidate_arms]
        return torch.tensor(posterior_mean), torch.tensor(posterior_variance)

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.

        if not (
            (len(searchspace.parameters) == 1)
            and isinstance(p := searchspace.parameters[0], CategoricalParameter)
            and p.encoding is CategoricalEncoding.OHE
        ):
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' currently only supports search spaces "
                f"spanned by exactly one categorical parameter using one-hot encoding."
            )

        wins = (train_x * train_y).sum(axis=0)
        losses = (train_x * (1 - train_y)).sum(axis=0)
        self._win_lose_counts = np.vstack([wins, losses])
