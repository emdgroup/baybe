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
    """A multi-armed bandit model with bernoulli likelihood and a beta prior."""

    joint_posterior: ClassVar[bool] = False
    supports_transfer_learning: ClassVar[bool] = False
    # see base class.

    prior_alpha_beta: np.ndarray[float] = field(
        factory=lambda: np.ones(2), converter=lambda x: np.asarray(x, float)
    )
    """ Prior parameters for the bandit """

    _win_lose_counts: Optional[np.ndarray[int]] = field(init=False, default=None)
    """ Storing win and lose counts for updating the prior"""

    @prior_alpha_beta.validator
    def _validate_prior_alpha_beta(self, attribute, value) -> None:
        if value.shape != (2,):
            raise ValueError(f"The shape of '{attribute.name}' must be (2,).")
        if not np.all(value > 0.0):
            raise ValueError(
                f"Both values in '{attribute.name}' must be strictly positive."
            )

    @property
    def _posterior_alpha_beta(self) -> np.ndarray[float]:
        return self._win_lose_counts + self.prior_alpha_beta

    @property
    def means(self) -> np.ndarray[float]:
        """Posterior means."""
        return beta(*self._posterior_alpha_beta.T).mean()

    @property
    def variances(self) -> np.ndarray[float]:
        """Posterior variance."""
        return beta(*self._posterior_alpha_beta.T).var()

    def _posterior(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        import torch

        candidate_arms = candidates.argmax(dim=-1)
        posterior_mean = self.means[candidate_arms]
        posterior_variance = self.variances[candidate_arms]
        return torch.tensor(posterior_mean), torch.tensor(posterior_variance)

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
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
