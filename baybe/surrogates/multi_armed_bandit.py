"""Multi-armed bandit surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
from attrs import define, field
from scipy.stats import beta

from baybe.exceptions import IncompatibleSearchSpaceError, NotFitError
from baybe.kernels.priors import BetaPrior
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

    prior: BetaPrior = field(default=BetaPrior(1, 1))
    """Beta prior parameters. By default, configured to produce a uniform prior."""

    _win_lose_counts: Optional[np.ndarray[int]] = field(
        init=False, default=None, eq=False
    )
    """Sufficient statistics of the trained model (i.e., win and lose counts)."""

    @property
    def _posterior_beta_parameters(self) -> np.ndarray[float]:
        """The parameters of the posterior beta distribution."""
        if self._win_lose_counts is None:
            raise NotFitError(
                f"'{self.__class__.__name__}' must be "
                "fitted to access likelihood information"
            )
            # TODO: this could be removed when the number of arms could be inferred
        return self._win_lose_counts + self.prior.numpy()

    @property
    def means(self) -> np.ndarray[float]:
        """Posterior means of the bandit arms."""
        return beta(*self._posterior_beta_parameters).mean()

    @property
    def variances(self) -> np.ndarray[float]:
        """Posterior variances of the bandit arms."""
        return beta(*self._posterior_beta_parameters).var()

    def _posterior(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        # See base class.

        import torch

        candidate_arms = candidates.argmax(dim=-1)
        posterior_mean = self.means[candidate_arms]
        posterior_variance = self.variances[candidate_arms]
        return torch.tensor(posterior_mean), torch.tensor(posterior_variance)

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.

        # TODO: Fix requirement of OHE encoding
        # TODO: Generalize to arbitrary number of categorical parameters
        if not (
            (len(searchspace.parameters) == 1)
            and isinstance(p := searchspace.parameters[0], CategoricalParameter)
            and p.encoding is CategoricalEncoding.OHE
        ):
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' currently only supports search spaces "
                f"spanned by exactly one categorical parameter using one-hot encoding."
            )

        # TODO: Incorporate training target validation at the appropriate place in
        #   the BayBE ecosystem.
        wins = (train_x * train_y).sum(axis=0)
        losses = (train_x * (1 - train_y)).sum(axis=0)
        self._win_lose_counts = np.vstack([wins, losses])
