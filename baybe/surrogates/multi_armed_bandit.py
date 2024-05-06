"""Multi-armed bandit surrogate."""

from typing import ClassVar

import numpy as np
from attrs import define, field
from attrs.validators import ge, instance_of
from scipy.stats import beta
from torch import Tensor

from baybe.parameters import CategoricalParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate


@define
class BernoulliMultiArmedBanditSurrogate(Surrogate):
    """A multi-armed bandit model with bernoulli likelihood and a beta prior."""

    joint_posterior: ClassVar[bool] = False
    supports_transfer_learning: ClassVar[bool] = False
    # see base class.

    n_arms: int = field(validator=[instance_of(int), ge(1)])
    """ Number of arms for the bandit """

    prior_alpha_beta: np.ndarray[int] = field()
    """ Prior parameters for the bandit of shape (n_arms, 2) """

    _win_lose_counts: np.ndarray[int] = field()
    """ Storing win and lose counts for updating the prior"""

    def __attrs_post_init__(self):
        if self.prior_alpha_beta is None:
            self.prior_alpha_beta = np.ones((self.n_arms, 2))
        else:
            assert self.prior_alpha_beta.shape == (self.n_arms, 2)
        self._win_lose_counts = np.zeros_like(self.prior_alpha_beta)

    @property
    def _posterior_alpha_beta(self):
        return self._win_lose_counts + self.prior_alpha_beta

    @property
    def means(self):
        """Posterior means."""
        return beta(*self._posterior_alpha_beta.T).stats(moments="m")

    @property
    def variance(self):
        """Posterior variance."""
        return beta(*self._posterior_alpha_beta.T).stats(moments="v")

    def _posterior(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        candidate_arms = candidates.argmax(dim=-1)
        posterior_mean = self.means[candidate_arms]
        posterior_variance = self.variance[candidate_arms]
        return (Tensor(posterior_mean), Tensor(posterior_variance))

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        if len(searchspace.parameters) != 1:
            raise Exception(
                "BernoulliMultiArmedBanditSurrogate only supports one"
                " categorical parameter in the search space."
            )
        if not isinstance(searchspace.parameters[0], CategoricalParameter):
            raise Exception(
                "BernoulliMultiArmedBanditSurrogate only supports one"
                " categorical parameter in the search space."
            )

        # win counts per arm
        self._win_lose_counts[:, 0] = (train_x * train_y).sum(axis=0)
        # lose counts per arm
        self._win_lose_counts[:, 1] = (train_x * (1 - train_y)).sum(axis=0)
