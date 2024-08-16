"""Multi-armed bandit surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from attrs import define, field
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import GetSampler
from torch.distributions import Beta  # TODO: how to import this?!

from baybe.exceptions import IncompatibleSearchSpaceError, ModelNotTrainedError
from baybe.parameters import CategoricalParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.priors import BetaPrior
from baybe.surrogates.base import Surrogate

if TYPE_CHECKING:
    from botorch.posteriors import TorchPosterior
    from torch import Tensor


@define
class BernoulliMultiArmedBanditSurrogate(Surrogate):
    """A multi-armed bandit model with Bernoulli likelihood and beta prior."""

    joint_posterior: ClassVar[bool] = False
    # See base class.

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    prior: BetaPrior = field(factory=lambda: BetaPrior(1, 1))
    """Beta prior parameters. By default, configured to produce a uniform prior."""

    _win_lose_counts: Tensor | None = field(init=False, default=None, eq=False)
    """Sufficient statistics of the likelihood model."""

    @property
    def maximum_a_posteriori_per_arm(self) -> Tensor:
        """Maximum a posteriori for each arm. Returning nan for arms without a mode."""
        from torch.distributions import Beta

        # shape: (n_arms, )
        return Beta(*self._posterior_beta_parameters.split(1, 1)).mode.squeeze()

    @property
    def _posterior_beta_parameters(self) -> Tensor:
        """The parameters of the posterior beta distribution."""
        if self._win_lose_counts is None:
            raise ModelNotTrainedError(
                f"'{self.__class__.__name__}' must be "
                "fitted to access likelihood information"
            )
        # shape: (n_arms, 2)
        return (self._win_lose_counts + self.prior.to_torch()).T

    def _posterior(self, candidates: Tensor, /) -> TorchPosterior:
        # See base class.

        from botorch.posteriors import TorchPosterior
        from torch.distributions import Beta

        beta_params_for_candidates = self._posterior_beta_parameters[
            candidates.argmax(-1)
        ]
        return TorchPosterior(Beta(*beta_params_for_candidates.split(1, -1)))

    def _fit(self, train_x: Tensor, train_y: Tensor, _: Any = None) -> None:
        # See base class.

        # TODO: Fix requirement of OHE encoding
        # TODO: Generalize to arbitrary number of categorical parameters
        if not (
            (len(self._searchspace.parameters) == 1)
            and isinstance(p := self._searchspace.parameters[0], CategoricalParameter)
            and p.encoding is CategoricalEncoding.OHE
        ):
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' currently only supports search spaces "
                f"spanned by exactly one categorical parameter using one-hot encoding."
            )
        import torch

        wins = (train_x * train_y).sum(axis=0)
        losses = (train_x * (1 - train_y)).sum(axis=0)
        self._win_lose_counts = torch.vstack([wins, losses])

    @staticmethod
    def _make_target_scaler_factory():
        """Use computational representation from binary target."""
        return None


class CustomMCSampler(MCSampler):
    """Customer sampler for beta posterior."""

    def forward(self, posterior: TorchPosterior) -> Tensor:
        """Sample the posterior."""
        samples = posterior.rsample(self.sample_shape)
        return samples


@GetSampler.register(Beta)
def get_custom_sampler(_, sample_shape):
    """Get the sampler for the beta posterior."""
    return CustomMCSampler(sample_shape=sample_shape)
