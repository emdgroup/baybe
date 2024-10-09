"""Multi-armed bandit surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, ClassVar

from attrs import define, field

from baybe.exceptions import IncompatibleSearchSpaceError, ModelNotTrainedError
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.priors import BetaPrior
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.targets.binary import _FAILURE_VALUE_COMP, _SUCCESS_VALUE_COMP
from baybe.utils.plotting import to_string
from baybe.utils.random import temporary_seed

if TYPE_CHECKING:
    from botorch.models.model import Model
    from botorch.posteriors import TorchPosterior
    from torch import Tensor


@define
class BetaBernoulliMultiArmedBanditSurrogate(Surrogate):
    """A multi-armed bandit model with Bernoulli likelihood and beta prior."""

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    prior: BetaPrior = field(factory=lambda: BetaPrior(1, 1))
    """The beta prior for the win rates of the bandit arms. Uniform by default."""

    # TODO: type should be `torch.Tensor | None` but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _win_lose_counts = field(init=False, default=None, eq=False)
    """Sufficient statistics for the Bernoulli likelihood model: (# wins, # losses)."""

    def posterior_modes(self) -> Tensor:
        """Compute the posterior mode win rates for all arms.

        Returns:
            A tensor of length ``N`` containing the posterior mode estimates of the win
            rates, where ``N`` is the number of bandit arms.
            Contains ``float('nan')`` for arms with undefined mode.
        """
        from torch.distributions import Beta

        return Beta(*self._posterior_beta_parameters()).mode

    def posterior_means(self) -> Tensor:
        """Compute the posterior mean win rates for all arms.

        Returns:
            A tensor of length ``N`` containing the posterior mean estimates of the win
            rates, where ``N`` is the number of bandit arms.
        """
        from torch.distributions import Beta

        return Beta(*self._posterior_beta_parameters()).mean

    def _posterior_beta_parameters(self) -> Tensor:
        """Compute the posterior parameters of the beta distribution.

        Raises:
            ModelNotTrainedError: If accessed before the model was trained.

        Returns:
            A tensors of shape ``(2, N)`` containing the posterior beta parameters,
            where ``N`` is the number of bandit arms.
        """
        if self._win_lose_counts is None:
            raise ModelNotTrainedError(
                f"'{self.__class__.__name__}' must be trained before posterior "
                f"information can be accessed."
            )

        import torch

        return self._win_lose_counts + torch.tensor(
            [self.prior.alpha, self.prior.beta]
        ).unsqueeze(-1)

    def to_botorch(self) -> Model:  # noqa: D102
        # See base class.

        # We register the sampler on the fly to avoid eager loading of torch

        from botorch.sampling.base import MCSampler
        from botorch.sampling.get_sampler import GetSampler
        from torch.distributions import Beta

        class CustomMCSampler(MCSampler):
            """Customer sampler for beta posterior."""

            def forward(self, posterior: TorchPosterior) -> Tensor:
                """Sample the posterior."""
                with temporary_seed(self.seed):
                    samples = posterior.rsample(self.sample_shape)
                return samples

        @GetSampler.register(Beta)
        def get_custom_sampler(_, sample_shape, seed: int | None = None):
            """Get the sampler for the beta posterior."""
            return CustomMCSampler(sample_shape=sample_shape, seed=seed)

        return super().to_botorch()

    @staticmethod
    def _make_input_scaler_factory():
        # See base class.
        #
        # Due to enforced one-hot encoding, no input scaling is needed.
        return None

    @staticmethod
    def _make_target_scaler_factory():
        # See base class.
        #
        # We directly use the binary computational representation from the target.
        return None

    def _posterior(self, candidates: Tensor, /) -> TorchPosterior:
        # See base class.

        from botorch.posteriors import TorchPosterior
        from torch.distributions import Beta

        beta_params_for_candidates = self._posterior_beta_parameters().T[
            candidates.argmax(-1)
        ]
        return TorchPosterior(Beta(*beta_params_for_candidates.split(1, -1)))

    def _fit(self, train_x: Tensor, train_y: Tensor, _: Any = None) -> None:
        # See base class.

        # TODO: Fix requirement of OHE encoding. This is likely a long-term goal since
        #   probably requires decoupling parameter from encodings and associating the
        #   latter with the surrogate.
        # TODO: Generalize to arbitrary number of categorical parameters
        match self._searchspace:
            case SearchSpace(
                parameters=[CategoricalParameter(encoding=CategoricalEncoding.OHE)]
            ):
                pass
            case _:
                raise IncompatibleSearchSpaceError(
                    f"'{self.__class__.__name__}' currently only supports search "
                    f"spaces spanned by exactly one categorical parameter using "
                    f"one-hot encoding."
                )

        import torch

        # IMPROVE: The training inputs/targets can actually be represented as
        #   integers/boolean values but the transformation pipeline currently
        #   converts them float. Potentially, this can be improved by making
        #   the type conversion configurable.

        wins = (train_x * (train_y == float(_SUCCESS_VALUE_COMP))).sum(dim=0)
        losses = (train_x * (train_y == float(_FAILURE_VALUE_COMP))).sum(dim=0)
        self._win_lose_counts = torch.vstack([wins, losses]).to(torch.int)

    def __str__(self) -> str:
        fields = [to_string("Prior", self.prior, single_line=True)]
        return to_string(super().__str__(), *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
