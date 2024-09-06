"""Multi-armed bandit surrogates."""

from __future__ import annotations

from attrs import define, field
from attrs.validators import instance_of
from pandas import DataFrame

from baybe.objectives.base import Objective
from baybe.objectives.single import SingleTargetObjective
from baybe.priors.base import Prior
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.bandits.beta_bernoulli import (
    BetaBernoulliMultiArmedBanditSurrogate,
)
from baybe.surrogates.base import Surrogate
from baybe.targets.binary import BinaryTarget


def _get_bandit_class(
    searchspace: SearchSpace, objective: Objective
) -> type[BetaBernoulliMultiArmedBanditSurrogate]:
    """Retrieve the appropriate bandit class for the given modelling context."""
    match searchspace, objective:
        case _, SingleTargetObjective(_target=BinaryTarget()):
            return BetaBernoulliMultiArmedBanditSurrogate
        case _:
            raise NotImplementedError(
                f"Currently, only a single target of type '{BinaryTarget.__name__}' "
                f"is supported."
            )


@define
class MultiArmedBanditSurrogate:
    """A bandit surrogate class dispatching class.

    Follows the strategy design pattern to dispatch to the appropriate bandit model.
    """

    prior: Prior = field(validator=instance_of(Prior))
    """The prior distribution assumed for each arm of the bandit."""

    _bandit_model: Surrogate | None = field(init=False, default=None, eq=False)
    """The specific bandit model to which is being dispatched."""

    def fit(
        self, searchspace: SearchSpace, objective: Objective, measurements: DataFrame
    ) -> None:
        """Instantiate an appropriate bandit model and fit it to the data."""
        cls = _get_bandit_class(searchspace, objective)
        self._bandit_model = cls(self.prior)
        self._bandit_model.fit(searchspace, objective, measurements)

    def __getattr__(self, name):
        # If the attribute is not found, try to get it from the bandit object
        return getattr(self._bandit_model, name)
