"""Mean-transfer Gaussian process surrogate for transfer learning."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from attrs import define, evolve, field
from typing_extensions import override

from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.transfer_learning.base import _SourceTargetTransferSurrogate

if TYPE_CHECKING:
    import pandas as pd
    from botorch.posteriors import Posterior
    from torch import Tensor

    from baybe.objectives.base import Objective
    from baybe.searchspace.core import SearchSpace


@define
class MeanTransferSurrogate(_SourceTargetTransferSurrogate):
    """A transfer learning surrogate that transfers a source model's posterior mean.

    Fits a single-task source Gaussian process on the source subset and a single-task
    target Gaussian process on the target subset, using the source GP's posterior mean
    as the target GP's prior mean, thereby transferring source knowledge to the target.

    Predictions are made for target points: the task column is stripped from the
    incoming candidates so that they match the reduced space of the target model, whose
    posterior is then returned.

    Note:
        Only a single source and a single target task are currently supported.
    """

    _target_gp: GaussianProcessSurrogate | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The single-task GP trained on the target data. Available after fitting."""

    @override
    def _fit_target(
        self,
        reduced_searchspace: SearchSpace,
        objective: Objective,
        source_measurements: pd.DataFrame,
        target_measurements: pd.DataFrame,
    ) -> None:
        """Fit the target GP using the source GP's posterior mean as its prior mean.

        Args:
            reduced_searchspace: The task-free search space for the target GP.
            objective: The objective (a single modeled quantity after replication).
            source_measurements: The measurements belonging to the source task.
            target_measurements: The measurements belonging to the target task.
        """
        assert self._source_gp is not None  # set by the base class
        self._target_gp = evolve(
            self.base_surrogate,
            mean_or_factory=self._source_gp.posterior_mean_function,
        )
        self._target_gp.fit(reduced_searchspace, objective, target_measurements)

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Return the target GP's posterior on the task-stripped candidates.

        Args:
            candidates_comp_scaled: Candidate points in the computational representation
                of the full search space (including the task column).

        Returns:
            The posterior of the target Gaussian process at the given candidates.
        """
        assert self._target_gp is not None  # set during fitting
        reduced_candidates = self._strip_task(candidates_comp_scaled)
        return self._target_gp._posterior(reduced_candidates)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
