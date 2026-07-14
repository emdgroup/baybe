"""Residual-learning Gaussian process surrogate for transfer learning."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, cast

from attrs import define, evolve, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibleObjectiveError
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.transfer_learning.base import _SourceTargetTransferSurrogate

if TYPE_CHECKING:
    import pandas as pd
    from botorch.posteriors import GPyTorchPosterior, Posterior
    from torch import Tensor

    from baybe.objectives.base import Objective
    from baybe.searchspace.core import SearchSpace


@define
class ResidualTransferSurrogate(_SourceTargetTransferSurrogate):
    """A transfer learning surrogate that learns the target as a residual.

    Fits a single-task source Gaussian process on the source subset, then a single-task
    residual Gaussian process on the residuals of the target data with respect to the
    source GP's posterior mean (computed in original target units). Predictions are the
    sum of the source and residual posterior means.

    Note:
        Only a single source and a single target task are currently supported.
    """

    propagate_source_uncertainty: bool = field(
        default=False, validator=instance_of(bool)
    )
    """Whether to add the source GP's variance to the residual GP's variance.

    If ``False``, the source model is treated as a fixed mean function and only the
    residual GP contributes to the predictive variance. If ``True``, source and
    residual GPs are treated as independent and their variances are summed.
    """

    _residual_gp: GaussianProcessSurrogate | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The single-task GP trained on the residuals. Available after fitting."""

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Reject incompatible objectives, then run the shared fitting logic.

        Args:
            train_x: Computational-representation inputs prepared by the base class.
            train_y: Target values prepared by the base class.

        Raises:
            IncompatibleObjectiveError: If the objective is a
                :class:`~baybe.objectives.desirability.DesirabilityObjective`. The
                residual is computed by subtracting a single source posterior mean from
                the raw target column(s), which is only meaningful for a single modeled
                target and not for the aggregated, multi-target desirability value.
        """
        from baybe.objectives.desirability import DesirabilityObjective

        assert self._objective is not None  # provided by base class
        if isinstance(self._objective, DesirabilityObjective):
            raise IncompatibleObjectiveError(
                f"'{self.__class__.__name__}' does not support "
                f"'{DesirabilityObjective.__name__}', because residuals are computed "
                f"per target and cannot be formed from an aggregated desirability "
                f"value. Please use a single-target objective instead."
            )
        super()._fit(train_x, train_y)

    @override
    def _fit_target(
        self,
        reduced_searchspace: SearchSpace,
        objective: Objective,
        source_measurements: pd.DataFrame,
        target_measurements: pd.DataFrame,
    ) -> None:
        """Fit the residual GP on the target residuals w.r.t. the source posterior mean.

        Args:
            reduced_searchspace: The task-free search space for the residual GP.
            objective: The objective (a single modeled quantity after replication).
            source_measurements: The measurements belonging to the source task.
            target_measurements: The measurements belonging to the target task.
        """
        assert self._source_gp is not None  # set by the base class

        # Source posterior mean at the target inputs, in original target units.
        source_posterior = cast(
            "GPyTorchPosterior", self._source_gp.posterior(target_measurements)
        )
        source_mean = source_posterior.mean.detach().cpu().numpy().reshape(-1)

        residual_measurements = target_measurements.copy()
        for target in objective.targets:
            residual_measurements[target.name] = (
                target_measurements[target.name].to_numpy() - source_mean
            )

        self._residual_gp = evolve(self.base_surrogate)
        self._residual_gp.fit(reduced_searchspace, objective, residual_measurements)

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Return the combined source-plus-residual posterior on stripped candidates.

        Args:
            candidates_comp_scaled: Candidate points in the computational representation
                of the full search space (including the task column).

        Returns:
            A posterior whose mean is the sum of the source and residual posterior
            means and whose covariance is the residual covariance (plus the source
            covariance if ``propagate_source_uncertainty`` is set).
        """
        from botorch.posteriors import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        assert self._source_gp is not None  # set during fitting
        assert self._residual_gp is not None  # set during fitting

        reduced_candidates = self._strip_task(candidates_comp_scaled)
        source_posterior = cast(
            "GPyTorchPosterior", self._source_gp._posterior(reduced_candidates)
        )
        residual_posterior = cast(
            "GPyTorchPosterior", self._residual_gp._posterior(reduced_candidates)
        )

        mean = source_posterior.mean + residual_posterior.mean
        covariance = residual_posterior.distribution.lazy_covariance_matrix
        if self.propagate_source_uncertainty:
            covariance = (
                covariance + source_posterior.distribution.lazy_covariance_matrix
            )

        combined = MultivariateNormal(mean.squeeze(-1), covariance)
        return GPyTorchPosterior(combined)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
