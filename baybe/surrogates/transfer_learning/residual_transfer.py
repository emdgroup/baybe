"""Residual-learning Gaussian process surrogate for transfer learning."""

from __future__ import annotations

import gc
import operator
from functools import reduce
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
    """A transfer learning surrogate that learns the target as a GP boosting chain.

    Fits a sequence of Gaussian processes, where each GP corrects the residuals of all
    previous ones. Source tasks are processed in alphabetical order (the order imposed
    by :class:`~baybe.parameters.categorical.TaskParameter`):

    - The **first source GP** is fitted on the raw first-source measurements.
    - Each **subsequent source GP** is fitted on the residuals of those measurements
      w.r.t. the sum of all previous GPs in the chain.
    - The **target GP** is fitted on the residuals of the target measurements w.r.t.
      the sum of all source GPs in the chain.

    Predictions are the sum of the posterior means of all GPs in the chain.

    Cold start: if the target task has no measurements yet, the chain contains only
    source GPs and predictions are their combined posterior mean.

    Note:
        Multiple source tasks and a single target task are supported.
        Source tasks with no measurements are silently skipped.
        :class:`~baybe.objectives.desirability.DesirabilityObjective` is not supported
        because residuals are computed per raw target column.
    """

    propagate_source_uncertainty: bool = field(
        default=False, validator=instance_of(bool)
    )
    """Whether to sum the variances of all GPs in the chain.

    If ``False``, only the last GP in the chain contributes to the predictive
    variance (the earlier GPs are treated as fixed mean functions). If ``True``,
    all GPs are treated as independent and their variances are summed.
    """

    _gp_chain: tuple[GaussianProcessSurrogate, ...] = field(
        init=False, factory=tuple, eq=False, repr=False
    )
    """The fitted GP chain.

    The first entry is fitted on the raw data of the first source task. Each subsequent
    entry is fitted on the residuals of those measurements w.r.t. the sum of all
    previous GPs. The last entry covers the target task if target data is available;
    otherwise the chain contains only source GPs (cold start). Empty before fitting.
    """

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Build the GP chain from source and target measurements.

        Args:
            train_x: Computational-representation inputs prepared by the base class.
                Not used directly; measurements are re-split internally.
            train_y: Target values prepared by the base class. Not used directly.

        Raises:
            IncompatibleObjectiveError: If the objective is a
                :class:`~baybe.objectives.desirability.DesirabilityObjective`.
        """
        from baybe.objectives.desirability import DesirabilityObjective
        from baybe.surrogates.gaussian_process.core import _ModelContext

        assert self._objective is not None  # provided by base class
        assert self._searchspace is not None  # provided by base class
        assert self._measurements is not None  # provided by base class

        if isinstance(self._objective, DesirabilityObjective):
            raise IncompatibleObjectiveError(
                f"'{self.__class__.__name__}' does not support "
                f"'{DesirabilityObjective.__name__}', because residuals are computed "
                f"per target and cannot be formed from an aggregated desirability "
                f"value. Please use a single-target objective instead."
            )

        reduced_searchspace, sources, target_measurements = self._split_measurements()

        context = _ModelContext(self._searchspace, self._objective, self._measurements)
        self._numerical_indices = context.numerical_indices

        chain: list[GaussianProcessSurrogate] = []
        for i, (_, source_measurements) in enumerate(sources):
            gp = evolve(self.base_surrogate)
            measurements = (
                source_measurements
                if i == 0
                else self._make_residual_measurements(chain, source_measurements)
            )
            gp.fit(reduced_searchspace, self._objective, measurements)
            chain.append(gp)

        if not target_measurements.empty:
            gp = evolve(self.base_surrogate)
            gp.fit(
                reduced_searchspace,
                self._objective,
                self._make_residual_measurements(chain, target_measurements),
            )
            chain.append(gp)

        self._gp_chain = tuple(chain)

    @override
    def _fit_target(
        self,
        reduced_searchspace: SearchSpace,
        objective: Objective,
        target_measurements: pd.DataFrame,
    ) -> None:
        # The chain is built in _fit, which is overridden and does not call super().
        # This method is therefore never invoked; it exists only to satisfy the
        # abstract base class.
        pass

    def _make_residual_measurements(
        self,
        gps: list[GaussianProcessSurrogate],
        measurements: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return measurements with target columns replaced by chain residuals.

        Computes the sum of posterior means of all GPs in ``gps`` at the input
        points of ``measurements`` and subtracts it from the target column(s).

        Args:
            gps: Fitted GPs whose posterior means are summed to form the baseline.
            measurements: Raw measurements to residualise.

        Returns:
            A copy of ``measurements`` with target column(s) replaced by residuals.
        """
        assert self._objective is not None

        stack_mean = sum(
            cast("GPyTorchPosterior", gp.posterior(measurements))
            .mean.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            for gp in gps
        )
        residual_measurements = measurements.copy()
        for target in self._objective.targets:
            residual_measurements[target.name] = (
                measurements[target.name].to_numpy() - stack_mean
            )
        return residual_measurements

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Return the chain posterior on task-stripped candidates.

        Args:
            candidates_comp_scaled: Candidate points in the computational representation
                of the full search space (including the task column).

        Returns:
            A posterior whose mean is the sum of all GP means in the chain and whose
            covariance is that of the last GP only (or the sum of all GP covariances if
            :attr:`propagate_source_uncertainty` is set).
        """
        from botorch.posteriors import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        reduced_candidates = self._strip_task(candidates_comp_scaled)
        posteriors = [
            cast("GPyTorchPosterior", gp._posterior(reduced_candidates))
            for gp in self._gp_chain
        ]

        import torch

        mean = torch.stack([p.mean for p in posteriors]).sum(dim=0)
        covariance = (
            reduce(operator.add, (p.distribution.lazy_covariance_matrix for p in posteriors))
            if self.propagate_source_uncertainty
            else posteriors[-1].distribution.lazy_covariance_matrix
        )
        return GPyTorchPosterior(MultivariateNormal(mean.squeeze(-1), covariance))


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
