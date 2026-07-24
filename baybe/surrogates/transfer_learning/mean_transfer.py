"""Mean-transfer Gaussian process surrogate for transfer learning."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar, cast

from attrs import define, evolve, field
from typing_extensions import override

from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.transfer_learning.base import _SourceTargetTransferSurrogate

if TYPE_CHECKING:
    import pandas as pd
    from botorch.posteriors import GPyTorchPosterior, Posterior
    from gpytorch.means import Mean as GPyTorchMean
    from torch import Tensor

    from baybe.objectives.base import Objective
    from baybe.searchspace.core import SearchSpace


def _zero_mean_factory(
    searchspace: SearchSpace,
    objective: Objective,
    measurements: pd.DataFrame,
) -> GPyTorchMean:
    """Return a zero mean function, independent of the given search space and data."""
    from gpytorch.means import ZeroMean

    return ZeroMean()


@define
class MeanTransferSurrogate(_SourceTargetTransferSurrogate):
    """A transfer learning surrogate that transfers a source model's posterior mean.

    Fits a single-task source Gaussian process on the source subset and a single-task
    target Gaussian process on the residuals of the target data w.r.t. the source GP's
    posterior mean (in original target units). Predictions are the sum of the source
    and target posterior means.

    This is structurally equivalent to :class:`ResidualTransferSurrogate` but uses a
    zero prior mean for the target GP rather than a constant mean, reflecting the
    assumption that the source GP fully accounts for the mean of the target function
    and the target GP corrects only for remaining residual variation.

    The key difference from :class:`ResidualTransferSurrogate` is the output
    standardization: the target GP here standardizes the original target values
    ``y_target`` (not the residuals), so the kernel hyperparameters are learned on the
    original output scale rather than the residual scale. See section 9 of
    ``TL_PROTOTYPES.md`` for the mathematical relationship between the two approaches.

    Cold start: if the target task has no measurements yet, the surrogate falls back to
    the source GP's posterior (the best available estimate of the target without data).

    Note:
        Only a single source and a single target task are currently supported.
    """

    _max_sources: ClassVar[int] = 1
    # See base class.

    _target_gp: GaussianProcessSurrogate | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The single-task GP trained on the target residuals. ``None`` before fitting or
    when the target task has no measurements yet (cold start)."""

    @override
    def _fit_target(
        self,
        reduced_searchspace: SearchSpace,
        objective: Objective,
        target_measurements: pd.DataFrame,
    ) -> None:
        """Fit the target GP on the residuals w.r.t. the source GP's posterior mean.

        The target GP uses a zero prior mean and is trained on
        ``y_target - μ_source(X_target)``. The source GP mean is added back explicitly
        at prediction time, keeping the two GP evaluations independent and avoiding a
        nested GP call inside the target GP's posterior machinery.

        If the target task has no measurements yet, no target GP is built and the
        surrogate falls back to the source GP at prediction time.

        Args:
            reduced_searchspace: The task-free search space for the target GP.
            objective: The objective (a single modeled quantity after replication).
            target_measurements: The measurements belonging to the target task (may be
                empty).
        """
        if target_measurements.empty:
            self._target_gp = None
            return

        # Source posterior mean at the target inputs, in original target units.
        source_posterior = cast(
            "GPyTorchPosterior", self._source_gp.posterior(target_measurements)
        )
        source_mean = source_posterior.mean.detach().cpu().numpy().reshape(-1)

        # Fit the target GP on residuals with a zero prior mean. The source mean is
        # added back in _posterior, so the two GP evaluations remain independent.
        residual_measurements = target_measurements.copy()
        for target in objective.targets:
            residual_measurements[target.name] = (
                target_measurements[target.name].to_numpy() - source_mean
            )

        self._target_gp = evolve(
            self.base_surrogate, mean_or_factory=_zero_mean_factory
        )
        self._target_gp.fit(reduced_searchspace, objective, residual_measurements)

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Return the combined source-plus-target posterior on task-stripped candidates.

        The source GP and target GP are evaluated independently — the source GP is
        never called from within the target GP's posterior machinery. This avoids
        a nested GP evaluation that would allocate extra memory proportional to the
        number of candidates × the source training set size.

        Falls back to the source GP's posterior during cold start (no target data).

        Args:
            candidates_comp_scaled: Candidate points in the computational representation
                of the full search space (including the task column).

        Returns:
            A posterior whose mean is the sum of the source and target posterior means
            and whose covariance is that of the target GP alone.
        """
        from botorch.posteriors import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        if self._target_gp is None:
            return self._source_only_posterior(candidates_comp_scaled)

        reduced_candidates = self._strip_task(candidates_comp_scaled)

        source_posterior = cast(
            "GPyTorchPosterior", self._source_gp._posterior(reduced_candidates)
        )
        target_posterior = cast(
            "GPyTorchPosterior", self._target_gp._posterior(reduced_candidates)
        )

        mean = source_posterior.mean + target_posterior.mean
        covariance = target_posterior.distribution.lazy_covariance_matrix

        return GPyTorchPosterior(MultivariateNormal(mean.squeeze(-1), covariance))


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
