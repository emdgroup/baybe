"""Mean-transfer Gaussian process surrogate for transfer learning."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar

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
    as the *prior mean* of the target GP. The target GP therefore models the target
    data directly (in original target units) while being anchored to the source
    predictions, and its posterior already incorporates the transferred source mean.

    This differs from :class:`ResidualTransferSurrogate`, which instead fits a
    zero-mean GP on the residuals ``y_target - μ_source(X_target)`` and adds the source
    mean back at prediction time. The two coincide in the idealized limit but differ in
    practice through their output standardization: the target GP here standardizes the
    original target values ``y_target``, whereas the residual GP standardizes the
    (typically smaller-spread) residuals, so the two learn kernel hyperparameters on
    different output scales. See section 9 of ``TL_PROTOTYPES.md`` for the mathematical
    relationship between the two approaches.

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
    """The single-task target GP anchored to the source posterior mean. ``None`` before
    fitting or when the target task has no measurements yet (cold start)."""

    @override
    def _fit_target(
        self,
        reduced_searchspace: SearchSpace,
        objective: Objective,
        target_measurements: pd.DataFrame,
    ) -> None:
        """Fit the target GP using the source GP's posterior mean as its prior mean.

        The target GP is trained on the original target measurements with the source
        posterior mean ``μ_source`` as its prior mean, so its posterior directly
        represents the transferred prediction. The source mean is supplied lazily via
        the source GP's ``posterior_mean_function`` method, which the target GP
        evaluates and caches during hyperparameter optimization.

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

        self._target_gp = evolve(
            self.base_surrogate,
            mean_or_factory=self._source_gp.posterior_mean_function,
        )
        self._target_gp.fit(reduced_searchspace, objective, target_measurements)

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Return the target GP posterior on task-stripped candidates.

        Because the source posterior mean is baked into the target GP as its prior
        mean, the target GP's own posterior already is the combined mean-transfer
        prediction; no explicit source-plus-target combination is needed.

        Falls back to the source GP's posterior during cold start (no target data).

        Args:
            candidates_comp_scaled: Candidate points in the computational representation
                of the full search space (including the task column).

        Returns:
            The target GP's posterior (or the source GP's posterior during cold start).
        """
        if self._target_gp is None:
            return self._source_only_posterior(candidates_comp_scaled)

        reduced_candidates = self._strip_task(candidates_comp_scaled)
        return self._target_gp._posterior(reduced_candidates)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
