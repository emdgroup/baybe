"""Composite optimizers."""

from __future__ import annotations

import gc
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pandas as pd
from attrs import define, field
from attrs.validators import gt, instance_of, min_len
from typing_extensions import override

from baybe.exceptions import (
    IncompatibleAcquisitionFunctionError,
    IncompatibleSearchSpaceError,
)
from baybe.optimizers.base import OptimizerProtocol
from baybe.parameters.selectors import ParameterSelectorProtocol, to_parameter_selector
from baybe.searchspace import SearchSpace
from baybe.serialization.mixin import SerialMixin
from baybe.settings import active_settings

if TYPE_CHECKING:
    from baybe.optimizers.base import OptimizationResult, ScoreFunction


@define(frozen=True)
class OptimizationStep(SerialMixin):
    """A parameter selector paired with the optimizer responsible for it."""

    selector: ParameterSelectorProtocol = field(converter=to_parameter_selector)
    """The selector identifying which parameters this step optimizes."""

    optimizer: OptimizerProtocol = field(validator=instance_of(OptimizerProtocol))
    """The optimizer to apply to the selected parameters."""


@define(frozen=True, slots=False)
class OptimizationSchedule(ABC, SerialMixin):
    """Base class for optimization schedules.

    An optimization schedule controls which parts of the search space are optimized
    and in what order. It yields :class:`OptimizationStep` instances on each
    iteration, one step at a time.
    """

    @abstractmethod
    def __call__(
        self, searchspace: SearchSpace
    ) -> Generator[OptimizationStep, OptimizationResult, None]:
        """Yield optimization steps to apply in sequence.

        Each time the generator yields an :class:`OptimizationStep`, the caller
        resolves the selector against the space, optimizes the free parameters while
        holding all others fixed, and sends back the resulting ``(point, score)`` pair
        via :meth:`~Generator.send`. The schedule may use this feedback to decide which
        step to yield next (e.g., adaptive schedules) or ignore it (e.g., fixed
        schedules). The generator terminates by returning when the sequence is complete.

        Args:
            searchspace: The full search space to optimize over.

        Yields:
            The next optimization step to apply.
        """


@define(frozen=True)
class CyclicOptimizationSchedule(OptimizationSchedule):
    """Cycle through steps in round-robin for a fixed number of cycles."""

    steps: tuple[OptimizationStep, ...] = field(validator=min_len(1))
    """The optimization steps to be cycled through."""

    n_cycles: int = field(default=1, validator=[instance_of(int), gt(0)])
    """The number of full cycles to perform."""

    @override
    def __call__(
        self, searchspace: SearchSpace
    ) -> Generator[OptimizationStep, OptimizationResult, None]:
        """Yield steps in round-robin for the specified number of cycles."""
        active_steps = [
            step
            for step in self.steps
            if any(step.selector(p) for p in searchspace.parameters)
        ]
        if not active_steps:
            raise IncompatibleSearchSpaceError(
                "No optimization can be performed because none of the specified steps "
                "matched any parameter in the given search space."
            )
        if n_skipped := len(self.steps) - len(active_steps):
            warnings.warn(
                f"{n_skipped} of {len(self.steps)} optimization step(s) matched no "
                "parameters in the given search space and will be skipped.",
                UserWarning,
                stacklevel=2,
            )
        for _ in range(self.n_cycles):
            for step in active_steps:
                _ = yield step


@define(frozen=True)
class BlockCoordinateOptimizer(OptimizerProtocol):
    """An optimizer that performs block optimization over specified subspaces.

    Each subspace is assigned to a dedicated optimizer in the form of an
    :class:`~OptimizationSchedule`. The subspaces are then optimized block-wise
    according to the schedule, holding the respective other part of the search space
    fixed. See also: https://en.wikipedia.org/wiki/Coordinate_descent

    For batch optimization, the same schedule is applied repeatedly per point,
    resulting in a greedy optimization strategy.
    """

    schedule: OptimizationSchedule = field(
        validator=instance_of(OptimizationSchedule),
    )
    """The schedule controlling which subspaces are optimized and in what order."""

    def _optimize_single_point(
        self,
        score_function: ScoreFunction,
        searchspace: SearchSpace,
        steps: Generator[OptimizationStep, OptimizationResult, None],
    ) -> OptimizationResult:
        """Optimize a single point.

        Args:
            score_function: The callable to optimize.
            searchspace: The full search space.
            steps: A generator yielding :class:`~OptimizationStep`s.

        Returns:
            The optimization result for a single point of the batch.
        """
        dfs: list[pd.DataFrame] = []
        if not searchspace.discrete.is_empty:
            dfs.append(
                searchspace.discrete.exp_rep.sample(1, replace=True).reset_index(
                    drop=True
                )
            )
        if not searchspace.continuous.is_empty:
            dfs.append(searchspace.continuous.sample_uniform(1))

        current_point: dict[str, Any] = {
            str(k): v for k, v in pd.concat(dfs, axis=1).iloc[0].items()
        }

        step = next(steps)

        while True:
            free_names = {p.name for p in searchspace.parameters if step.selector(p)}
            fixed_values = {
                k: v for k, v in current_point.items() if k not in free_names
            }
            constrained_space = searchspace._fix_parameters(**fixed_values)

            result_point, result_score = step.optimizer(
                1, score_function, constrained_space
            )
            result_point = result_point.squeeze(0)
            result_score = result_score.squeeze(0)

            # Merge optimized free-parameter values back into current exp-rep point.
            result_comp = dict(zip(searchspace.comp_rep_columns, result_point.tolist()))
            current_point.update(searchspace._comp_rep_to_exp_rep(result_comp))

            try:
                step = steps.send((result_point, result_score))
            except StopIteration:
                break

        return result_point, result_score

    @override
    def __call__(
        self,
        batch_size: int,
        score_function: ScoreFunction,
        searchspace: SearchSpace,
    ) -> OptimizationResult:
        import torch
        from botorch.acquisition import AnalyticAcquisitionFunction

        n_cols = len(searchspace.comp_rep_columns)
        points = torch.empty(batch_size, n_cols, dtype=active_settings.DTypeFloatTorch)
        scores = torch.empty(batch_size, dtype=active_settings.DTypeFloatTorch)

        if batch_size == 1:
            steps = self.schedule(searchspace)
            points[0], scores[0] = self._optimize_single_point(
                score_function, searchspace, steps
            )
            return points, scores

        if searchspace.continuous.has_interpoint_constraints:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' does not support batch recommendation "
                f"when interpoint constraints are present."
            )

        if isinstance(score_function, AnalyticAcquisitionFunction):
            raise IncompatibleAcquisitionFunctionError(
                f"'{type(self).__name__}' does not support analytic acquisition "
                f"functions for batch sizes greater than 1 but got an acquisition "
                f"function of type '{type(score_function).__name__}'."
            )

        base_X_pending = score_function.X_pending

        for b in range(batch_size):
            steps = self.schedule(searchspace)
            points[b], scores[b] = self._optimize_single_point(
                score_function, searchspace, steps
            )

            pending = points[: b + 1]
            if base_X_pending is not None:
                pending = torch.cat([base_X_pending, pending], dim=0)
            score_function.set_X_pending(pending)

        score_function.set_X_pending(base_X_pending)

        return points, scores


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
