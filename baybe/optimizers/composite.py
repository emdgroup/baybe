"""Composite optimizers."""

from __future__ import annotations

import gc
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from attrs import define, field
from attrs.validators import gt, instance_of, min_len
from typing_extensions import override

from baybe.optimizers.base import OptimizerProtocol
from baybe.parameters.selectors import (
    ParameterSelectorProtocol,
    to_parameter_selector,
)
from baybe.searchspace import SearchSpace
from baybe.serialization.mixin import SerialMixin

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.optimizers.base import ScoreFunction


@define(frozen=True)
class OptimizationStep(SerialMixin):
    """A parameter selector paired with the optimizer responsible for it."""

    selector: ParameterSelectorProtocol = field(converter=to_parameter_selector)
    """The selector identifying which parameters this component optimizes."""

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
    ) -> Generator[OptimizationStep, tuple[Tensor, Tensor], None]:
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
    """Number of full cycles."""

    @override
    def __call__(
        self, searchspace: SearchSpace
    ) -> Generator[OptimizationStep, tuple[Tensor, Tensor], None]:
        """Yield steps in round-robin for ``n_cycles`` cycles."""
        for _ in range(self.n_cycles):
            for step in self.steps:
                selected_names = {
                    p.name for p in searchspace.parameters if step.selector(p)
                }
                if not selected_names:
                    warnings.warn(
                        "A parameter selector matched no parameters in the "
                        "given search space and the corresponding optimizer "
                        "is skipped.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                yield step


@define(frozen=True)
class SequentialOptimizer(OptimizerProtocol):
    """Optimizer that combines multiple optimizers over different search space parts.

    Each part of the search space is assigned to a dedicated optimizer. Points are
    optimized one at a time: for each point, the strategy cycles through the parts,
    optimizing one part while holding the others fixed. This means batch points are
    produced sequentially, not jointly.
    """

    schedule: OptimizationSchedule = field(
        validator=instance_of(OptimizationSchedule),
    )
    """The schedule controlling which parts are optimized and in what order."""

    def _optimize_single_point(
        self,
        searchspace: SearchSpace,
        schedule_gen: Generator[OptimizationStep, tuple[Tensor, Tensor], None],
        score_function: ScoreFunction,
    ) -> tuple[Tensor, Tensor]:
        """Optimize a single point.

        Args:
            searchspace: The full search space.
            schedule_gen: A generator yielding optimization steps.
            score_function: The callable to optimize.

        Returns:
            The optimized point ``(n_cols,)`` and its score as a scalar tensor.
        """
        current_point: dict[str, Any] = {
            str(k): v for k, v in searchspace.sample_uniform(1).iloc[0].items()
        }
        step = next(schedule_gen)

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
                step = schedule_gen.send((result_point, result_score))
            except StopIteration:
                break

        return result_point, result_score

    @override
    def __call__(
        self,
        batch_size: int,
        score_function: ScoreFunction,
        searchspace: SearchSpace,
    ) -> tuple[Tensor, Tensor]:
        import torch

        n_cols = len(searchspace.comp_rep_columns)
        base_X_pending = score_function.X_pending

        points = torch.empty(batch_size, n_cols)
        scores = torch.empty(batch_size)

        for b in range(batch_size):
            step = self.schedule(searchspace)
            point, score = self._optimize_single_point(
                searchspace, step, score_function
            )
            points[b], scores[b] = point, score

            pending = points[: b + 1]
            if base_X_pending is not None:
                pending = torch.cat([base_X_pending, pending], dim=0)
            score_function.set_X_pending(pending)

        score_function.set_X_pending(base_X_pending)

        return points, scores


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
