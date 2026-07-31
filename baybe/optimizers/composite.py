"""Composite optimizers."""

from __future__ import annotations

import gc
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

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
        self, space: SearchSpace
    ) -> Generator[OptimizationStep, tuple[Tensor, Tensor], None]:
        """Yield optimization steps to apply in sequence.

        Each time the generator yields an :class:`OptimizationStep`, the caller
        resolves the selector against the space, optimizes the free parameters while
        holding all others fixed, and sends back the resulting ``(point, score)`` pair
        via :meth:`~Generator.send`. The schedule may use this feedback to decide which
        step to yield next (e.g., adaptive schedules) or ignore it (e.g., fixed
        schedules). The generator terminates by returning when the sequence is complete.

        Args:
            space: The full search space to optimize over.

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
        self, space: SearchSpace
    ) -> Generator[OptimizationStep, tuple[Tensor, Tensor], None]:
        """Yield steps in round-robin for ``n_cycles`` cycles."""
        for _ in range(self.n_cycles):
            for step in self.steps:
                selected_names = {p.name for p in space.parameters if step.selector(p)}
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
class SequentialOptimizer(OptimizerProtocol[SearchSpace]):
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

    @staticmethod
    def _sample_initial_point(space: SearchSpace) -> Tensor:
        """Sample a random point from the space in comp-rep.

        Args:
            space: The search space to sample from.

        Returns:
            A 1-D tensor of length ``len(space.comp_rep_columns)``.
        """
        import torch

        init_exp = space.sample_uniform(1)
        init_comp = space.transform(init_exp)
        return torch.tensor(init_comp.values[0], dtype=torch.float64)

    def _optimize_single_point(
        self,
        space: SearchSpace,
        schedule_gen: Generator[OptimizationStep, tuple[Tensor, Tensor], None],
        score_function: ScoreFunction,
    ) -> tuple[Tensor, Tensor]:
        """Optimize a single point.

        Args:
            space: The full search space.
            schedule_gen: A generator yielding optimization steps.
            score_function: The callable to optimize.

        Returns:
            The optimized point ``(1, n_cols)`` and its score ``(1,)``.
        """
        comp_rep_columns = space.comp_rep_columns
        current_point = self._sample_initial_point(space)

        component = next(schedule_gen)

        while True:
            selected_names = {p.name for p in space.parameters if component.selector(p)}
            free_columns = frozenset(
                col
                for p in space.parameters
                if p.name in selected_names
                for col in p.comp_rep_columns
            )
            fixed_values = {
                col: current_point[i].item()
                for i, col in enumerate(comp_rep_columns)
                if col not in free_columns
            }
            constrained_space = space._fix_parameters(fixed_values)

            result_point, result_score = component.optimizer(
                1, score_function, constrained_space
            )
            current_point = result_point.squeeze(0)

            try:
                component = schedule_gen.send((result_point, result_score))
            except StopIteration:
                break

        return current_point.unsqueeze(0), result_score

    @override
    def __call__(
        self,
        batch_size: int,
        score_function: ScoreFunction,
        space: SearchSpace,
    ) -> tuple[Tensor, Tensor]:
        import torch

        n_cols = len(space.comp_rep_columns)
        base_X_pending = getattr(score_function, "X_pending", None)

        points = torch.empty(batch_size, n_cols)
        scores = torch.empty(batch_size)

        for b in range(batch_size):
            schedule_gen = self.schedule(space)
            point, score = self._optimize_single_point(
                space, schedule_gen, score_function
            )
            points[b] = point.squeeze(0)
            scores[b] = score.squeeze(0)

            if b < batch_size - 1:
                new_pending = points[: b + 1]
                if base_X_pending is not None:
                    new_pending = torch.cat([base_X_pending, new_pending], dim=0)
                score_function.set_X_pending(new_pending)

        if batch_size > 1:
            score_function.set_X_pending(base_X_pending)

        return points, scores


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
