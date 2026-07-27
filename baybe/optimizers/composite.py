"""Composite optimizers."""

from __future__ import annotations

import gc
import warnings
from abc import ABC, abstractmethod
from collections.abc import Collection, Generator, Iterable
from typing import TYPE_CHECKING, TypeAlias

from attrs import define, field
from attrs.validators import gt, instance_of, min_len
from typing_extensions import override

from baybe.optimizers.base import OptimizerProtocol
from baybe.parameters.base import Parameter
from baybe.parameters.selectors import (
    ParameterSelectorProtocol,
    to_parameter_selector,
)
from baybe.searchspace import SearchSpace
from baybe.serialization.mixin import SerialMixin
from baybe.utils.validation import validate_optimizer_input

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.optimizers.base import ScoreFunction

    ParameterIndicator: TypeAlias = (
        str
        | Collection[str]
        | type[Parameter]
        | Collection[type[Parameter]]
        | ParameterSelectorProtocol
    )
    """The type of input that indicates which parameters to optimize."""

    SearchSpacePart: TypeAlias = tuple[frozenset[str], OptimizerProtocol]
    """Parameter names paired with the optimizer responsible for them."""


def _convert_components_to_selectors(
    raw: Iterable[tuple[ParameterIndicator, OptimizerProtocol]],
) -> tuple[tuple[ParameterSelectorProtocol, OptimizerProtocol], ...]:
    """Convert raw component specifications to normalized selector-optimizer pairs."""
    return tuple(
        (to_parameter_selector(selector), optimizer) for selector, optimizer in raw
    )


@define(frozen=True, slots=False)
class CompositionStrategy(ABC, SerialMixin):
    """Base class for composite optimizer strategies."""

    @abstractmethod
    def __call__(self, n_parts: int) -> Generator[int, tuple[Tensor, Tensor], None]:
        """Yield part indices to optimize in sequence.

        A composition strategy controls the order in which parts of the search
        space are optimized. Each time it yields an index ``i``, the caller
        optimizes part ``i`` while holding all other parts fixed. After each
        optimization step, the caller sends back the resulting ``(point, score)``
        pair via :meth:`~Generator.send`. The strategy may use this feedback to
        decide which part to optimize next (e.g., adaptive strategies) or ignore
        it (e.g., fixed schedules). The generator terminates by returning when
        the optimization sequence is complete.

        Args:
            n_parts: The number of parts in the partition.

        Yields:
            The index of the part to optimize next.
        """


@define(frozen=True, slots=False)
class Alternating(CompositionStrategy):
    """Cycle through parts for a fixed number of iterations."""

    n_iterations: int = field(default=3, validator=[instance_of(int), gt(0)])
    """Number of full alternating cycles."""

    @override
    def __call__(self, n_parts: int) -> Generator[int, tuple[Tensor, Tensor], None]:
        """Yield part indices in round-robin for ``n_iterations`` cycles."""
        for _ in range(self.n_iterations):
            yield from range(n_parts)


@define(frozen=True, slots=False, kw_only=True)
class SequentialOptimizer(OptimizerProtocol[SearchSpace]):
    """Optimizer that combines multiple optimizers over different search space parts.

    Each part of the search space is assigned to a dedicated optimizer. Points are
    optimized one at a time: for each point, the strategy cycles through the parts,
    optimizing one part while holding the others fixed. This means batch points are
    produced sequentially, not jointly.
    """

    components: tuple[tuple[ParameterSelectorProtocol, OptimizerProtocol], ...] = field(
        converter=_convert_components_to_selectors,
        validator=[min_len(1), validate_optimizer_input],
    )
    """Parameter selectors and their respective optimizers to be combined."""

    strategy: CompositionStrategy = field(
        factory=Alternating,
        validator=instance_of(CompositionStrategy),
    )
    """The strategy to use for combining the optimizers."""

    def _split_parameters(
        self,
        space: SearchSpace,
    ) -> list[SearchSpacePart]:
        """Resolve selectors to parameter name sets.

        Args:
            space: The full search space to split.

        Raises:
            ValueError: If a parameter is matched by multiple components.
            ValueError: If a constraint spans multiple parts.

        Returns:
            A list of (parameter names, optimizer) pairs.
        """
        assigned: set[str] = set()
        parts: list[SearchSpacePart] = []

        for selector, optimizer in self.components:
            selected_names = {p.name for p in space.parameters if selector(p)}

            if not selected_names:
                continue

            overlap = assigned & selected_names
            if overlap:
                raise ValueError(
                    f"Parameters {overlap} are matched by multiple components."
                )
            assigned.update(selected_names)

            for constraint in space.constraints:
                required = constraint._required_parameters
                if required & selected_names and not required <= selected_names:
                    raise ValueError(f"Constraint '{constraint}' spans multiple parts.")

            parts.append((frozenset(selected_names), optimizer))

        all_param_names = {p.name for p in space.parameters}
        unassigned = all_param_names - assigned
        if unassigned:
            warnings.warn(
                f"Parameters {sorted(unassigned)} are not assigned to any "
                f"optimizer component and will remain at their initial values.",
                UserWarning,
                stacklevel=2,
            )

        return parts

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
        parts: list[SearchSpacePart],
        score_function: ScoreFunction,
    ) -> tuple[Tensor, Tensor]:
        """Optimize a single point.

        Args:
            space: The full search space.
            parts: Resolved parts (parameter names + optimizer).
            score_function: The callable to optimize.

        Returns:
            The optimized point ``(1, n_cols)`` and its score ``(1,)``.
        """
        comp_rep_columns = space.comp_rep_columns

        optimizable_columns: list[frozenset[str]] = [
            frozenset(
                col
                for p in space.parameters
                if p.name in param_names
                for col in p.comp_rep_columns
            )
            for param_names, _ in parts
        ]

        current_point = self._sample_initial_point(space)

        strategy = self.strategy(len(parts))
        part_idx = next(strategy)

        while True:
            _, optimizer = parts[part_idx]
            free_columns = optimizable_columns[part_idx]

            fixed_values = {
                col: current_point[i].item()
                for i, col in enumerate(comp_rep_columns)
                if col not in free_columns
            }
            constrained_space = space.fix_parameters(fixed_values)

            result_point, result_score = optimizer(1, score_function, constrained_space)

            current_point = result_point.squeeze(0)

            try:
                part_idx = strategy.send((result_point, result_score))
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

        parts = self._split_parameters(space)
        n_cols = len(space.comp_rep_columns)

        base_X_pending = getattr(score_function, "X_pending", None)

        points = torch.empty(batch_size, n_cols)
        scores = torch.empty(batch_size)

        for b in range(batch_size):
            point, score = self._optimize_single_point(space, parts, score_function)
            points[b] = point.squeeze(0)
            scores[b] = score.squeeze(0)

            if b < batch_size - 1:
                new_pending = points[: b + 1]
                if base_X_pending is not None:
                    new_pending = torch.cat([base_X_pending, new_pending], dim=0)
                score_function.set_X_pending(new_pending)  # type: ignore[attr-defined]

        if batch_size > 1:
            score_function.set_X_pending(base_X_pending)  # type: ignore[attr-defined]

        return points, scores


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
