"""Composite optimizers."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
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
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
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

    Partition: TypeAlias = tuple[
        SearchSpace | SubspaceContinuous | SubspaceDiscrete,
        OptimizerProtocol,
    ]
    """A sub-space paired with the optimizer responsible for it."""


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
    def execute(
        self,
        partitions: list[Partition],
        score_function: ScoreFunction,
        space: SearchSpace,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Execute the composition strategy.

        Args:
            partitions: The already-partitioned sub-spaces and their optimizers.
            score_function: The function to be optimized.
            space: The full search space.
            batch_size: The number of points to find.

        Returns:
            The optimal parameter configurations and their corresponding scores.
        """


@define(frozen=True, slots=False)
class Alternating(CompositionStrategy):
    """Alternately optimize partitions, fixing others to current best.

    Uses multiple random starts with convergence tracking, inspired by
    BoTorch's ``optimize_acqf_mixed_alternating``. For batch sizes > 1,
    candidates are generated sequentially with ``X_pending`` to ensure
    diversity.
    """

    n_iterations: int = field(default=3, validator=[instance_of(int), gt(0)])
    """Maximum number of alternating rounds."""

    n_starts: int = field(default=10, validator=[instance_of(int), gt(0)])
    """Number of random starting points."""

    convergence_tol: float = field(default=1e-8, validator=instance_of(float))
    """Stop a start when improvement falls below this threshold."""

    @override
    def execute(
        self,
        partitions: list[Partition],
        score_function: ScoreFunction,
        space: SearchSpace,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        if len(partitions) == 1:
            sub_space, optimizer = partitions[0]
            return optimizer(batch_size, score_function, sub_space)

        partition_cols = self._compute_partition_columns(partitions, space)

        return self._optimize_batch(
            partitions, score_function, space, partition_cols, batch_size
        )

    def _compute_partition_columns(
        self,
        partitions: list[Partition],
        space: SearchSpace,
    ) -> tuple[tuple[int, ...], ...]:
        """Compute comp-rep column indices for each partition."""
        return tuple(
            space.get_comp_rep_parameter_indices(
                lambda p, _ss=sub_space: p.name in _ss.parameter_names
            )
            for sub_space, _ in partitions
        )

    def _optimize_batch(
        self,
        partitions: list[Partition],
        score_function: ScoreFunction,
        space: SearchSpace,
        partition_cols: tuple[tuple[int, ...], ...],
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Generate a batch of candidates using sequential greedy with X_pending.

        Args:
            partitions: The partitioned sub-spaces and their optimizers.
            score_function: The function to be optimized.
            space: The full search space.
            partition_cols: Comp-rep column indices per partition.
            batch_size: The number of points to find.

        Returns:
            The optimal parameter configurations and their corresponding scores.

        Raises:
            NotImplementedError: If the score function does not support X_pending
                and the batch size is greater than 1.
        """
        import torch

        if batch_size > 1 and not hasattr(score_function, "set_X_pending"):
            raise NotImplementedError(
                f"'{self.__class__.__name__}' requires a `score_function` that "
                f"supports `set_X_pending` for batch sizes > 1."
            )

        bounds = torch.from_numpy(space.comp_rep_bounds.to_numpy(copy=True))
        d_full = bounds.shape[1]
        base_X_pending = getattr(score_function, "X_pending", None)
        candidates = torch.empty(0, d_full, dtype=bounds.dtype)

        for _q in range(batch_size):
            new_candidate, _ = self._optimize_single_candidate(
                partitions, score_function, space, bounds, partition_cols
            )
            candidates = torch.cat([candidates, new_candidate.unsqueeze(0)], dim=0)

            if batch_size > 1:
                pending = (
                    torch.cat([base_X_pending, candidates], dim=0)
                    if base_X_pending is not None
                    else candidates
                )
                score_function.set_X_pending(pending)  # type: ignore[attr-defined]

        if batch_size > 1:
            score_function.set_X_pending(base_X_pending)  # type: ignore[attr-defined]

        with torch.no_grad():
            joint_scores = score_function(candidates.unsqueeze(0))

        return candidates, joint_scores

    def _optimize_single_candidate(
        self,
        partitions: list[Partition],
        score_function: ScoreFunction,
        space: SearchSpace,
        bounds: Tensor,
        partition_cols: tuple[tuple[int, ...], ...],
    ) -> tuple[Tensor, Tensor]:
        """Run multi-restart alternating optimization for a single candidate.

        Args:
            partitions: The partitioned sub-spaces and their optimizers.
            score_function: The function to be optimized.
            space: The full search space.
            bounds: The comp-rep bounds tensor of shape ``(2, d)``.
            partition_cols: Comp-rep column indices per partition.

        Returns:
            The best candidate and its score.
        """
        import torch

        global_vectors = self._initialize_global_vectors(
            partitions, partition_cols, bounds
        )
        current_scores = torch.full((self.n_starts,), float("-inf"), dtype=bounds.dtype)
        done = torch.zeros(self.n_starts, dtype=torch.bool)

        for _ in range(self.n_iterations):
            prev_scores = current_scores.clone()

            self._alternating_step(
                partitions,
                score_function,
                space,
                partition_cols,
                global_vectors,
                current_scores,
                done,
            )

            improvement = current_scores[~done] - prev_scores[~done]
            newly_done = improvement < self.convergence_tol
            active_indices = (~done).nonzero(as_tuple=True)[0]
            done[active_indices[newly_done]] = True

            if done.all():
                break

        best_idx = torch.argmax(current_scores)
        return global_vectors[best_idx], current_scores[best_idx]

    def _alternating_step(
        self,
        partitions: list[Partition],
        score_function: ScoreFunction,
        space: SearchSpace,
        partition_cols: tuple[tuple[int, ...], ...],
        global_vectors: Tensor,
        current_scores: Tensor,
        done: Tensor,
    ) -> None:
        """Run one round of alternating optimization across all partitions.

        Updates ``global_vectors`` and ``current_scores`` in place.

        Args:
            partitions: The partitioned sub-spaces and their optimizers.
            score_function: The function to be optimized.
            space: The full search space.
            partition_cols: Comp-rep column indices per partition.
            global_vectors: The current best vectors for all restarts.
            current_scores: The current best scores for all restarts.
            done: Convergence flags for each restart.
        """
        for i, (_sub_space, optimizer) in enumerate(partitions):
            cols_i = partition_cols[i]
            all_other_cols = [
                col for j, cols in enumerate(partition_cols) if j != i for col in cols
            ]

            for r in (~done).nonzero(as_tuple=True)[0]:
                fixed_features = {
                    int(col): global_vectors[r, col].item() for col in all_other_cols
                }
                points, scores = optimizer(
                    1,
                    score_function,
                    space,
                    fixed_features=fixed_features,
                )
                for col in cols_i:
                    global_vectors[r, col] = points[0, col]
                current_scores[r] = scores.item()

    def _initialize_global_vectors(
        self,
        partitions: list[Partition],
        partition_cols: tuple[tuple[int, ...], ...],
        bounds: Tensor,
    ) -> Tensor:
        """Initialize starting points, sampling appropriately per partition type.

        Args:
            partitions: The partitioned sub-spaces and their optimizers.
            partition_cols: Comp-rep column indices per partition.
            bounds: The comp-rep bounds tensor of shape ``(2, d)``.

        Returns:
            A tensor of shape ``(n_starts, d_full)`` with valid initial points.
        """
        import torch

        from baybe.utils.dataframe import to_tensor

        d_full = bounds.shape[1]
        global_vectors = torch.empty(self.n_starts, d_full, dtype=bounds.dtype)

        for (sub_space, _), cols in zip(partitions, partition_cols):
            if isinstance(sub_space, SubspaceDiscrete):
                samples = to_tensor(
                    sub_space.comp_rep.sample(self.n_starts, replace=True)
                )
            elif isinstance(sub_space, SubspaceContinuous):
                samples = to_tensor(sub_space.sample_uniform(self.n_starts))
            else:
                disc_samples = to_tensor(
                    sub_space.discrete.comp_rep.sample(self.n_starts, replace=True)
                )
                cont_samples = to_tensor(
                    sub_space.continuous.sample_uniform(self.n_starts)
                )
                samples = torch.cat([disc_samples, cont_samples], dim=1)

            for j, col in enumerate(cols):
                global_vectors[:, col] = samples[:, j]

        return global_vectors


@define(kw_only=True)
class CompositeOptimizer(
    OptimizerProtocol[SearchSpace | SubspaceContinuous | SubspaceDiscrete],
):
    """Optimizer that allows stacking multiple optimizers."""

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

    def _partition_parameters(
        self,
        space: SearchSpace,
    ) -> list[Partition]:
        """Partition the search space and build sub-spaces per component.

        Args:
            space: The full search space to partition.

        Raises:
            ValueError: If a parameter is matched by multiple components.
            ValueError: If a constraint spans multiple partitions.

        Returns:
            A list of (sub-space, optimizer) pairs.
        """
        assigned: set[str] = set()
        partitions: list[Partition] = []

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
                    raise ValueError(
                        f"Constraint '{constraint}' spans multiple partitions."
                    )

            sub_space = SearchSpace.from_product(
                parameters=[p for p in space.parameters if p.name in selected_names],
                constraints=[
                    c
                    for c in space.constraints
                    if c._required_parameters <= selected_names
                ],
            )

            if sub_space.type is SearchSpaceType.CONTINUOUS:
                partitions.append((sub_space.continuous, optimizer))
            elif sub_space.type is SearchSpaceType.DISCRETE:
                partitions.append((sub_space.discrete, optimizer))
            else:
                partitions.append((sub_space, optimizer))

        return partitions

    @override
    def __call__(
        self,
        batch_size: int,
        score_function: ScoreFunction,
        space: SearchSpace | SubspaceContinuous | SubspaceDiscrete,
        fixed_features: dict[int, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(space, SubspaceContinuous):
            space = SearchSpace(continuous=space)
        elif isinstance(space, SubspaceDiscrete):
            space = SearchSpace(discrete=space)

        partitions = self._partition_parameters(space)
        return self.strategy.execute(partitions, score_function, space, batch_size)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
