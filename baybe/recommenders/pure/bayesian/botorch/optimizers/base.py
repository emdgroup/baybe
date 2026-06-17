"""Base protocol for all optimizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from baybe.searchspace import SearchSpace

if TYPE_CHECKING:
    from torch import Tensor


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Type protocol specifying the interface optimizers need to implement."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def __call__(
        self,
        batch_size: int,
        acquisition_function: UtilityFunction,
        searchspace: SearchSpace,
    ) -> tuple[Tensor, Tensor]:
        """Recommend a batch of points from the given search space.

        Args:
            batch_size: The size of the recommendation batch.
            acquisition_function: The utility function to be optimized.
            searchspace: The search space from which to generate recommendations.

        Returns:
            The recommendations and corresponding acquisition values.
        """
        ...


@runtime_checkable
class UtilityFunction(Protocol):
    """Interface for callable utility functions used in acquisition optimization.

    Any object implementing this protocol can be used as an objective for
    gradient-based optimizers, decoupling the optimizer from BoTorch-specific
    acquisition function types.
    """

    __slots__ = ()

    def optimize(
        self,
        batch_size: int,
        searchspace: SearchSpace,
    ) -> tuple[Tensor, Tensor]:
        """Find the optimal points in the given search space.

        Args:
            batch_size: The number of points to recommend.
            searchspace: The search space to optimize over.

        Returns:
            The optimal points and their utility values.
        """
        ...
