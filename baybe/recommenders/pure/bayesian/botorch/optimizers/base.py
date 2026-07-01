"""Base protocol for all optimizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeAlias, runtime_checkable

from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType

if TYPE_CHECKING:
    from torch import Tensor

    Optimand: TypeAlias = Callable[[Tensor], Tensor]
    "Type alias for the callable to be optimized."


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Type protocol specifying the interface optimizers need to implement."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    compatibility: ClassVar[SearchSpaceType]
    """Class variable reflecting the search space compatibility."""

    def __call__(
        self,
        batch_size: int,
        acquisition_function: Optimand,
        searchspace: SearchSpace,
        fixed_parameters: dict[str, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Recommend a batch of points from the given search space.

        Args:
            batch_size: The size of the recommendation batch.
            acquisition_function: The acquisition function to be optimized.
            searchspace: The search space from which to generate recommendations.
            fixed_parameters: A dictionary mapping parameter names to fixed values.

        Returns:
            The recommendations and corresponding acquisition values.
        """
