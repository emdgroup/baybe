"""Base protocol for all optimizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable, TypeAlias, ClassVar
from collections.abc import Callable

from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType

Optimand: TypeAlias = Callable[[Tensor], Tensor]
"Type alias for the callable to be optimized."

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
    from torch import Tensor


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
        fixed_parameters: dict[int, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Recommend a batch of points from the given search space.

        Args:
            batch_size: The size of the recommendation batch.
            acquisition_function: The acquisition function to be optimized.
            searchspace: The search space from which to generate recommendations.
            fixed_parameters: A dictionary mapping parameter indices to fixed values.

        Returns:
            The recommendations and corresponding acquisition values.
        """
        ...
