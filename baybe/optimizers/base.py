"""Base protocol for all optimizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeAlias, runtime_checkable

from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType

if TYPE_CHECKING:
    from torch import Tensor

    ScoreFunction: TypeAlias = Callable[[Tensor], Tensor]
    "Type alias for a callable to be optimized."


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
        score_function: ScoreFunction,
        searchspace: SearchSpace,
        fixed_parameters: dict[str, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Optimize a given callable over the specified search space.

        Args:
            batch_size: The number of points to find.
            score_function: The callable to be optimized.
            searchspace: The search space to optimize over.
            fixed_parameters: A dictionary mapping parameter names to fixed values.

        Returns:
            The optimal parameter configurations and their corresponding scores.
        """
