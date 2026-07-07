"""Base protocol for all optimizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from baybe.searchspace import SearchSpace, SubspaceContinuous, SubspaceDiscrete

TSpace = TypeVar("TSpace", bound=SearchSpace | SubspaceDiscrete | SubspaceContinuous)
"The type of space to optimize over."

if TYPE_CHECKING:
    from torch import Tensor

    ScoreFunction: TypeAlias = Callable[[Tensor], Tensor]
    "Type alias for a callable to be optimized."


@runtime_checkable
class OptimizerProtocol(Protocol, Generic[TSpace]):  # type: ignore[misc]
    """Type protocol specifying the interface optimizers need to implement."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def __call__(
        self, batch_size: int, score_function: ScoreFunction, space: TSpace
    ) -> tuple[Tensor, Tensor]:
        """Optimize a given callable over the specified space.

        Args:
            batch_size: The number of points to find.
            score_function: The callable to be optimized.
            space: The space to optimize over.

        Returns:
            The optimal parameter configurations and their corresponding scores.
        """
