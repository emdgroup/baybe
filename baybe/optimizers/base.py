"""Base protocol for all optimizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from baybe.searchspace import SearchSpace

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
    from torch import Tensor

    ScoreFunction: TypeAlias = BoAcquisitionFunction
    """Type alias for a callable to be optimized."""


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Type protocol specifying the interface optimizers need to implement."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def __call__(
        self, batch_size: int, score_function: ScoreFunction, searchspace: SearchSpace
    ) -> tuple[Tensor, Tensor]:
        """Optimize a given callable over the specified space.

        Args:
            batch_size: The number of points to find.
            score_function: The callable to be optimized.
            searchspace: The space to optimize over.

        Returns:
            The optimal parameter configurations and their corresponding scores.
        """
