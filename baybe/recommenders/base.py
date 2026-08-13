"""Base protocol for all recommenders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from baybe.objectives.base import Objective
from baybe.searchspace import SearchSpace

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrameT


@runtime_checkable
class RecommenderProtocol(Protocol):
    """Type protocol specifying the interface recommenders need to implement."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: IntoDataFrameT | None = None,
        pending_experiments: IntoDataFrameT | None = None,
    ) -> IntoDataFrameT:
        """Recommend a batch of points from the given search space.

        Args:
            batch_size: The number of points to be recommended.
            searchspace: The search space from which to recommend the points.
            objective: An optional objective to be optimized.
            measurements: Optional experimentation data that can be used for model
                training. The data is to be provided in "experimental representation":
                It needs to contain one column for each parameter spanning the search
                space (column name matching the parameter name) and one column for each
                target tracked by the objective (column name matching the target name).
                Each row corresponds to one conducted experiment, where the parameter
                columns define the experimental setting and the target columns report
                the measured outcomes.
            pending_experiments: Parameter configurations in "experimental
                representation" specifying experiments that are currently pending.

        Returns:
            A dataframe containing the recommendations in experimental representation
            as individual rows.
        """
        ...
