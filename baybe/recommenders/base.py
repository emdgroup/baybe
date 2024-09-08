"""Base protocol for all recommenders."""

from typing import Protocol, runtime_checkable

import cattrs
import pandas as pd
from cattrs import override

from baybe.objectives.base import Objective
from baybe.searchspace import SearchSpace
from baybe.serialization import converter, unstructure_base
from baybe.serialization.core import get_base_structure_hook


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
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
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


# TODO: The workarounds below are currently required since the hooks created through
#   `unstructure_base` and `get_base_structure_hook` do not reuse the hooks of the
#   actual class, hence we cannot control things there. Fix is already planned and also
#   needed for other reasons.

# Register (un-)structure hooks
converter.register_unstructure_hook(
    RecommenderProtocol,
    lambda x: unstructure_base(
        x,
        # TODO: Remove once deprecation got expired:
        overrides=dict(
            acquisition_function_cls=cattrs.override(omit=True),
            # Temporary workaround (see TODO note above)
            _surrogate_model=override(rename="surrogate_model"),
            _current_recommender=override(omit=False),
            _used_recommender_ids=override(omit=False),
        ),
    ),
)
converter.register_structure_hook(
    RecommenderProtocol,
    get_base_structure_hook(
        RecommenderProtocol,
        # Temporary workaround (see TODO note above)
        overrides=dict(
            _surrogate_model=override(rename="surrogate_model"),
            _current_recommender=override(omit=False),
            _used_recommender_ids=override(omit=False),
        ),
    ),
)
