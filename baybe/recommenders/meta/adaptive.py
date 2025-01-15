"""Meta recommenders that adaptively select recommenders based on the context."""

import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, instance_of
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.meta.base import MetaRecommender
from baybe.searchspace.core import SearchSpace
from baybe.utils.interval import Partition


@define
class BatchSizeAdaptiveMetaRecommender(MetaRecommender):
    """A meta recommender that selects recommenders according to the batch size."""

    recommenders: list[RecommenderProtocol] = field(
        converter=list, validator=deep_iterable(instance_of(RecommenderProtocol))
    )
    """The recommenders for the individual batch size intervals."""

    partition: Partition = field(
        converter=lambda x: Partition(x) if not isinstance(x, Partition) else x
    )
    """The partition mapping batch size intervals to recommenders. """

    @partition.validator
    def _validate_partitioning(self, _, value):
        if (lr := len(self.recommenders)) != (lp := len(value)):
            raise ValueError(
                f"The number of recommenders (given: {lr}) must be equal to the number "
                f"of intervals defined by the partition (given: {lp})."
            )

    @override
    def select_recommender(
        self,
        batch_size: int | None = None,
        searchspace: SearchSpace | None = None,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> RecommenderProtocol:
        if batch_size is None:
            raise ValueError("A batch size is required.")
        return self.recommenders[self.partition.get_interval_index(batch_size)]
