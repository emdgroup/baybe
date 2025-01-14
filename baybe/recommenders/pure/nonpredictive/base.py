"""Base class for all nonpredictive recommenders."""

import gc
import warnings
from abc import ABC

import pandas as pd
from attrs import define
from typing_extensions import override

from baybe.exceptions import IncompatibleArgumentError, UnusedObjectWarning
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace.core import SearchSpace


@define
class NonPredictiveRecommender(PureRecommender, ABC):
    """Abstract base class for all nonpredictive recommenders."""

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if pending_experiments is not None:
            raise IncompatibleArgumentError(
                f"Pending experiments were passed to '{self.__class__.__name__}"
                f".{self.recommend.__name__}' but non-predictive recommenders "
                f"cannot use this information. If you want to exclude the pending "
                f"experiments from the candidate set, adjust the search space "
                f"accordingly."
            )
        if (measurements is not None) and (len(measurements) != 0):
            warnings.warn(
                f"'{self.recommend.__name__}' was called with a non-empty "
                f"set of measurements but '{self.__class__.__name__}' does not "
                f"utilize any training data, meaning that the argument is ignored.",
                UnusedObjectWarning,
            )
        if objective is not None:
            warnings.warn(
                f"'{self.recommend.__name__}' was called with a an explicit objective "
                f"but '{self.__class__.__name__}' does not "
                f"consider any objectives, meaning that the argument is ignored.",
                UnusedObjectWarning,
            )
        return super().recommend(
            batch_size=batch_size,
            searchspace=searchspace,
            objective=objective,
            measurements=measurements,
            pending_experiments=None,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
