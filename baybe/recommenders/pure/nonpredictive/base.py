"""Base class for all nonpredictive recommenders."""

import gc
import warnings
from abc import ABC

import pandas as pd
from attr import fields
from attrs import define

from baybe.exceptions import UnusedObjectWarning
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace.core import SearchSpace, SearchSpaceType


@define
class NonPredictiveRecommender(PureRecommender, ABC):
    """Abstract base class for all nonpredictive recommenders."""

    def recommend(  # noqa: D102
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # See base class.

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
        if (pending_experiments is not None) and (
            self.allow_recommending_pending_experiments
            or searchspace.type is not SearchSpaceType.DISCRETE
        ):
            warnings.warn(
                f"Pending experiments were provided but the selected recommender "
                f"'{self.__class__.__name__}' only utilizes this information for "
                f"purely discrete spaces and "
                f"{fields(self.__class__).allow_recommending_pending_experiments.name}"
                f"=False.",
                UnusedObjectWarning,
            )
        return super().recommend(
            batch_size=batch_size,
            searchspace=searchspace,
            objective=objective,
            measurements=measurements,
            pending_experiments=pending_experiments,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
