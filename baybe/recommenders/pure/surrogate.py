"""Base class for surrogate-based recommenders."""

import gc
from abc import ABC, abstractmethod

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace import SearchSpace
from baybe.settings import Settings
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.base import Surrogate, SurrogateProtocol
from baybe.utils.validation import preprocess_dataframe, validate_object_names


def _autoreplicate(surrogate: SurrogateProtocol, /) -> SurrogateProtocol:
    """Replicate single-output surrogate models and pass through everything else."""
    if isinstance(surrogate, Surrogate) and not surrogate.supports_multi_output:
        return surrogate.replicate()
    return surrogate


@define
class SurrogateRecommender(PureRecommender, ABC):
    """Abstract base class for recommenders that use a surrogate model."""

    _surrogate_model: SurrogateProtocol = field(
        alias="surrogate_model",
        factory=GaussianProcessSurrogate,
        converter=_autoreplicate,
    )
    """The surrogate model."""

    @abstractmethod
    def _prepare_recommendation(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None,
    ) -> None:
        """Prepare the surrogate-dependent recommendation state."""

    def get_surrogate(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> SurrogateProtocol:
        """Get the trained surrogate model."""
        # This fit applies internal caching and does not necessarily involve computation
        self._surrogate_model.fit(searchspace, objective, measurements)
        return self._surrogate_model

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if objective is None:
            raise NotImplementedError(
                f"Recommenders of type '{self.__class__.__name__}' require "
                "that an objective is specified."
            )

        validate_object_names(searchspace.parameters + objective.targets)

        if (measurements is None) or measurements.empty:
            raise NotImplementedError(
                f"Recommenders of type '{self.__class__.__name__}' do not support "
                "empty training data."
            )

        measurements = preprocess_dataframe(
            measurements,
            searchspace,
            objective,
            numerical_measurements_must_be_within_tolerance=False,
        )

        if pending_experiments is not None:
            pending_experiments = preprocess_dataframe(
                pending_experiments,
                searchspace,
                numerical_measurements_must_be_within_tolerance=False,
            )

        self._prepare_recommendation(
            searchspace=searchspace,
            objective=objective,
            measurements=measurements,
            pending_experiments=pending_experiments,
        )

        with Settings(preprocess_dataframes=False):
            return super().recommend(
                batch_size=batch_size,
                searchspace=searchspace,
                objective=objective,
                measurements=measurements,
                pending_experiments=pending_experiments,
            )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
