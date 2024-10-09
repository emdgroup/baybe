"""Base class for all Bayesian recommenders."""

import gc
import warnings
from abc import ABC

import pandas as pd
from attrs import define, field, fields

from baybe.acquisition.acqfs import qLogExpectedImprovement
from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import DeprecationError, InvalidSurrogateModelError
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import CustomONNXSurrogate, GaussianProcessSurrogate
from baybe.surrogates.base import IndependentGaussianSurrogate, SurrogateProtocol


@define
class BayesianRecommender(PureRecommender, ABC):
    """An abstract class for Bayesian Recommenders."""

    _surrogate_model: SurrogateProtocol = field(
        alias="surrogate_model", factory=GaussianProcessSurrogate
    )
    """The used surrogate model."""

    acquisition_function: AcquisitionFunction = field(
        converter=convert_acqf, factory=qLogExpectedImprovement
    )
    """The used acquisition function class."""

    _botorch_acqf = field(default=None, init=False)
    """The current acquisition function."""

    acquisition_function_cls: str | None = field(default=None, kw_only=True)
    "Deprecated! Raises an error when used."

    @acquisition_function_cls.validator
    def _validate_deprecated_argument(self, _, value) -> None:
        """Raise DeprecationError if old acquisition_function_cls parameter is used."""
        if value is not None:
            raise DeprecationError(
                "Passing 'acquisition_function_cls' to the constructor is deprecated. "
                "The parameter has been renamed to 'acquisition_function'."
            )

    @property
    def surrogate_model(self) -> SurrogateProtocol:
        """Deprecated!"""
        warnings.warn(
            f"Accessing the surrogate model via 'surrogate_model' has been "
            f"deprecated. Use '{self.get_surrogate.__name__}' instead to get the "
            f"trained model instance (or "
            f"'{fields(type(self))._surrogate_model.name}' to access the raw object).",
            DeprecationWarning,
        )
        return self._surrogate_model

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

    def _setup_botorch_acqf(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
    ) -> None:
        """Create the acquisition function for the current training data."""  # noqa: E501
        surrogate = self.get_surrogate(searchspace, objective, measurements)
        self._botorch_acqf = self.acquisition_function.to_botorch(
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments,
        )

    def recommend(  # noqa: D102
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # See base class.

        if objective is None:
            raise NotImplementedError(
                f"Recommenders of type '{BayesianRecommender.__name__}' require "
                f"that an objective is specified."
            )

        if (measurements is None) or (len(measurements) == 0):
            raise NotImplementedError(
                f"Recommenders of type '{BayesianRecommender.__name__}' do not support "
                f"empty training data."
            )

        if (
            isinstance(self._surrogate_model, IndependentGaussianSurrogate)
            and batch_size > 1
        ):
            raise InvalidSurrogateModelError(
                f"The specified surrogate model of type "
                f"'{self._surrogate_model.__class__.__name__}' "
                f"cannot be used for batch recommendation."
            )

        if isinstance(self._surrogate_model, CustomONNXSurrogate):
            CustomONNXSurrogate.validate_compatibility(searchspace)

        self._setup_botorch_acqf(
            searchspace, objective, measurements, pending_experiments
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
