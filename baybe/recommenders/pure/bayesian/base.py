"""Base class for all Bayesian recommenders."""

import gc
import warnings
from abc import ABC

import pandas as pd
from attrs import define, field, fields
from attrs.converters import optional
from typing_extensions import override

from baybe.acquisition import qLogEI, qLogNEHVI
from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import (
    DeprecationError,
    IncompatibleAcquisitionFunctionError,
    InvalidSurrogateModelError,
)
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import CustomONNXSurrogate, GaussianProcessSurrogate
from baybe.surrogates.base import (
    IndependentGaussianSurrogate,
    Surrogate,
    SurrogateProtocol,
)
from baybe.utils.dataframe import _ValidatedDataFrame
from baybe.utils.validation import validate_parameter_input, validate_target_input


def _autobroadcast(surrogate: SurrogateProtocol, /) -> SurrogateProtocol:
    """Broadcasts single-output surrogate models and passes through everything else."""
    if isinstance(surrogate, Surrogate) and not surrogate.supports_multi_output:
        return surrogate.broadcast()
    return surrogate


@define
class BayesianRecommender(PureRecommender, ABC):
    """An abstract class for Bayesian Recommenders."""

    _surrogate_model: SurrogateProtocol = field(
        alias="surrogate_model",
        factory=GaussianProcessSurrogate,
        converter=_autobroadcast,
    )
    """The used surrogate model."""

    acquisition_function: AcquisitionFunction | None = field(
        default=None, converter=optional(convert_acqf)
    )
    """The user-specified acquisition function. When omitted, a default is used."""

    _acqf: AcquisitionFunction | None = field(default=None, init=False, eq=False)
    """The used acquisition function."""

    _botorch_acqf = field(default=None, init=False, eq=False)
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

    def _default_acquisition_function(
        self, objective: Objective
    ) -> AcquisitionFunction:
        """Select the appropriate default acquisition function for the given context."""
        if self.acquisition_function is None:
            return qLogNEHVI() if objective.is_multi_output else qLogEI()
        return self.acquisition_function

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
        self._acqf = self._default_acquisition_function(objective)

        if objective.is_multi_output and not self._acqf.supports_multi_output:
            raise IncompatibleAcquisitionFunctionError(
                f"You attempted to use a single-target acquisition function in a "
                f"{len(objective.targets)}-target context."
            )

        surrogate = self.get_surrogate(searchspace, objective, measurements)
        self._botorch_acqf = self._acqf.to_botorch(
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments,
        )

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
                f"Recommenders of type '{BayesianRecommender.__name__}' require "
                f"that an objective is specified."
            )

        # Experimental input validation
        if (measurements is None) or measurements.empty:
            raise NotImplementedError(
                f"Recommenders of type '{BayesianRecommender.__name__}' do not support "
                f"empty training data."
            )
        if not isinstance(measurements, _ValidatedDataFrame):
            validate_target_input(measurements, objective.targets)
            validate_parameter_input(measurements, searchspace.parameters)
            measurements.__class__ = _ValidatedDataFrame
        if pending_experiments is not None and not isinstance(
            pending_experiments, _ValidatedDataFrame
        ):
            validate_parameter_input(pending_experiments, searchspace.parameters)
            pending_experiments.__class__ = _ValidatedDataFrame

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
