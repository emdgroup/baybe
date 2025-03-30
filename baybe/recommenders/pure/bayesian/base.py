"""Base class for all Bayesian recommenders."""

import gc
import warnings
from abc import ABC

import pandas as pd
import torch
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
from baybe.utils.device_utils import clear_gpu_memory, device_context, device_mode
from baybe.utils.validation import validate_parameter_input, validate_target_input


def _autoreplicate(surrogate: SurrogateProtocol, /) -> SurrogateProtocol:
    """Replicates single-output surrogate models and passes through everything else."""
    if isinstance(surrogate, Surrogate) and not surrogate.supports_multi_output:
        return surrogate.replicate()
    return surrogate


@define
class BayesianRecommender(PureRecommender, ABC):
    """An abstract class for Bayesian Recommenders."""

    _surrogate_model: SurrogateProtocol = field(
        alias="surrogate_model", factory=GaussianProcessSurrogate
    )
    """The surrogate model."""

    acquisition_function: AcquisitionFunction | None = field(
        default=None, converter=optional(convert_acqf)
    )
    """The acquisition function. When omitted, a default is used."""

    # TODO: The objective is currently only required for validating the recommendation
    #   context. Once multi-target support is complete, we might want to refactor
    #   the validation mechanism, e.g. by
    #   * storing only the minimal low-level information required
    #   * switching to a strategy where we catch the BoTorch exceptions
    #   * ...
    _objective: Objective | None = field(default=None, init=False, eq=False)
    """The encountered objective to be optimized."""

    _botorch_acqf = field(default=None, init=False, eq=False)
    """The induced BoTorch acquisition function."""

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

    def _get_acquisition_function(self, objective: Objective) -> AcquisitionFunction:
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
        surrogate = (
            _autoreplicate(self._surrogate_model)
            if objective.is_multi_output
            else self._surrogate_model
        )
        surrogate.fit(searchspace, objective, measurements)

        return surrogate

    def _setup_botorch_acqf(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
    ) -> None:
        """Create the acquisition function for the current training data."""
        # Use device_mode to ensure consistent device handling during setup
        with device_mode(True):
            self._objective = objective
            acqf = self._get_acquisition_function(objective)

            if objective.is_multi_output and not acqf.supports_multi_output:
                raise IncompatibleAcquisitionFunctionError(
                    f"You attempted to use a single-output acquisition function in a "
                    f"{len(objective.targets)}-target multi-output context."
                )

            surrogate = self.get_surrogate(searchspace, objective, measurements)

            # Clear CUDA cache before creating acquisition function
            clear_gpu_memory()

            self._botorch_acqf = acqf.to_botorch(
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
        # Get the device if this recommender has one
        device = getattr(self, "device", None)

        # Use device_context for consistent device management
        with device_context(device):
            if objective is None:
                raise NotImplementedError(
                    f"Recommenders of type '{BayesianRecommender.__name__}' require "
                    f"that an objective is specified."
                )

            # Experimental input validation
            if (measurements is None) or measurements.empty:
                raise NotImplementedError(
                    f"Recommenders of type '{BayesianRecommender.__name__}' "
                    f"do not support "
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

            # Clear caches before setup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
