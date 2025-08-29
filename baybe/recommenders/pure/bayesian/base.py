"""Base class for all Bayesian recommenders."""

from __future__ import annotations

import gc
import warnings
from abc import ABC
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define, field, fields
from attrs.converters import optional
from typing_extensions import override

from baybe.acquisition import qLogEI, qLogNEHVI
from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import (
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
from baybe.utils.dataframe import _ValidatedDataFrame, normalize_input_dtypes
from baybe.utils.validation import (
    validate_object_names,
    validate_objective_input,
    validate_parameter_input,
    validate_target_input,
)

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction


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
            if objective._is_multi_model
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
        """Create the acquisition function for the current training data."""  # noqa: E501
        self._objective = objective
        acqf = self._get_acquisition_function(objective)

        if objective.is_multi_output and not acqf.supports_multi_output:
            raise IncompatibleAcquisitionFunctionError(
                f"You attempted to use a single-output acquisition function in a "
                f"{len(objective.targets)}-target multi-output context."
            )

        surrogate = self.get_surrogate(searchspace, objective, measurements)
        self._botorch_acqf = acqf.to_botorch(
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments,
        )

    def get_acquisition_function(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
    ) -> BoAcquisitionFunction:
        """Get the acquisition function for the given recommendation context.

        For details on the method arguments, see :meth:`recommend`.
        """
        self._setup_botorch_acqf(
            searchspace, objective, measurements, pending_experiments
        )
        return self._botorch_acqf

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

        validate_object_names(searchspace.parameters + objective.targets)

        # Experimental input validation
        if (measurements is None) or measurements.empty:
            raise NotImplementedError(
                f"Recommenders of type '{BayesianRecommender.__name__}' do not support "
                f"empty training data."
            )
        if not isinstance(measurements, _ValidatedDataFrame):
            validate_target_input(measurements, objective.targets)
            validate_objective_input(measurements, objective)
            validate_parameter_input(measurements, searchspace.parameters)
            measurements = normalize_input_dtypes(
                measurements, searchspace.parameters + objective.targets
            )
            measurements.__class__ = _ValidatedDataFrame
        if pending_experiments is not None and not isinstance(
            pending_experiments, _ValidatedDataFrame
        ):
            validate_parameter_input(pending_experiments, searchspace.parameters)
            pending_experiments = normalize_input_dtypes(
                pending_experiments, searchspace.parameters
            )
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

    def acquisition_values(
        self,
        candidates: pd.DataFrame,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
        acquisition_function: AcquisitionFunction | None = None,
    ) -> pd.Series:
        """Compute the acquisition values for the given candidates.

        Args:
            candidates: The candidate points in experimental representation.
                For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
            searchspace:
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            objective:
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            measurements:
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            pending_experiments:
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            acquisition_function: The acquisition function to be evaluated.
                If not provided, the acquisition function of the recommender is used.

        Returns:
            A series of individual acquisition values, one for each candidate.
        """
        surrogate = self.get_surrogate(searchspace, objective, measurements)
        acqf = acquisition_function or self._get_acquisition_function(objective)
        return acqf.evaluate(
            candidates,
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments,
            jointly=False,
        )

    def joint_acquisition_value(  # noqa: DOC101, DOC103
        self,
        candidates: pd.DataFrame,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
        acquisition_function: AcquisitionFunction | None = None,
    ) -> float:
        """Compute the joint acquisition value for the given candidate batch.

        For details on the method arguments, see :meth:`acquisition_values`.

        Returns:
            The joint acquisition value of the batch.
        """
        surrogate = self.get_surrogate(searchspace, objective, measurements)
        acqf = acquisition_function or self._get_acquisition_function(objective)
        return acqf.evaluate(
            candidates,
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments,
            jointly=True,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
