"""Base class for all Bayesian recommenders."""

from abc import ABC
from typing import Optional

import pandas as pd
from attrs import define, field

from baybe.acquisition.acqfs import qExpectedImprovement
from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import DeprecationError
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import _ONNX_INSTALLED, GaussianProcessSurrogate
from baybe.surrogates.base import Surrogate
from baybe.utils.dataframe import to_tensor

if _ONNX_INSTALLED:
    from baybe.surrogates import CustomONNXSurrogate


@define
class BayesianRecommender(PureRecommender, ABC):
    """An abstract class for Bayesian Recommenders."""

    surrogate_model: Surrogate = field(factory=GaussianProcessSurrogate)
    """The used surrogate model."""

    acquisition_function: AcquisitionFunction = field(
        converter=convert_acqf, factory=qExpectedImprovement, kw_only=True
    )
    """The used acquisition function class."""

    _botorch_acqf = field(default=None, init=False)
    """The current acquisition function."""

    acquisition_function_cls: bool = field(default=None)
    "Deprecated! Raises an error when used."

    @acquisition_function_cls.validator
    def _validate_deprecated_argument(self, _, value) -> None:
        """Raise DeprecationError if old acquisition_function_cls parameter is used."""
        if value is not None:
            raise DeprecationError(
                "Passing 'acquisition_function_cls' to the constructor is deprecated. "
                "The parameter has been renamed to 'acquisition_function'."
            )

    def _setup_botorch_acqf(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: Optional[pd.DataFrame] = None,
    ) -> None:
        """Create the acquisition function for the current training data.

        The acquisition function is stored in the private attribute
        ``_acquisition_function``.

        Args:
            searchspace:
                See :meth:`baybe.recommenders.pure.bayesian.BayesianRecommender.recommend`.
            objective:
                See :meth:`baybe.recommenders.pure.bayesian.BayesianRecommender.recommend`.
            measurements:
                See :meth:`baybe.recommenders.pure.bayesian.BayesianRecommender.recommend`.
        """  # noqa: E501
        # TODO: Transition point from dataframe to tensor needs to be refactored.
        #   Currently, surrogate models operate with tensors, while acquisition
        #   functions with dataframes.
        train_x = searchspace.transform(measurements)
        train_y = objective.transform(measurements)
        surrogate_model = self.surrogate_model._fit(
            searchspace, *to_tensor(train_x, train_y)
        )
        self._botorch_acqf = self.acquisition_function.to_botorch(
            surrogate_model, train_x, train_y
        )

    def recommend(  # noqa: D102
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.
        if measurements is None:
            raise NotImplementedError(
                "Bayesian recommenders do not support empty training data yet."
            )

        if _ONNX_INSTALLED and isinstance(self.surrogate_model, CustomONNXSurrogate):
            CustomONNXSurrogate.validate_compatibility(searchspace)

        self._setup_botorch_acqf(searchspace, objective, measurements)

        return super().recommend(searchspace, objective, batch_size, measurements)
