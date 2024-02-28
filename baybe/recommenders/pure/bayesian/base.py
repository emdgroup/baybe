"""Base class for all Bayesian recommenders."""

from abc import ABC
from functools import partial
from typing import Callable, Literal, Optional

import pandas as pd
from attrs import define, field
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)

from baybe.acquisition import debotorchize
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

    acquisition_function_cls: Literal[
        "PM", "PI", "EI", "UCB", "qPI", "qEI", "qUCB", "VarUCB", "qVarUCB"
    ] = field(default="qEI")
    """The used acquisition function class."""

    _acquisition_function: Optional[AcquisitionFunction] = field(
        default=None, init=False
    )
    """The current acquisition function."""

    def _get_acquisition_function_cls(
        self,
    ) -> Callable:
        """Get the actual acquisition function class.

        Returns:
            The debotorchized acquisition function class.
        """
        mapping = {
            "PM": PosteriorMean,
            "PI": ProbabilityOfImprovement,
            "EI": ExpectedImprovement,
            "UCB": partial(UpperConfidenceBound, beta=1.0),
            "qEI": qExpectedImprovement,
            "qPI": qProbabilityOfImprovement,
            "qUCB": partial(qUpperConfidenceBound, beta=1.0),
            "VarUCB": partial(UpperConfidenceBound, beta=100.0),
            "qVarUCB": partial(qUpperConfidenceBound, beta=100.0),
        }
        fun = debotorchize(mapping[self.acquisition_function_cls])
        return fun

    def setup_acquisition_function(
        self,
        searchspace: SearchSpace,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> None:
        """Create the current acquisition function from provided training data.

        The acquisition function is stored in the private attribute
        ``_acquisition_function``.

        Args:
            searchspace: The search space in which the experiments are to be conducted.
            train_x: The features of the conducted experiments.
            train_y: The corresponding response values.

        Raises:
            NotImplementedError: If the setup is attempted from empty training data
        """
        if train_x is None or train_y is None:
            raise NotImplementedError(
                "Bayesian recommenders do not support empty training data yet."
            )

        best_f = train_y.max()
        surrogate_model = self._fit(searchspace, train_x, train_y)
        acquisition_function_cls = self._get_acquisition_function_cls()

        self._acquisition_function = acquisition_function_cls(surrogate_model, best_f)

    def _fit(
        self,
        searchspace: SearchSpace,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
    ) -> Surrogate:
        """Train a fresh surrogate model instance.

        Args:
            searchspace: The search space.
            train_x: The features of the conducted experiments.
            train_y: The corresponding response values.

        Returns:
            A surrogate model fitted to the provided data.

        Raises:
            ValueError: If the training inputs and targets do not have the same index.
        """
        # validate input
        if not train_x.index.equals(train_y.index):
            raise ValueError("Training inputs and targets must have the same index.")

        self.surrogate_model.fit(searchspace, *to_tensor(train_x, train_y))

        return self.surrogate_model

    def recommend(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_size: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        if _ONNX_INSTALLED and isinstance(self.surrogate_model, CustomONNXSurrogate):
            CustomONNXSurrogate.validate_compatibility(searchspace)

        self.setup_acquisition_function(searchspace, train_x, train_y)

        return super().recommend(searchspace, batch_size, train_x, train_y)
