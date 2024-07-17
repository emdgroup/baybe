"""Base functionality for all BayBE surrogates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from attrs import define, field
from cattrs import override
from cattrs.dispatch import (
    StructuredValue,
    StructureHook,
    TargetType,
    UnstructuredValue,
    UnstructureHook,
)

from baybe.exceptions import ModelNotTrainedError
from baybe.objectives.base import Objective
from baybe.parameters.base import Parameter
from baybe.searchspace import SearchSpace
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.utils.dataframe import to_tensor
from baybe.utils.scaling import ScalingMethod, make_scaler

if TYPE_CHECKING:
    from botorch.models.model import Model
    from botorch.posteriors import GPyTorchPosterior, Posterior
    from sklearn.compose import ColumnTransformer
    from torch import Tensor

_ONNX_ENCODING = "latin-1"
"""Constant signifying the encoding for onnx byte strings in pretrained models.

NOTE: This encoding is selected by choice for ONNX byte strings.
This is not a requirement from ONNX but simply for the JSON format.
The byte string from ONNX `.SerializeToString()` method has unknown encoding,
which results in UnicodeDecodeError when using `.decode('utf-8')`.
The use of latin-1 ensures there are no loss from the conversion of
bytes to string and back, since the specification is a bijection between
0-255 and the character set.
"""


@define
class Surrogate(ABC, SerialMixin):
    """Abstract base class for all surrogate models."""

    # Class variables
    joint_posterior: ClassVar[bool]
    """Class variable encoding whether or not a joint posterior is calculated."""

    supports_transfer_learning: ClassVar[bool]
    """Class variable encoding whether or not the surrogate supports transfer
    learning."""

    _input_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = field(
        init=False, default=None, eq=False
    )
    """Callable preparing surrogate inputs for training/prediction.

    Transforms a dataframe containing parameter configurations in experimental
    representation to a corresponding dataframe containing their computational
    representation. Only available after the surrogate has been fitted."""

    _target_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = field(
        init=False, default=None, eq=False
    )
    """Callable preparing surrogate targets for training.

    Transforms a dataframe containing target measurements in experimental
    representation to a corresponding dataframe containing their computational
    representation. Only available after the surrogate has been fitted."""

    def to_botorch(self) -> Model:
        """Create the botorch-ready representation of the model."""
        from baybe.surrogates._adapter import AdapterModel

        return AdapterModel(self)

    @staticmethod
    def _get_parameter_scaling(parameter: Parameter) -> ScalingMethod:
        """Return the scaling method to be used for the given parameter."""
        return ScalingMethod.MINMAX

    def _make_input_scaler(
        self, searchspace: SearchSpace, measurements: pd.DataFrame
    ) -> ColumnTransformer:
        """Make a scaler to be used for transforming computational dataframes."""
        from sklearn.compose import make_column_transformer

        # Create the composite scaler from the parameter-wise scaler objects
        # TODO: Filter down to columns that actually remain in the comp rep of the
        #   searchspace, since the transformer can break down otherwise.
        transformers = [
            (make_scaler(self._get_parameter_scaling(p)), p.comp_df.columns)
            for p in searchspace.parameters
        ]
        scaler = make_column_transformer(*transformers)

        # TODO: Decide whether scaler is to be fit to parameter bounds and/or
        #   extreme points in the given measurement data
        scaler.fit(searchspace.comp_rep_bounds)

        return scaler

    def transform_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform an experimental parameter dataframe."""
        if self._input_transform is None:
            raise ModelNotTrainedError("The model must be trained first.")
        return self._input_transform(data)

    def transform_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform an experimental measurement dataframe."""
        if self._target_transform is None:
            raise ModelNotTrainedError("The model must be trained first.")
        return self._target_transform(data)

    def posterior(self, candidates: pd.DataFrame) -> Posterior:
        """Evaluate the surrogate model at the given candidate points."""
        return self._posterior(to_tensor(self.transform_inputs(candidates)))

    @abstractmethod
    def _posterior(self, candidates: Tensor) -> Posterior:
        """Perform the actual posterior evaluation logic."""

    @staticmethod
    def _get_model_context(searchspace: SearchSpace, objective: Objective) -> Any:
        """Get the surrogate-specific context for model fitting.

        By default, no context is created. If context is required, subclasses are
        expected to override this method.
        """
        return None

    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Train the surrogate model on the provided data.

        Args:
            searchspace: The search space in which experiments are conducted.
            objective: The objective to be optimized.
            measurements: The training data in experimental representation.

        Raises:
            ValueError: If the search space contains task parameters but the selected
                surrogate model type does not support transfer learning.
            NotImplementedError: When using a continuous search space and a non-GP
                model.
        """
        # TODO: consider adding a validation step for `measurements`

        # Check if transfer learning capabilities are needed
        if (searchspace.n_tasks > 1) and (not self.supports_transfer_learning):
            raise ValueError(
                f"The search space contains task parameters but the selected "
                f"surrogate model type ({self.__class__.__name__}) does not "
                f"support transfer learning."
            )
        # TODO: Adjust scale_model decorator to support other model types as well.
        if (not searchspace.continuous.is_empty) and (
            "GaussianProcess" not in self.__class__.__name__
        ):
            raise NotImplementedError(
                "Continuous search spaces are currently only supported by GPs."
            )

        input_scaler = self._make_input_scaler(searchspace, measurements)

        # Store context-specific transformations
        self._input_transform = lambda x: input_scaler.transform(
            searchspace.transform(x, allow_missing=True)
        )
        self._target_transform = lambda x: objective.transform(x)

        # Transform and fit
        train_x, train_y = to_tensor(
            self.transform_inputs(measurements),
            self.transform_targets(measurements),
        )
        self._fit(train_x, train_y, self._get_model_context(searchspace, objective))

    @abstractmethod
    def _fit(self, train_x: Tensor, train_y: Tensor, context: Any = None) -> None:
        """Perform the actual fitting logic."""


@define
class GaussianSurrogate(Surrogate, ABC):
    """A surrogate model providing Gaussian posterior estimates."""

    def _posterior(self, candidates: Tensor) -> GPyTorchPosterior:
        # See base class.

        import torch
        from botorch.posteriors import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        # Construct the Gaussian posterior from the estimated first and second moment
        mean, var = self._estimate_moments(candidates)
        if not self.joint_posterior:
            var = torch.diag_embed(var)
        mvn = MultivariateNormal(mean, var)
        return GPyTorchPosterior(mvn)

    @abstractmethod
    def _estimate_moments(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        """Estimate first and second moments of the Gaussian posterior.

        The second moment may either be a 1-D tensor of marginal variances for the
        candidates or a 2-D tensor representing a full covariance matrix over all
        candidates, depending on the ``joint_posterior`` flag of the model.
        """


def _make_hook_decode_onnx_str(
    raw_unstructure_hook: UnstructureHook
) -> UnstructureHook:
    """Wrap an unstructuring hook to let it also decode the contained ONNX string."""

    def wrapper(obj: StructuredValue) -> UnstructuredValue:
        dct = raw_unstructure_hook(obj)
        if "onnx_str" in dct:
            dct["onnx_str"] = dct["onnx_str"].decode(_ONNX_ENCODING)

        return dct

    return wrapper


def _make_hook_encode_onnx_str(raw_structure_hook: StructureHook) -> StructureHook:
    """Wrap a structuring hook to let it also encode the contained ONNX string."""

    def wrapper(dct: UnstructuredValue, _: TargetType) -> StructuredValue:
        if (onnx_str := dct.get("onnx_str")) and isinstance(onnx_str, str):
            dct["onnx_str"] = onnx_str.encode(_ONNX_ENCODING)
        obj = raw_structure_hook(dct, _)

        return obj

    return wrapper


def _block_serialize_custom_architecture(
    raw_unstructure_hook: UnstructureHook
) -> UnstructureHook:
    """Raise error if attempt to serialize a custom architecture surrogate."""
    # TODO: Ideally, this hook should be removed and unstructuring the Surrogate
    #   base class should automatically invoke the blocking hook that is already
    #   registered for the "CustomArchitectureSurrogate" subclass. However, it's
    #   not clear how the base unstructuring hook needs to be modified to accomplish
    #   this, and furthermore the problem will most likely become obsolete in the future
    #   because the role of the subclass will probably be replaced with a surrogate
    #   protocol.

    def wrapper(obj: StructuredValue) -> UnstructuredValue:
        if obj.__class__.__name__ == "CustomArchitectureSurrogate":
            raise NotImplementedError(
                "Serializing objects of type 'CustomArchitectureSurrogate' "
                "is not supported."
            )

        return raw_unstructure_hook(obj)

    return wrapper


# Register (un-)structure hooks
# IMPROVE: Ideally, the ONNX-specific hooks should simply be registered with the ONNX
#   class, which would avoid the nested wrapping below. However, this requires
#   adjusting the base class (un-)structure hooks such that they consistently apply
#   existing hooks of the concrete subclasses.
converter.register_unstructure_hook(
    Surrogate,
    _make_hook_decode_onnx_str(
        _block_serialize_custom_architecture(
            lambda x: unstructure_base(x, overrides={"_model": override(omit=True)})
        )
    ),
)
converter.register_structure_hook(
    Surrogate, _make_hook_encode_onnx_str(get_base_structure_hook(Surrogate))
)
