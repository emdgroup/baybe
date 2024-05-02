"""Base functionality for all BayBE surrogates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from attrs import define
from cattrs import override
from cattrs.dispatch import (
    StructuredValue,
    StructureHook,
    TargetType,
    UnstructuredValue,
    UnstructureHook,
)

from baybe.searchspace import SearchSpace
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.utils import _prepare_inputs, _prepare_targets

if TYPE_CHECKING:
    from torch import Tensor

# Define constants
_MIN_VARIANCE = 1e-6

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

    def posterior(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the surrogate model at the given candidate points.

        Args:
            candidates: The candidate points, represented as a tensor of shape
                ``(*t, q, d)``, where ``t`` denotes the "t-batch" shape, ``q``
                denotes the "q-batch" shape, and ``d`` is the input dimension. For
                more details about batch shapes, see: https://botorch.org/docs/batching

        Returns:
            The posterior means and posterior covariance matrices of the t-batched
            candidate points.
        """
        import torch

        # Prepare the input
        candidates = _prepare_inputs(candidates)

        # Evaluate the posterior distribution
        mean, covar = self._posterior(candidates)

        # Apply covariance transformation for marginal posterior models
        if not self.joint_posterior:
            # Convert to tensor containing covariance matrices
            covar = torch.diag_embed(covar)

        # Add small diagonal variances for numerical stability
        covar.add_(torch.eye(covar.shape[-1]) * _MIN_VARIANCE)

        return mean, covar

    @abstractmethod
    def _posterior(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        """Perform the actual posterior evaluation logic.

        In contrast to its public counterpart
        :func:`baybe.surrogates.Surrogate.posterior`, no data
        validation/transformation is carried out but only the raw posterior computation
        is conducted.

        Note that the public ``posterior`` method *always* returns a full covariance
        matrix. By contrast, this method may return either a covariance matrix or a
        tensor of marginal variances, depending on the models ``joint_posterior``
        flag. The optional conversion to a covariance matrix is handled by the public
        method.

        See :func:`baybe.surrogates.Surrogate.posterior` for details on the
        parameters.

        Args:
            candidates: The candidates.

        Returns:
            See :func:`baybe.surrogates.Surrogate.posterior`.
        """

    def fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """Train the surrogate model on the provided data.

        Args:
            searchspace: The search space in which experiments are conducted.
            train_x: The training data points.
            train_y: The training data labels.

        Raises:
            ValueError: If the search space contains task parameters but the selected
                surrogate model type does not support transfer learning.
            NotImplementedError: When using a continuous search space and a non-GP
                model.
        """
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

        # Validate and prepare the training data
        train_x = _prepare_inputs(train_x)
        train_y = _prepare_targets(train_y)

        self._fit(searchspace, train_x, train_y)

    @abstractmethod
    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """Perform the actual fitting logic.

        In contrast to its public counterpart :func:`baybe.surrogates.Surrogate.fit`,
        no data validation/transformation is carried out but only the raw fitting
        operation is conducted.

        See :func:`baybe.surrogates.Surrogate.fit` for details on the parameters.
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
