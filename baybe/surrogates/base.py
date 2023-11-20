"""Base functionality for all BayBE surrogates."""

import gc
import sys
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Tuple, Type

import torch
from attr import define, field
from torch import Tensor

from baybe.searchspace import SearchSpace
from baybe.surrogates.utils import _prepare_inputs, _prepare_targets
from baybe.utils import SerialMixin
from baybe.utils.serialization import converter, get_subclasses, unstructure_base

# Define constants
_MIN_VARIANCE = 1e-6
_WRAPPER_MODELS = (
    "SplitModel",
    "ScaledModel",
    "CustomArchitectureSurrogate",
    "CustomONNXSurrogate",
)

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

    # Object variables
    # TODO: In a next refactoring, the user friendliness could be improved by directly
    #   exposing the individual model parameters via the constructor, instead of
    #   expecting them in the form of an unstructured dictionary. This would also
    #   remove the need for the current `_get_model_params_validator` logic.
    model_params: Dict[str, Any] = field(factory=dict)
    """Optional model parameters."""

    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
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
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
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

        return self._fit(searchspace, train_x, train_y)

    @abstractmethod
    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """Perform the actual fitting logic.

        In contrast to its public counterpart :func:`baybe.surrogates.Surrogate.fit`,
        no data validation/transformation is carried out but only the raw fitting
        operation is conducted.

        See :func:`baybe.surrogates.Surrogate.fit` for details on the parameters.
        """


def _decode_onnx_str(raw_unstructure_hook):
    """Decode ONNX string for serialization purposes."""

    def wrapper(obj):
        dict_ = raw_unstructure_hook(obj)
        if "onnx_str" in dict_:
            dict_["onnx_str"] = dict_["onnx_str"].decode(_ONNX_ENCODING)

        return dict_

    return wrapper


def _block_serialize_custom_architecture(raw_unstructure_hook):
    """Raise error if attempt to serialize a custom architecture surrogate."""
    # TODO: Should be replaced with `serialization.block_serialization_hook`.
    #   However, the class definition of `CustomArchitectureSurrogate` is needs
    #   to be fixed first, which is broken due to the handling of `model_params`.
    #   To reproduce the problem, run for example `custom_architecture_torch` and
    #   try to print the created surrogate model object.

    def wrapper(obj):
        if obj.__class__.__name__ == "CustomArchitectureSurrogate":
            raise NotImplementedError(
                "Custom Architecture Surrogate Serialization is not supported"
            )

        return raw_unstructure_hook(obj)

    return wrapper


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Temporary workaround >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def _structure_surrogate(val, _):
    """Structure a surrogate model."""
    # TODO [15436]
    # https://***REMOVED***/_boards/board/t/SDK%20Devs/Features/?workitem=15436

    # NOTE:
    # Due to above issue,
    # it is difficult to find the wrapped class in the subclass structure.
    # The renaming only happens in the init method of wrapper classes
    # (classes that haven't been initialized won't have the overwritten name)
    # Since any method revolving `cls()` will not work as expected,
    # we rely temporarily on `getattr` to allow the wrappers to be called on demand.

    _type = val["type"]

    cls = getattr(sys.modules[__package__], _type, None)
    # cls = getattr(baybe.surrogates, ...) if used in another module

    if cls is None:
        raise ValueError(f"Unknown subclass {_type}.")

    # NOTE:
    # This is a workaround for onnx str type being `str` or `bytes`
    onnx_str = val.get("onnx_str", None)
    if onnx_str and isinstance(onnx_str, str):
        val["onnx_str"] = onnx_str.encode(_ONNX_ENCODING)

    return converter.structure_attrs_fromdict(val, cls)


def get_available_surrogates() -> List[Type[Surrogate]]:
    """List all available surrogate models.

    Returns:
        A list of available surrogate classes.
    """
    # List available names
    available_names = {
        cl.__name__
        for cl in get_subclasses(Surrogate)
        if cl.__name__ not in _WRAPPER_MODELS
    }

    # Convert them to classes
    available_classes = [
        getattr(sys.modules[__package__], mdl_name, None)
        for mdl_name in available_names
    ]

    # TODO: The initialization of the classes is currently necessary for the renaming
    #  to take place (see [15436] and NOTE in `structure_surrogate`).
    [cl() for cl in available_classes if cl is not None]

    return [cl for cl in available_classes if cl is not None]


# Register (un-)structure hooks
# TODO: Needs to be refactored
converter.register_unstructure_hook(
    Surrogate,
    _decode_onnx_str(_block_serialize_custom_architecture(unstructure_base)),
)
converter.register_structure_hook(Surrogate, _structure_surrogate)

# Related to [15436]
gc.collect()
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Temporary workaround <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
