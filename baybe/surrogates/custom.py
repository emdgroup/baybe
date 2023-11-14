"""Functionality for building custom surrogates.

Note that ONNX surrogate models cannot be retrained. However, having the surrogates
raise a ``NotImplementedError`` would currently break the code since
:class:`baybe.recommenders.bayesian.BayesianRecommender` assumes that surrogates can be
trained and attempts to do so for each new DOE iteration.

It is planned to solve this issue in the future.
"""

from typing import Callable, ClassVar, Tuple

import torch
from attrs import define, field, validators
from torch import Tensor

from baybe.exceptions import ModelParamsNotSupportedError
from baybe.parameters import (
    CategoricalEncoding,
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.utils import batchify, catch_constant_targets
from baybe.surrogates.validation import validate_custom_architecture_cls
from baybe.utils import DTypeFloatONNX, DTypeFloatTorch

try:
    import onnxruntime as ort

    _ONNX_INSTALLED = True
except ImportError:
    _ONNX_INSTALLED = False


def register_custom_architecture(
    joint_posterior_attr: bool = False,
    constant_target_catching: bool = True,
    batchify_posterior: bool = True,
) -> Callable:
    """Wrap a given custom model architecture class into a ```Surrogate```.

    Args:
        joint_posterior_attr: Boolean indicating if the model returns a posterior
            distribution jointly across candidates or on individual points.
        constant_target_catching: Boolean indicating if the model cannot handle
            constant target values and needs the @catch_constant_targets decorator.
        batchify_posterior: Boolean indicating if the model is incompatible
            with t- and q-batching and needs the @batchify decorator for its posterior.

    Returns:
        A function that wraps around a model class based on the specifications.
    """

    def construct_custom_architecture(model_cls):
        """Construct a surrogate class wrapped around the custom class."""
        validate_custom_architecture_cls(model_cls)

        class CustomArchitectureSurrogate(Surrogate):
            """Wraps around a custom architecture class."""

            joint_posterior: ClassVar[bool] = joint_posterior_attr
            supports_transfer_learning: ClassVar[bool] = False

            def __init__(self, *args, **kwargs):
                self.model = model_cls(*args, **kwargs)

            def _fit(
                self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
            ) -> None:
                return self.model._fit(searchspace, train_x, train_y)

            def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
                return self.model._posterior(candidates)

            def __get_attribute__(self, attr):
                """Access the attributes of the class instance if available.

                If the attributes are not available,
                it uses the attributes of the internal model instance.
                """
                # Try to retrieve the attribute in the class
                try:
                    val = super().__getattribute__(attr)
                except AttributeError:
                    pass
                else:
                    return val

                # If the attribute is not overwritten, use that of the internal model
                return self.model.__getattribute__(attr)

        # Catch constant targets if needed
        cls = (
            catch_constant_targets(CustomArchitectureSurrogate)
            if constant_target_catching
            else CustomArchitectureSurrogate
        )

        # batchify posterior if needed
        if batchify_posterior:
            cls._posterior = batchify(cls._posterior)

        return cls

    return construct_custom_architecture


if _ONNX_INSTALLED:

    @define(kw_only=True)
    class CustomONNXSurrogate(Surrogate):
        """A wrapper class for custom pretrained surrogate models.

        Note that these surrogates cannot be retrained.
        """

        # Class variables
        joint_posterior: ClassVar[bool] = False
        # See base class.

        supports_transfer_learning: ClassVar[bool] = False
        # See base class.

        # Object variables
        onnx_input_name: str = field(validator=validators.instance_of(str))
        """The input name used for constructing the ONNX str."""

        onnx_str: bytes = field(validator=validators.instance_of(bytes))
        """The ONNX byte str representing the model."""

        _model: ort.InferenceSession = field(init=False, eq=False)
        """The internal model."""

        @_model.default
        def default_model(self) -> ort.InferenceSession:
            """Instantiate the ONNX inference session."""
            try:
                return ort.InferenceSession(self.onnx_str)
            except Exception as exc:
                raise ValueError("Invalid ONNX string") from exc

        def __attrs_post_init__(self) -> None:
            # TODO: This is a temporary workaround to avoid silent errors when users
            #   provide model parameters to this class.
            if self.model_params or not isinstance(self.model_params, dict):
                raise ModelParamsNotSupportedError()

        @batchify
        def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
            model_inputs = {
                self.onnx_input_name: candidates.numpy().astype(DTypeFloatONNX)
            }
            results = self._model.run(None, model_inputs)

            # IMPROVE: At the moment, we assume that the second model output contains
            #   standard deviations. Currently, most available ONNX converters care
            #   about the mean only and it's not clear how this will be handled in the
            #   future. Once there are more choices available, this should be revisited.
            return (
                torch.from_numpy(results[0]).to(DTypeFloatTorch),
                torch.from_numpy(results[1]).pow(2).to(DTypeFloatTorch),
            )

        def _fit(
            self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
        ) -> None:
            # TODO: This method actually needs to raise a NotImplementedError because
            #   ONNX surrogate models cannot be retrained. However, this would currently
            #   break the code since `BayesianRecommender` assumes that surrogates
            #   can be trained and attempts to do so for each new DOE iteration.
            #   Therefore, a refactoring is required in order to properly incorporate
            #   "static" surrogates and account for them in the exposed APIs.
            pass

        @classmethod
        def validate_compatibility(cls, searchspace: SearchSpace) -> None:
            """Validate if the class is compatible with a given search space.

            Args:
                searchspace: The search space to be tested for compatibility.

            Raises:
                TypeError: If the search space is incompatible with the class.
            """
            if not all(
                isinstance(
                    p,
                    (
                        NumericalContinuousParameter,
                        NumericalDiscreteParameter,
                        TaskParameter,
                    ),
                )
                or (isinstance(p, CustomDiscreteParameter) and not p.decorrelate)
                or (
                    isinstance(p, CategoricalParameter)
                    and p.encoding is CategoricalEncoding.INT
                )
                for p in searchspace.parameters
            ):
                raise TypeError(
                    f"To prevent potential hard-to-detect bugs that stem from wrong "
                    f"wiring of model inputs, {cls.__name__} "
                    f"is currently restricted for use with parameters that have "
                    f"a one-dimensional computational representation or "
                    f"{CustomDiscreteParameter.__name__}."
                )
