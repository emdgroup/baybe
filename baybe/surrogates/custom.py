"""Functionality for building custom surrogates.

Note that ONNX surrogate models cannot be retrained. However, having the surrogates
raise a ``NotImplementedError`` would currently break the code since
:class:`baybe.recommenders.pure.bayesian.base.BayesianRecommender` assumes that
surrogates can be trained and attempts to do so for each new DOE iteration.

It is planned to solve this issue in the future.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

import cattrs
from attrs import define, field, validators
from typing_extensions import override

from baybe.exceptions import DeprecationError
from baybe.parameters import (
    CategoricalEncoding,
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.serialization import converter
from baybe.surrogates.base import IndependentGaussianSurrogate
from baybe.surrogates.utils import batchify_mean_var_prediction
from baybe.utils.conversion import to_string
from baybe.utils.numerical import DTypeFloatONNX

if TYPE_CHECKING:
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


def register_custom_architecture(*args, **kwargs) -> NoReturn:
    """Deprecated! Raises an error when used."""  # noqa: D401
    raise DeprecationError(
        "The 'register_custom_architecture' decorator is no longer available. "
        "Use :class:`baybe.surrogates.base.SurrogateProtocol` instead to define "
        "your custom architectures."
    )


@define(kw_only=True)
class CustomONNXSurrogate(IndependentGaussianSurrogate):
    """A wrapper class for custom pretrained surrogate models.

    Note that these surrogates cannot be retrained.
    """

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    onnx_input_name: str = field(validator=validators.instance_of(str))
    """The input name used for constructing the ONNX str."""

    onnx_str: bytes = field(validator=validators.instance_of(bytes))
    """The ONNX byte str representing the model."""

    # TODO: type should be `onnxruntime.InferenceSession` but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @override
    @batchify_mean_var_prediction
    def _estimate_moments(
        self, candidates_comp_scaled: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        import torch

        from baybe._optional.onnx import onnxruntime as ort
        from baybe.utils.torch import DTypeFloatTorch

        model_inputs = {
            self.onnx_input_name: candidates_comp_scaled.numpy().astype(DTypeFloatONNX)
        }
        if self._model is None:
            self._model = ort.InferenceSession(self.onnx_str)
        results = self._model.run(None, model_inputs)

        # IMPROVE: At the moment, we assume that the second model output contains
        #   standard deviations. Currently, most available ONNX converters care
        #   about the mean only and it's not clear how this will be handled in the
        #   future. Once there are more choices available, this should be revisited.
        return (
            torch.from_numpy(results[0]).to(DTypeFloatTorch),
            torch.from_numpy(results[1]).pow(2).to(DTypeFloatTorch),
        )

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
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

    @override
    def __str__(self) -> str:
        fields = [to_string("ONNX input name", self.onnx_input_name, single_line=True)]
        return to_string(super().__str__(), *fields)


# Register (un-)structure hooks
@converter.register_unstructure_hook
def _decode_onnx_string(obj: CustomONNXSurrogate) -> dict[str, Any]:
    """Unstructure ONNX surrogates by decoding the ONNX byte string."""
    fn = cattrs.gen.make_dict_unstructure_fn(
        CustomONNXSurrogate,
        converter,
        onnx_str=cattrs.override(unstruct_hook=lambda x: x.decode(_ONNX_ENCODING)),
    )
    return fn(obj)


@converter.register_structure_hook
def _encode_onnx_string(dct: dict[str, Any], _) -> CustomONNXSurrogate:
    """Structure ONNX surrogates by encoding the ONNX byte string."""
    fn = cattrs.gen.make_dict_structure_fn(
        CustomONNXSurrogate,
        converter,
        onnx_str=cattrs.override(
            struct_hook=lambda x, _: x.encode(_ONNX_ENCODING)
            if isinstance(x, str)
            else x
        ),
    )
    return fn(dct)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
