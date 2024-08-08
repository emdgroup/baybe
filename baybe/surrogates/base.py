"""Base functionality for all BayBE surrogates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

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
from baybe.utils.scaling import ColumnTransformer

if TYPE_CHECKING:
    from botorch.models.model import Model
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import GPyTorchPosterior, Posterior
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

_ModelContext = TypeVar("_ModelContext", Any, None)
"""Context information that is provided to the model fitting method."""


class _NoTransform(Enum):
    """Sentinel class."""

    IDENTITY_TRANSFORM = auto()


_IDENTITY_TRANSFORM = _NoTransform.IDENTITY_TRANSFORM
"""Sentinel to indicate the absence of a transform where `None` is ambiguous."""


class SurrogateProtocol(Protocol):
    """Type protocol specifying the interface surrogate models need to implement."""

    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Fit the surrogate to training data in a given modelling context.

        For details on the expected method arguments, see
        :meth:`baybe.recommenders.base.RecommenderProtocol`.
        """

    def to_botorch(self) -> Model:
        """Create the botorch-ready representation of the fitted model.

        The :class:`botorch.models.model.Model` created by this method needs to be
        configured such that it can be called with candidate points in **computational
        representation**, that is, input of the form as obtained via
        :meth:`baybe.searchspace.core.SearchSpace.transform`.
        """


@define
class Surrogate(ABC, SurrogateProtocol, SerialMixin, Generic[_ModelContext]):
    """Abstract base class for all surrogate models."""

    # Class variables
    joint_posterior: ClassVar[bool]
    """Class variable encoding whether or not a joint posterior is calculated."""

    supports_transfer_learning: ClassVar[bool]
    """Class variable encoding whether or not the surrogate supports transfer
    learning."""

    _searchspace: SearchSpace | None = field(init=False, default=None, eq=False)
    """The search space on which the surrogate operates. Available after fitting."""

    # TODO: type should be
    #   `botorch.models.transforms.outcome.Standardize | _NoTransform`
    #   but is currently omitted due to:
    #   https://github.com/python-attrs/cattrs/issues/531
    _output_scaler = field(init=False, default=None, eq=False)
    """Scaler for transforming output values. Available after fitting.

    Scales a tensor containing target measurements in computational representation
    to make them digestible for the model-specific, scale-agnostic posterior logic."""

    def to_botorch(self) -> Model:  # noqa: D102
        # See base class.
        from baybe.surrogates._adapter import AdapterModel

        return AdapterModel(self)

    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        """Return the scaler factory to be used for the given parameter.

        This method is supposed to be overridden by subclasses to implement their
        custom parameter scaling logic.
        """
        from botorch.models.transforms.input import Normalize

        return Normalize

    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        """Return the scaler factory to be used for target scaling.

        This method is supposed to be overridden by subclasses to implement their
        custom target scaling logic.
        """
        from botorch.models.transforms.outcome import Standardize

        return Standardize

    def _make_input_scaler(self, searchspace: SearchSpace) -> ColumnTransformer:
        """Make the input scaler for transforming computational dataframes."""
        # Create a composite scaler from parameter-wise scaler objects
        mapping: dict[tuple[int, ...], InputTransform] = {}
        for p in searchspace.parameters:
            if (factory := self._make_parameter_scaler_factory(p)) is None:
                continue
            idxs = searchspace.get_comp_rep_parameter_indices(p.name)
            transformer = factory(len(idxs))
            mapping[idxs] = transformer
        scaler = ColumnTransformer(mapping)

        # Fit the scaler to the parameter bounds
        scaler.fit(to_tensor(searchspace.comp_rep_bounds))

        return scaler

    def _make_output_scaler(
        self, objective: Objective, measurements: pd.DataFrame
    ) -> OutcomeTransform | _NoTransform:
        """Make the output scaler for transforming computational dataframes."""
        if (factory := self._make_target_scaler_factory()) is None:
            return _IDENTITY_TRANSFORM

        # TODO: Multi-target extension
        scaler = factory(1)

        # TODO: Consider taking into account target boundaries when available
        scaler(to_tensor(objective.transform(measurements)))
        scaler.eval()

        return scaler

    def posterior(self, candidates: pd.DataFrame, /) -> Posterior:
        """Compute the posterior for candidates in experimental representation.

        Takes a dataframe of parameter configurations in **experimental representation**
        and returns the corresponding posterior object. Therefore, the method serves as
        the user-facing entry point for accessing model predictions.

        Args:
            candidates: A dataframe containing parameter configurations in
                **experimental representation**.

        Raises:
            ModelNotTrainedError: When called before the model has been trained.

        Returns:
            A :class:`botorch.posteriors.Posterior` object representing the posterior
            distribution at the given candidate points, where the posterior is also
            described in **experimental representation**. That is, the posterior values
            lie in the same domain as the modelled targets/objective on which the
            surrogate was trained via :meth:`baybe.surrogates.base.Surrogate.fit`.
        """
        if self._searchspace is None:
            raise ModelNotTrainedError(
                "The surrogate must be trained before a posterior can be computed."
            )
        return self._posterior_comp_rep(
            to_tensor(self._searchspace.transform(candidates))
        )

    def _posterior_comp_rep(self, candidates: Tensor, /) -> Posterior:
        """Compute the posterior for candidates in computational representation.

        Takes a tensor of parameter configurations in **computational representation**
        and returns the corresponding posterior object. Therefore, the method provides
        the entry point for queries coming from computational layers, for instance,
        BoTorch's `optimize_*` functions.

        Args:
            candidates: A tensor containing parameter configurations in **computational
                representation**.

        Returns:
            The same :class:`botorch.posteriors.Posterior` object as returned via
            :meth:`baybe.surrogates.base.Surrogate.posterior`.
        """
        p = self._posterior(self._input_scaler.transform(candidates))
        if self._output_scaler is not _IDENTITY_TRANSFORM:
            p = self._output_scaler.untransform_posterior(p)
        return p

    @abstractmethod
    def _posterior(self, candidates: Tensor, /) -> Posterior:
        """Perform the actual model-specific posterior evaluation logic.

        This method is supposed to be overridden by subclasses to implement their
        model-specific surrogate architecture. Internally, the method is called by the
        base class with a **scaled** tensor of candidates in **computational
        representation**, where the scaling is configurable by the subclass by
        overriding the default scaler factory methods of the base. The base class also
        takes care of transforming the returned posterior back to the original scale
        according to the defined scalers.

        This means:
        -----------
        Subclasses implementing this method do not have to bother about
        pre-/postprocessing of the in-/output. Instead, they only need to implement the
        mathematical operation of computing the posterior for the given input according
        to their model specifications and can implicitly assume that scaling is handled
        appropriately outside. In short: the returned posterior simply needs to be on
        the same scale as the given input.

        Args:
            candidates: A tensor containing **pre-scaled** parameter configurations
                in **computational representation**, as defined through the input scaler
                obtained via :meth:`baybe.surrogates.base.Surrogate._make_input_scaler`.

        Returns:
            A :class:`botorch.posteriors.Posterior` object representing the
            **scale-transformed** posterior distributions at the given candidate points,
            where the posterior is described on the scale dictated by the output scaler
            obtained via :meth:`baybe.surrogates.base.Surrogate._make_output_scaler`.
        """

    @staticmethod
    def _get_model_context(
        searchspace: SearchSpace, objective: Objective
    ) -> _ModelContext:
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
        if (not searchspace.continuous.is_empty) and (
            "GaussianProcess" not in self.__class__.__name__
        ):
            raise NotImplementedError(
                "Continuous search spaces are currently only supported by GPs."
            )

        # Remember on which search space the model is trained
        self._searchspace = searchspace

        # Create context-specific transformations
        self._input_scaler = self._make_input_scaler(searchspace)
        self._output_scaler = self._make_output_scaler(objective, measurements)

        # Transform and fit
        train_x_comp_rep, train_y_comp_rep = to_tensor(
            searchspace.transform(measurements, allow_extra=True),
            objective.transform(measurements),
        )
        train_x = self._input_scaler.transform(train_x_comp_rep)
        train_y = (
            train_y_comp_rep
            if self._output_scaler is _IDENTITY_TRANSFORM
            else self._output_scaler(train_y_comp_rep)[0]
        )
        self._fit(train_x, train_y, self._get_model_context(searchspace, objective))

    @abstractmethod
    def _fit(self, train_x: Tensor, train_y: Tensor, context: _ModelContext) -> None:
        """Perform the actual fitting logic."""


@define
class GaussianSurrogate(Surrogate, ABC):
    """A surrogate model providing Gaussian posterior estimates."""

    def _posterior(self, candidates: Tensor, /) -> GPyTorchPosterior:
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
    def _estimate_moments(self, candidates: Tensor, /) -> tuple[Tensor, Tensor]:
        """Estimate first and second moments of the Gaussian posterior.

        The second moment may either be a 1-D tensor of marginal variances for the
        candidates or a 2-D tensor representing a full covariance matrix over all
        candidates, depending on the ``joint_posterior`` flag of the model.
        """


def _make_hook_decode_onnx_str(
    raw_unstructure_hook: UnstructureHook,
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
    raw_unstructure_hook: UnstructureHook,
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
_unstructure_hook = _make_hook_decode_onnx_str(
    _block_serialize_custom_architecture(
        lambda x: unstructure_base(x, overrides={"_model": override(omit=True)})
    )
)
converter.register_unstructure_hook(Surrogate, _unstructure_hook)
converter.register_structure_hook(
    Surrogate, _make_hook_encode_onnx_str(get_base_structure_hook(Surrogate))
)
converter.register_unstructure_hook(SurrogateProtocol, _unstructure_hook)
converter.register_structure_hook(
    SurrogateProtocol,
    _make_hook_encode_onnx_str(get_base_structure_hook(SurrogateProtocol)),
)
