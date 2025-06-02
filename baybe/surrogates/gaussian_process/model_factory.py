"""Model factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Protocol

from attrs import define, field
from attrs.validators import instance_of
from botorch.models.model import Model
from typing_extensions import override

from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin

if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from gpytorch.likelihoods import Likelihood
    from gpytorch.means import Mean
    from torch import Tensor

    from baybe.searchspace.core import SearchSpace
    from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory


class ModelFactory(Protocol):
    """A protocol defining the interface expected for model factories."""

    def __call__(
        self,
        searchspace: SearchSpace,
        train_x: Tensor,
        train_y: Tensor,
        input_transform: InputTransform,
        outcome_transform: OutcomeTransform | None = None,
        kernel_factory: KernelFactory | None = None,
        mean_module: Mean | None = None,
        likelihood: Likelihood | None = None,
    ) -> Model:
        """Create a :class:`botorch.models.model.Model` for the given DOE context."""
        ...


# Register (un-)structure hooks
converter.register_structure_hook(ModelFactory, get_base_structure_hook(ModelFactory))
converter.register_unstructure_hook(ModelFactory, unstructure_base)


@define(frozen=True)
class PlainModelFactory(ModelFactory, SerialMixin):
    """A trivial factory that returns a fixed pre-defined model upon request."""

    model: Model = field(validator=instance_of(Model))
    """The fixed model to be returned by the factory."""

    @override
    def __call__(
        self,
        searchspace: SearchSpace,
        train_x: Tensor,
        train_y: Tensor,
        input_transform: InputTransform,
        outcome_transform: OutcomeTransform | None = None,
        kernel_factory: KernelFactory | None = None,
        mean_module: Mean | None = None,
        likelihood: Likelihood | None = None,
    ) -> Model:
        return self.model


def to_model_factory(x: Model | ModelFactory | None, /) -> ModelFactory | None:
    """Wrap a model into a plain model factory.

    With factory or None passthrough.
    """
    return PlainModelFactory(x) if isinstance(x, Model) else x


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
