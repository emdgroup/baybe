"""Base classes for all acquisition functions."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from attrs import define

from baybe.acquisition.adapter import debotorchize
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.base import Surrogate


@define(frozen=True)
class AcquisitionFunction(ABC, SerialMixin):
    """Abstract base class for all acquisition functions."""

    _abbreviation: ClassVar[str]
    """An alternative name for type resolution."""

    def to_botorch(self, surrogate: Surrogate, best_f: float):
        """Create the botorch-ready representation of the function."""
        import botorch.acquisition as botorch_acqf

        acqf_cls = getattr(botorch_acqf, self.__class__.__name__)

        return debotorchize(acqf_cls)(surrogate, best_f)


# Register de-/serialization hooks
converter.register_structure_hook(
    AcquisitionFunction, get_base_structure_hook(AcquisitionFunction)
)
converter.register_unstructure_hook(AcquisitionFunction, unstructure_base)
