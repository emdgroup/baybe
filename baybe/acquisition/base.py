"""Base classes for all acquisition functions."""

from __future__ import annotations

import warnings
from abc import ABC
from typing import ClassVar, Union

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
def _add_deprecation_hook(hook):
    """Add deprecation warnings to the default hook.

    Used for backward compatibility only and will be removed in future versions.
    """

    def added_deprecation_hook(val: Union[dict, str], cls: type):
        UCB_DEPRECATIONS = {
            "VarUCB": "UpperConfidenceBound",
            "qVarUCB": "qUpperConfidenceBound",
        }
        if (entry := val if isinstance(val, str) else val["type"]) in UCB_DEPRECATIONS:
            warnings.warn(
                f"The use of `{entry}` is deprecated and will be disabled in a "
                f"future version. The get the same outcome, use the new "
                f"{UCB_DEPRECATIONS[entry]} class instead with a beta of 100.0.",
                DeprecationWarning,
            )
            val = {"type": UCB_DEPRECATIONS[entry], "beta": 100.0}

        return hook(val, cls)

    return added_deprecation_hook


converter.register_structure_hook(
    AcquisitionFunction,
    _add_deprecation_hook(get_base_structure_hook(AcquisitionFunction)),
)
converter.register_unstructure_hook(AcquisitionFunction, unstructure_base)
