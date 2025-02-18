"""Base classes for all acquisition functions."""

from __future__ import annotations

import gc
import warnings
from abc import ABC
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from attrs import define

from baybe.exceptions import (
    UnidentifiedSubclassError,
)
from baybe.objectives.base import Objective
from baybe.searchspace.core import SearchSpace
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.base import SurrogateProtocol
from baybe.utils.basic import classproperty
from baybe.utils.boolean import is_abstract

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BotorchAcquisitionFunction


@define(frozen=True)
class AcquisitionFunction(ABC, SerialMixin):
    """Abstract base class for all acquisition functions."""

    abbreviation: ClassVar[str]
    """An alternative name for type resolution."""

    @classproperty
    def supports_batching(cls) -> bool:
        """Flag indicating whether batch recommendation is supported."""
        return cls.abbreviation.startswith("q")

    @classproperty
    def supports_pending_experiments(cls) -> bool:
        """Flag indicating whether pending experiments are supported.

        This is based on the same mechanism underlying batched recommendations.
        """
        return cls.supports_batching

    @classproperty
    def supports_multi_output(cls) -> bool:
        """Flag indicating whether multiple outputs are supported."""
        return "Hypervolume" in cls.__name__  # type: ignore[attr-defined]

    @classproperty
    def _non_botorch_attrs(cls) -> tuple[str, ...]:
        """Names of attributes that are not passed to the BoTorch constructor."""
        return ()

    def to_botorch(
        self,
        surrogate: SurrogateProtocol,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
    ) -> BotorchAcquisitionFunction:
        """Create the botorch-ready representation of the function.

        The required structure of `measurements` is specified in
        :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
        """
        from baybe.acquisition._builder import BotorchAcquisitionFunctionBuilder

        return BotorchAcquisitionFunctionBuilder(
            surrogate, searchspace, objective, measurements, pending_experiments
        ).build(self)


def _get_botorch_acqf_class(
    baybe_acqf_cls: type[AcquisitionFunction], /
) -> type[BotorchAcquisitionFunction]:
    """Extract the BoTorch acquisition class for the given BayBE acquisition class."""
    import botorch

    for cls in baybe_acqf_cls.mro():
        if acqf_cls := getattr(botorch.acquisition, cls.__name__, False) or getattr(
            botorch.acquisition.multi_objective, cls.__name__, False
        ):
            if is_abstract(acqf_cls):
                continue
            return acqf_cls  # type: ignore

    raise UnidentifiedSubclassError(
        f"No BoTorch acquisition function class match found for "
        f"'{baybe_acqf_cls.__name__}'."
    )


# Register (un-)structure hooks
def _add_deprecation_hook(hook):
    """Add deprecation warnings to the default hook.

    Used for backward compatibility only and will be removed in future versions.
    """

    def added_deprecation_hook(val: dict | str, cls: type):
        # Backwards-compatibility needs to be ensured only for deserialization from
        # base class using string-based type specifiers as listed below,
        # since the concrete classes were available only after the change.
        if is_abstract(cls):
            UCB_DEPRECATIONS = {
                "VarUCB": "UpperConfidenceBound",
                "qVarUCB": "qUpperConfidenceBound",
            }
            if (
                entry := val if isinstance(val, str) else val["type"]
            ) in UCB_DEPRECATIONS:
                warnings.warn(
                    f"The use of `{entry}` is deprecated and will be disabled in a "
                    f"future version. To get the same outcome, use the new "
                    f"`{UCB_DEPRECATIONS[entry]}` class instead with a beta of 100.0.",
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

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
