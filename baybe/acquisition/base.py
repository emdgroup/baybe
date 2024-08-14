"""Base classes for all acquisition functions."""

from __future__ import annotations

import warnings
from abc import ABC
from inspect import signature
from typing import ClassVar

import pandas as pd
from attrs import define

from baybe.objectives.base import Objective
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.searchspace.core import SearchSpace
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.base import SurrogateProtocol
from baybe.targets.enum import TargetMode
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import classproperty, match_attributes
from baybe.utils.boolean import is_abstract
from baybe.utils.dataframe import to_tensor


@define(frozen=True)
class AcquisitionFunction(ABC, SerialMixin):
    """Abstract base class for all acquisition functions."""

    abbreviation: ClassVar[str]
    """An alternative name for type resolution."""

    @classproperty
    def is_mc(cls) -> bool:
        """Flag indicating whether this is a Monte-Carlo acquisition function."""
        return cls.abbreviation.startswith("q")

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
    ):
        """Create the botorch-ready representation of the function.

        The required structure of `measurements` is specified in
        :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
        """
        import botorch.acquisition as bo_acqf
        import torch
        from botorch.acquisition.objective import LinearMCObjective

        # Get computational data representations
        train_x = searchspace.transform(measurements, allow_extra=True)
        train_y = objective.transform(measurements)

        # Retrieve corresponding botorch class
        acqf_cls = _get_botorch_acqf_class(self)

        # Match relevant attributes
        params_dict = match_attributes(
            self, acqf_cls.__init__, ignore=self._non_botorch_attrs
        )[0]

        # Collect remaining (context-specific) parameters
        signature_params = signature(acqf_cls).parameters
        additional_params = {}
        if "model" in signature_params:
            additional_params["model"] = surrogate.to_botorch()
        if "X_baseline" in signature_params:
            additional_params["X_baseline"] = to_tensor(train_x)
        if "mc_points" in signature_params:
            additional_params["mc_points"] = to_tensor(
                self.get_integration_points(searchspace)  # type: ignore[attr-defined]
            )

        # Add acquisition objective / best observed value
        match objective:
            case SingleTargetObjective(NumericalTarget(mode=TargetMode.MIN)):
                if "best_f" in signature_params:
                    additional_params["best_f"] = train_y.min().item()

                if issubclass(acqf_cls, bo_acqf.AnalyticAcquisitionFunction):
                    additional_params["maximize"] = False
                elif issubclass(acqf_cls, bo_acqf.MCAcquisitionFunction):
                    additional_params["objective"] = LinearMCObjective(
                        torch.tensor([-1.0])
                    )
                else:
                    raise ValueError(
                        f"Unsupported acquisition function type: {acqf_cls}."
                    )
            case SingleTargetObjective() | DesirabilityObjective():
                if "best_f" in signature_params:
                    additional_params["best_f"] = train_y.max().item()
            case _:
                raise ValueError(f"Unsupported objective type: {objective}")

        params_dict.update(additional_params)

        acqf = acqf_cls(**params_dict)
        if hasattr(self, "_default_sample_shape"):
            acqf._default_sample_shape = self._default_sample_shape

        return acqf


def _get_botorch_acqf_class(baybe_acqf: AcquisitionFunction):
    import botorch.acquisition as botorch_acqf_module

    for cls in baybe_acqf.__class__.mro():
        if acqf_cls := getattr(botorch_acqf_module, cls.__name__, False):
            return acqf_cls


# Register de-/serialization hooks
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
