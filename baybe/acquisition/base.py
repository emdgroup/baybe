"""Base classes for all acquisition functions."""

from __future__ import annotations

import gc
import warnings
from abc import ABC
from inspect import signature
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from attrs import define

from baybe.exceptions import (
    IncompatibleAcquisitionFunctionError,
    UnidentifiedSubclassError,
)
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

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BotorchAcquisitionFunction


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
        pending_experiments: pd.DataFrame | None = None,
    ):
        """Create the botorch-ready representation of the function.

        The required structure of `measurements` is specified in
        :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
        """
        import botorch.acquisition as bo_acqf
        import torch
        from botorch.acquisition.objective import LinearMCObjective

        from baybe.acquisition.acqfs import qThompsonSampling

        # Retrieve botorch acquisition function class and match attributes
        acqf_cls = _get_botorch_acqf_class(type(self))
        params_dict = match_attributes(
            self, acqf_cls.__init__, ignore=self._non_botorch_attrs
        )[0]

        # Create botorch surrogate model
        bo_surrogate = surrogate.to_botorch()

        # Get computational data representation
        train_x = to_tensor(searchspace.transform(measurements, allow_extra=True))

        # Collect remaining (context-specific) parameters
        signature_params = signature(acqf_cls).parameters
        additional_params = {}
        additional_params["model"] = bo_surrogate
        if "X_baseline" in signature_params:
            additional_params["X_baseline"] = train_x
        if "mc_points" in signature_params:
            additional_params["mc_points"] = to_tensor(
                self.get_integration_points(searchspace)  # type: ignore[attr-defined]
            )
        if pending_experiments is not None:
            if self.is_mc:
                pending_x = searchspace.transform(pending_experiments, allow_extra=True)
                additional_params["X_pending"] = to_tensor(pending_x)
            else:
                raise IncompatibleAcquisitionFunctionError(
                    f"Pending experiments were provided but the chosen acquisition "
                    f"function '{self.__class__.__name__}' does not support this."
                )

        # Add acquisition objective / best observed value
        match objective:
            case SingleTargetObjective(NumericalTarget(mode=TargetMode.MIN)):
                if "best_f" in signature_params:
                    additional_params["best_f"] = (
                        bo_surrogate.posterior(train_x).mean.min().item()
                    )

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
                    additional_params["best_f"] = (
                        bo_surrogate.posterior(train_x).mean.max().item()
                    )
            case _:
                raise ValueError(f"Unsupported objective type: {objective}")

        params_dict.update(additional_params)

        acqf = acqf_cls(**params_dict)

        if isinstance(self, qThompsonSampling):
            assert hasattr(acqf, "_default_sample_shape")
            acqf._default_sample_shape = torch.Size([self.n_mc_samples])

        return acqf


def _get_botorch_acqf_class(
    baybe_acqf_cls: type[AcquisitionFunction], /
) -> type[BotorchAcquisitionFunction]:
    """Extract the BoTorch acquisition class for the given BayBE acquisition class."""
    import botorch

    for cls in baybe_acqf_cls.mro():
        if acqf_cls := getattr(botorch.acquisition, cls.__name__, False):
            if is_abstract(acqf_cls):
                continue
            return acqf_cls  # type: ignore

    raise UnidentifiedSubclassError(
        f"No BoTorch acquisition function class match found for "
        f"'{baybe_acqf_cls.__name__}'."
    )


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

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
