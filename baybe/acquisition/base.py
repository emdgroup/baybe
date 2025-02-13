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
        """Create the botorch-ready representation of the function."""
        import botorch.acquisition as bo_acqf
        import torch
        from botorch.acquisition.objective import LinearMCObjective
        from gpytorch.utils.warnings import GPInputWarning
        from baybe.acquisition.acqfs import qThompsonSampling
        import warnings
        from gpytorch.settings import debug, fast_computations
        from baybe.utils.device_mode import single_device_mode

        # Get device from surrogate and ensure it's set
        device = getattr(surrogate, "device", None)
        if device is None and torch.cuda.is_available():
            device = torch.device("cuda:0")
        
        # Create botorch surrogate model
        with warnings.catch_warnings(), single_device_mode(True), debug(True), fast_computations(solves=False):
            warnings.filterwarnings('ignore', category=GPInputWarning)
            bo_surrogate = surrogate.to_botorch()
            
            if device is not None:
                bo_surrogate = bo_surrogate.to(device)
                
                # Move all model components to device
                if hasattr(bo_surrogate, "covar_module"):
                    bo_surrogate.covar_module = bo_surrogate.covar_module.to(device)
                    if hasattr(bo_surrogate.covar_module, "kernels"):
                        bo_surrogate.covar_module.kernels = tuple(
                            k.to(device) for k in bo_surrogate.covar_module.kernels
                        )
                        # Move base kernels
                        for k in bo_surrogate.covar_module.kernels:
                            if hasattr(k, "base_kernel"):
                                k.base_kernel = k.base_kernel.to(device)
                
                # Move training data
                if hasattr(bo_surrogate, "train_inputs"):
                    bo_surrogate.train_inputs = tuple(
                        x.to(device) for x in bo_surrogate.train_inputs
                    )
                if hasattr(bo_surrogate, "train_targets"):
                    bo_surrogate.train_targets = bo_surrogate.train_targets.to(device)
                
                # Move likelihood
                if hasattr(bo_surrogate, "likelihood") and bo_surrogate.likelihood is not None:
                    bo_surrogate.likelihood = bo_surrogate.likelihood.to(device)

            # Get computational data representation (ensure tensor is on surrogate.device)
            train_x = to_tensor(
                searchspace.transform(measurements, allow_extra=True),
                device=device,
            )

            # Force a dummy forward pass to re-compute the prediction strategy caches
            with torch.no_grad():
                try:
                    _ = bo_surrogate(train_x)
                except Exception:
                    pass

            # Move model to device again after forward pass
            if device is not None:
                bo_surrogate = bo_surrogate.to(device)

            # Get botorch acquisition function class and match attributes
            acqf_cls = _get_botorch_acqf_class(type(self))
            params_dict = match_attributes(
                self, acqf_cls.__init__, ignore=self._non_botorch_attrs
            )[0]

            # Collect remaining (context-specific) parameters
            signature_params = signature(acqf_cls).parameters
            additional_params = {}
            additional_params["model"] = bo_surrogate
            if "X_baseline" in signature_params:
                additional_params["X_baseline"] = train_x

            # Initialize X_pending with correct device and dtype
            if pending_experiments is not None:
                if self.supports_pending_experiments:
                    pending_x = searchspace.transform(pending_experiments, allow_extra=True)
                    pending_tensor = to_tensor(pending_x, device=device)
                    # Ensure pending tensor matches train_x dtype
                    pending_tensor = pending_tensor.to(dtype=train_x.dtype)
                    additional_params["X_pending"] = pending_tensor
                else:
                    raise IncompatibleAcquisitionFunctionError(
                        f"Pending experiments were provided but the chosen acquisition "
                        f"function '{self.__class__.__name__}' does not support this."
                    )
            else:
                # Initialize empty X_pending tensor with matching device and dtype
                additional_params["X_pending"] = torch.empty(
                    (0, train_x.shape[1]), 
                    device=device, 
                    dtype=train_x.dtype
                )

            if "mc_points" in signature_params:
                mc_points = to_tensor(
                    self.get_integration_points(searchspace),  # type: ignore[attr-defined]
                    device=device,
                )
                # Ensure mc_points matches dtype
                mc_points = mc_points.to(dtype=train_x.dtype)
                additional_params["mc_points"] = mc_points

            # Add acquisition objective / best observed value
            match objective:
                case SingleTargetObjective(NumericalTarget(mode=TargetMode.MIN)):
                    # Adjust best_f
                    if "best_f" in signature_params:
                        # Move to CPU before converting to item
                        additional_params["best_f"] = (
                            bo_surrogate.posterior(train_x).mean.cpu().min().item()
                        )
                        if issubclass(acqf_cls, bo_acqf.MCAcquisitionFunction):
                            additional_params["best_f"] *= -1.0

                    # Adjust objective
                    if issubclass(
                        acqf_cls,
                        (
                            bo_acqf.qNegIntegratedPosteriorVariance,
                            bo_acqf.PosteriorStandardDeviation,
                            bo_acqf.qPosteriorStandardDeviation,
                        ),
                    ):
                        pass
                    elif issubclass(acqf_cls, bo_acqf.AnalyticAcquisitionFunction):
                        additional_params["maximize"] = False
                    elif issubclass(acqf_cls, bo_acqf.MCAcquisitionFunction):
                        additional_params["objective"] = LinearMCObjective(
                            torch.tensor([-1.0], device=device)
                        )
                    else:
                        raise ValueError(
                            f"Unsupported acquisition function type: {acqf_cls}."
                        )
                case SingleTargetObjective() | DesirabilityObjective():
                    if "best_f" in signature_params:
                        # Move to CPU before converting to item
                        additional_params["best_f"] = (
                            bo_surrogate.posterior(train_x).mean.cpu().max().item()
                        )
                case _:
                    raise ValueError(f"Unsupported objective type: {objective}")

            params_dict.update(additional_params)

            # Create acquisition function and ensure it's on the correct device
            acqf = acqf_cls(**params_dict)
            if device is not None:
                acqf = acqf.to(device)

            if isinstance(self, qThompsonSampling):
                assert hasattr(acqf, "_default_sample_shape")
                acqf._default_sample_shape = torch.Size([self.n_mc_samples])

            # Final device and dtype check for all tensors
            for name, param in acqf.__dict__.items():
                if isinstance(param, torch.Tensor):
                    acqf.__dict__[name] = param.to(device=device, dtype=train_x.dtype)

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
