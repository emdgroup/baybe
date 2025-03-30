"""Functionality for building BoTorch acquisition functions."""

import warnings
from collections.abc import Callable, Iterable
from functools import cached_property
from inspect import signature
from types import MappingProxyType
from typing import Any

import botorch.acquisition as bo_acqf
import pandas as pd
import torch
from attrs import asdict, define, field, fields
from attrs.validators import instance_of, optional
from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionObjective as BoObjective
from botorch.acquisition.multi_objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import LinearMCObjective
from botorch.models.model import Model
from gpytorch.utils.warnings import GPInputWarning
from torch import Tensor

from baybe.acquisition.acqfs import (
    qLogNoisyExpectedHypervolumeImprovement,
    qNegIntegratedPosteriorVariance,
    qThompsonSampling,
)
from baybe.acquisition.base import AcquisitionFunction, _get_botorch_acqf_class
from baybe.objectives.base import Objective
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.targets.enum import TargetMode
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import is_all_instance, match_attributes
from baybe.utils.dataframe import to_tensor
from baybe.utils.device_utils import (
    clear_gpu_memory,
    device_mode,
    get_default_device,
    to_device,
)


def opt_v(x: Any, /) -> Callable:
    """Shorthand for an optional attrs isinstance validator."""  # noqa: D401
    return optional(instance_of(x))


@define(kw_only=True)
class BotorchAcquisitionArgs:
    """The collection of (possible) arguments for BoTorch acquisition functions."""

    # Always required
    model: Model = field(validator=instance_of(Model))

    # Optional, depending on the specific acquisition function being used
    best_f: float | None = field(default=None, validator=opt_v(float))
    beta: float | None = field(default=None, validator=opt_v(float))
    maximize: bool | None = field(default=None, validator=opt_v(bool))
    mc_points: Tensor | None = field(default=None, validator=opt_v(Tensor))
    num_fantasies: int | None = field(default=None, validator=opt_v(int))
    objective: BoObjective | None = field(default=None, validator=opt_v(BoObjective))
    prune_baseline: bool | None = field(default=None, validator=opt_v(bool))
    ref_point: Tensor | None = field(default=None, validator=opt_v(Tensor))
    X_baseline: Tensor | None = field(default=None, validator=opt_v(Tensor))
    X_pending: Tensor | None = field(default=None, validator=opt_v(Tensor))

    def collect(self) -> dict[str, Any]:
        """Collect the assigned arguments into a dictionary."""
        return asdict(self, filter=lambda _, x: x is not None)


flds = fields(BotorchAcquisitionArgs)
"""Shorthand for the argument field references."""


@define
class BotorchAcquisitionFunctionBuilder:
    """A class for building BoTorch acquisition functions from BayBE objects."""

    # The BayBE acquisition function to be translated
    acqf: AcquisitionFunction = field(validator=instance_of(AcquisitionFunction))

    # The (pre-validated) BayBE objects
    surrogate: SurrogateProtocol = field()
    searchspace: SearchSpace = field()
    objective: Objective = field()
    measurements: pd.DataFrame = field()
    pending_experiments: pd.DataFrame | None = field(default=None)
    device: torch.device | None = field(default=None)

    # Context shared across building methods
    _args: BotorchAcquisitionArgs = field(init=False)
    _botorch_acqf_cls: BoAcquisitionFunction = field(init=False)
    _signature: MappingProxyType = field(init=False)
    _set_best_f_called: bool = field(init=False, default=False)

    def __attrs_post_init__(self) -> None:
        """Initialize the building process."""
        # Use provided device or get default
        if self.device is None:
            self.device = getattr(self.surrogate, "device", None)
            if self.device is None and torch.cuda.is_available():
                self.device = get_default_device()

        # Retrieve botorch acquisition function class and match attributes
        self._botorch_acqf_cls = _get_botorch_acqf_class(type(self.acqf))
        self._signature = signature(self._botorch_acqf_cls).parameters
        args, _ = match_attributes(
            self.acqf,
            self._botorch_acqf_cls.__init__,
            ignore=self.acqf._non_botorch_attrs,
        )

        # Use device_mode to ensure consistent device usage during to_botorch
        with warnings.catch_warnings(), device_mode(True):
            warnings.filterwarnings("ignore", category=GPInputWarning)
            bo_surrogate = self.surrogate.to_botorch()

            # Move surrogate to device using to_device utility
            bo_surrogate = to_device(bo_surrogate, self.device)

        self._args = BotorchAcquisitionArgs(model=bo_surrogate, **args)

    @cached_property
    def _botorch_surrogate(self) -> Model:
        """The botorch surrogate object."""
        with device_mode(True):
            model = self.surrogate.to_botorch()
            model = to_device(model, self.device)
            return model

    @property
    def _maximize_flags(self) -> list[bool]:
        """Booleans indicating which target is to be minimized/maximized."""
        assert is_all_instance(self.objective.targets, NumericalTarget)
        return [t.mode is TargetMode.MAX for t in self.objective.targets]

    @property
    def _multiplier(self) -> list[float]:
        """Signs indicating which target is to be minimized/maximized."""
        return [1.0 if m else -1.0 for m in self._maximize_flags]

    @cached_property
    def _train_x(self) -> pd.DataFrame:
        """The training parameter values."""
        return self.searchspace.transform(self.measurements, allow_extra=True)

    @cached_property
    def _train_y(self) -> pd.DataFrame:
        """The training target values."""
        return self.measurements[[t.name for t in self.objective.targets]]

    def build(self) -> BoAcquisitionFunction:
        """Build the BoTorch acquisition function object."""
        # Use device_mode to ensure consistent device usage
        with device_mode(True):
            # Set context-specific parameters
            self._set_best_f()
            self._invert_optimization_direction()
            self._set_X_baseline()
            self._set_X_pending()
            self._set_mc_points()
            self._set_ref_point()

            botorch_acqf = self._botorch_acqf_cls(**self._args.collect())
            self.set_default_sample_shape(botorch_acqf)

            # Use to_device for moving to the right device
            botorch_acqf = to_device(botorch_acqf, self.device)

            # Move all tensor attributes to the correct device
            for name, param in botorch_acqf.__dict__.items():
                if isinstance(param, torch.Tensor):
                    botorch_acqf.__dict__[name] = to_device(param, self.device)

            # Clean up memory
            clear_gpu_memory()

            return botorch_acqf

    def _invert_optimization_direction(self) -> None:
        """Invert optimization direction for minimization targets."""
        # ``best_f`` must have been already set (for the inversion below to work)
        assert self._set_best_f_called

        if issubclass(
            type(self.acqf),
            (
                bo_acqf.qNegIntegratedPosteriorVariance,
                bo_acqf.PosteriorStandardDeviation,
                bo_acqf.qPosteriorStandardDeviation,
            ),
        ):
            # No action needed for the active learning acquisition functions:
            # - PSTD: Minimization happens by setting `maximize=False`, which is
            #   already take care of by auto-matching attributes
            # - qPSTD and qNIPV do not support minimization yet
            # In both cases, the setting is independent of the target mode.
            return

        match self.objective:
            case SingleTargetObjective(NumericalTarget(mode=TargetMode.MIN)):
                if issubclass(self._botorch_acqf_cls, bo_acqf.MCAcquisitionFunction):
                    if self._args.best_f is not None:
                        self._args.best_f *= -1.0
                    self._args.objective = LinearMCObjective(torch.tensor([-1.0]))
                elif issubclass(
                    self._botorch_acqf_cls, bo_acqf.AnalyticAcquisitionFunction
                ):
                    self._args.maximize = False

            case ParetoObjective():
                self._args.objective = WeightedMCMultiOutputObjective(
                    torch.tensor(self._multiplier)
                )

    def _set_best_f(self) -> None:
        """Set BoTorch's ``best_f`` argument."""
        self._set_best_f_called = True

        if flds.best_f.name not in self._signature:
            return

        # Get tensor of train_x and move to device
        train_x_tensor = to_tensor(self._train_x, device=self.device)
        post_mean = self._botorch_surrogate.posterior(train_x_tensor).mean

        match self.objective:
            case SingleTargetObjective(NumericalTarget(mode=TargetMode.MIN)):
                # Move to CPU before converting to item for numerical stability
                self._args.best_f = post_mean.cpu().min().item()
            case SingleTargetObjective() | DesirabilityObjective():
                # Move to CPU before converting to item for numerical stability
                self._args.best_f = post_mean.cpu().max().item()

    def set_default_sample_shape(self, acqf: BoAcquisitionFunction, /):
        """Apply temporary workaround for Thompson sampling."""
        if not isinstance(self.acqf, qThompsonSampling):
            return

        assert hasattr(acqf, "_default_sample_shape")
        acqf._default_sample_shape = torch.Size([self.acqf.n_mc_samples])

    def _set_mc_points(self) -> None:
        """Set BoTorch's ``mc_points`` argument."""
        if flds.mc_points.name not in self._signature:
            return

        assert isinstance(self.acqf, qNegIntegratedPosteriorVariance)
        # Get integration points and move to device
        self._args.mc_points = to_tensor(
            self.acqf.get_integration_points(self.searchspace), device=self.device
        )

    def _set_ref_point(self) -> None:
        """Set BoTorch's ``ref_point`` argument."""
        if flds.ref_point.name not in self._signature:
            return

        assert isinstance(self.acqf, qLogNoisyExpectedHypervolumeImprovement)

        if isinstance(ref_point := self.acqf.reference_point, Iterable):
            tensor = torch.tensor(
                [p * m for p, m in zip(ref_point, self._multiplier, strict=True)]
            )
            # Move tensor to device if needed
            if self.device is not None:
                tensor = tensor.to(self.device)
            self._args.ref_point = tensor
        else:
            kwargs = {} if ref_point is None else {"factor": ref_point}
            tensor = torch.tensor(
                self.acqf.compute_ref_point(
                    self._train_y.to_numpy(), self._maximize_flags, **kwargs
                )
                * self._multiplier
            )
            # Move tensor to device if needed
            if self.device is not None:
                tensor = tensor.to(self.device)
            self._args.ref_point = tensor

    def _set_X_baseline(self) -> None:
        """Set BoTorch's ``X_baseline`` argument."""
        if flds.X_baseline.name not in self._signature:
            return

        # Get tensor of train_x and move to device
        self._args.X_baseline = to_tensor(self._train_x, device=self.device)

    def _set_X_pending(self) -> None:
        """Set BoTorch's ``X_pending`` argument."""
        if self.pending_experiments is None:
            return

        pending_x = self.searchspace.transform(
            self.pending_experiments, allow_extra=True
        )
        # Get tensor of pending_x and move to device
        self._args.X_pending = to_tensor(pending_x, device=self.device)
