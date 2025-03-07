"""Functionality for building BoTorch acquisition functions."""

from collections.abc import Callable, Iterable
from functools import cached_property
from inspect import signature
from types import MappingProxyType
from typing import Any

import botorch.acquisition as bo_acqf
import numpy as np
import pandas as pd
import torch
from attrs import asdict, define, field, fields
from attrs.validators import instance_of, optional
from botorch.acquisition import AcquisitionFunction as BotorchAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionObjective as BObjective
from botorch.acquisition.multi_objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import LinearMCObjective
from botorch.models.model import Model
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
    objective: BObjective | None = field(default=None, validator=opt_v(BObjective))
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

    # Context shared across building methods
    _args: BotorchAcquisitionArgs = field(init=False)
    _botorch_acqf_cls: BotorchAcquisitionFunction = field(init=False)
    _signature: MappingProxyType = field(init=False)
    _set_best_f_called: bool = field(init=False, default=False)

    def __attrs_post_init__(self) -> None:
        """Initialize the building process."""
        # Retrieve botorch acquisition function class and match attributes
        self._botorch_acqf_cls = _get_botorch_acqf_class(type(self.acqf))
        self._signature = signature(self._botorch_acqf_cls).parameters
        args, _ = match_attributes(
            self.acqf,
            self._botorch_acqf_cls.__init__,
            ignore=self.acqf._non_botorch_attrs,
        )

        # Pre-populate the acqf arguments with the content of the BayBE acqf
        self._args = BotorchAcquisitionArgs(model=self.surrogate.to_botorch(), **args)

    @cached_property
    def _train_x(self) -> Tensor:
        return to_tensor(
            self.searchspace.transform(self.measurements, allow_extra=True)
        )

    @cached_property
    def _train_y(self) -> np.ndarray:
        return self.measurements[[t.name for t in self.objective.targets]].to_numpy()

    @cached_property
    def _botorch_surrogate(self) -> Model:
        return self.surrogate.to_botorch()

    @property
    def _maximize_flags(self) -> list[bool]:
        assert is_all_instance(self.objective.targets, NumericalTarget)
        return [t.mode is TargetMode.MAX for t in self.objective.targets]

    @property
    def _multiplier(self) -> list[float]:
        return [1.0 if m else -1.0 for m in self._maximize_flags]

    def build(self) -> BotorchAcquisitionFunction:
        """Build the BoTorch acquisition function object."""
        # Set context-specific parameters
        self._set_best_f()
        self._invert_optimization_direction()
        self._set_X_baseline()
        self._set_X_pending()
        self._set_mc_points()
        self._set_ref_point()

        botorch_acqf = self._botorch_acqf_cls(**self._args.collect())
        self.set_default_sample_shape(botorch_acqf)

        return botorch_acqf

    def _set_X_baseline(self) -> None:
        """Set BoTorch's ``X_baseline`` argument."""
        if flds.X_baseline.name not in self._signature:
            return

        self._args.X_baseline = self._train_x

    def _set_mc_points(self) -> None:
        """Set BoTorch's ``mc_points`` argument."""
        if flds.mc_points.name not in self._signature:
            return

        assert isinstance(self.acqf, qNegIntegratedPosteriorVariance)
        self._args.mc_points = to_tensor(
            self.acqf.get_integration_points(self.searchspace)
        )

    def _set_X_pending(self) -> None:
        """Set BoTorch's ``X_pending`` argument."""
        if self.pending_experiments is None:
            return

        pending_x = self.searchspace.transform(
            self.pending_experiments, allow_extra=True
        )
        self._args.X_pending = to_tensor(pending_x)

    def _set_best_f(self) -> None:
        """Set BoTorch's ``best_f`` argument."""
        self._set_best_f_called = True

        if flds.best_f.name not in self._signature:
            return

        posterior_mean = self._botorch_surrogate.posterior(self._train_x).mean

        match self.objective:
            case SingleTargetObjective(NumericalTarget(mode=TargetMode.MIN)):
                self._args.best_f = posterior_mean.min().item()
            case SingleTargetObjective() | DesirabilityObjective():
                self._args.best_f = posterior_mean.max().item()

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

    def _set_ref_point(self) -> None:
        """Set BoTorch's ``ref_point`` argument."""
        if flds.ref_point.name not in self._signature:
            return

        assert isinstance(self.acqf, qLogNoisyExpectedHypervolumeImprovement)

        if isinstance(ref_point := self.acqf.reference_point, Iterable):
            self._args.ref_point = torch.tensor(
                [p * m for p, m in zip(ref_point, self._multiplier, strict=True)]
            )
        else:
            kwargs = {} if ref_point is None else {"factor": ref_point}
            self._args.ref_point = torch.tensor(
                self.acqf.compute_ref_point(
                    self._train_y, self._maximize_flags, **kwargs
                )
                * self._multiplier
            )

    def set_default_sample_shape(self, acqf: BotorchAcquisitionFunction, /):
        """Apply temporary workaround for Thompson sampling."""
        if not isinstance(self.acqf, qThompsonSampling):
            return

        assert hasattr(acqf, "_default_sample_shape")
        acqf._default_sample_shape = torch.Size([self.acqf.n_mc_samples])
