"""Functionality for building BoTorch acquisition functions."""

from collections.abc import Callable, Iterable
from functools import cached_property
from inspect import signature
from types import MappingProxyType
from typing import Any, cast

import pandas as pd
import torch
from attrs import Attribute, asdict, define, field, fields
from attrs.validators import instance_of, optional
from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionObjective
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from torch import Tensor

from baybe.acquisition.acqfs import (
    _ExpectedHypervolumeImprovement,
    qExpectedHypervolumeImprovement,
    qLogExpectedHypervolumeImprovement,
    qNegIntegratedPosteriorVariance,
    qThompsonSampling,
)
from baybe.acquisition.base import AcquisitionFunction, _get_botorch_acqf_class
from baybe.acquisition.utils import make_partitioning
from baybe.exceptions import IncompatibilityError, IncompleteMeasurementsError
from baybe.objectives.base import Objective
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.targets.binary import BinaryTarget
from baybe.targets.numerical import NumericalTarget
from baybe.transformations import AffineTransformation, IdentityTransformation
from baybe.utils.basic import match_attributes
from baybe.utils.dataframe import handle_missing_values, to_tensor

_OPT_FIELD: None = object()  # type: ignore[assignment]
"""Sentinel value indicating optional acquisition function attributes."""


def opt_v(x: Any, /) -> Callable:
    """Shorthand for an optional attrs isinstance validator."""  # noqa: D401
    return optional(instance_of(x))


def _make_optional_fields(_, fields: list[Attribute]) -> list[Attribute]:
    """Automatically set default values and validators for optional fields."""

    def set_default_and_validator(fld: Attribute) -> Attribute:
        """Set default value and validator for the given optional field."""
        if fld.default is not _OPT_FIELD:
            return fld
        return fld.evolve(default=None, validator=instance_of(fld.type))  # type: ignore[arg-type]

    return [set_default_and_validator(fld) for fld in fields]


@define(kw_only=True, field_transformer=_make_optional_fields)
class BotorchAcquisitionArgs:
    """The collection of (possible) arguments for BoTorch acquisition functions."""

    # Always required
    model: Model = field(validator=instance_of(Model))

    # Optional, depending on the specific acquisition function being used
    best_f: float | None = _OPT_FIELD
    beta: float | None = _OPT_FIELD
    maximize: bool | None = _OPT_FIELD
    mc_points: Tensor | None = _OPT_FIELD
    num_fantasies: int | None = _OPT_FIELD
    objective: MCAcquisitionObjective | None = _OPT_FIELD
    partitioning: BoxDecomposition | None = _OPT_FIELD
    posterior_transform: PosteriorTransform | None = _OPT_FIELD
    prune_baseline: bool | None = _OPT_FIELD
    ref_point: Tensor | None = _OPT_FIELD
    X_baseline: Tensor | None = _OPT_FIELD
    X_pending: Tensor | None = _OPT_FIELD

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
    _botorch_acqf_cls: BoAcquisitionFunction = field(init=False)
    _signature: MappingProxyType = field(init=False)

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
    def _botorch_surrogate(self) -> Model:
        """The botorch surrogate object."""
        return self.surrogate.to_botorch()

    @cached_property
    def _train_x(self) -> pd.DataFrame:
        """The training parameter values."""
        return self.searchspace.transform(self.measurements, allow_extra=True)

    @cached_property
    def _posterior_mean_comp(self) -> Tensor:
        """The posterior mean of the (transformed) targets of the training data."""
        # TODO: Currently, this is the "transformed posterior mean of the targets"
        #   rather than the "posterior mean of the transformed targets".
        posterior = self._botorch_surrogate.posterior(to_tensor(self._train_x))
        return self.objective.to_botorch()(posterior.mean)

    @cached_property
    def _target_configurations(self) -> pd.DataFrame:
        """The target configurations used for reference point calculation.

        Only completely measured points are considered.

        Returns:
            A dataframe of target configurations.

        Raises:
            ValueError: If no complete measurement exists.
        """
        configurations = handle_missing_values(
            self.measurements[[t.name for t in self.objective.targets]],
            [t.name for t in self.objective.targets],
            drop=True,
        )

        # TODO: A smarter treatment might be possible in the case that not at least
        #  one complete measurement exists, e.g. by considering target bounds or other
        #  heuristics.
        if configurations.empty:
            raise IncompleteMeasurementsError(
                f"For calculating a default reference point, at least one "
                f"configuration must have a measured value for all targets. You can "
                f"fix this by setting the "
                f"'{fields(_ExpectedHypervolumeImprovement).reference_point.name}' "
                f"argument of '{self.acqf.__class__.__name__}' explicitly."
            )

        return configurations

    def build(self) -> BoAcquisitionFunction:
        """Build the BoTorch acquisition function object."""
        # Set context-specific parameters
        self._set_best_f()
        self._set_target_transformation()
        self._set_X_baseline()
        self._set_X_pending()
        self._set_mc_points()
        self._set_ref_point()
        self._set_partitioning()

        botorch_acqf = self._botorch_acqf_cls(**self._args.collect())
        self.set_default_sample_shape(botorch_acqf)

        return botorch_acqf

    def _set_target_transformation(self) -> None:
        """Apply potential target transformations."""
        # NOTE: BoTorch offers two distinct pathways for implementing target
        #   transformations, with partly overlapping functionality: posterior transforms
        #   and objectives (https://github.com/pytorch/botorch/discussions/2164).
        #   We use the former for affine transformations and the latter to handle
        #   all other cases.

        if self.acqf.is_analytic:
            if not isinstance(self.objective, SingleTargetObjective):
                targets = self.objective.targets
                raise IncompatibilityError(
                    f"The selected analytic acquisition "
                    f"'{self.acqf.__class__.__name__}' can handle one target only but "
                    f"the specified objective comprises {len(targets)} targets: "
                    f"{[t.name for t in targets]}"
                )

            match target := self.objective._target:
                case NumericalTarget():
                    pass
                case BinaryTarget():
                    return
                case _:
                    raise NotImplementedError("No transformation handling implemented.")

            match t := target.transformation:
                case IdentityTransformation() | AffineTransformation():
                    # The identity/affine type narrowing is lost due to the `negate`
                    # call, but we know that the result will be an AffineTransformation
                    # in this specific context
                    oriented = cast(
                        AffineTransformation, t.negate() if target.minimize else t
                    )
                    self._args.posterior_transform = (
                        oriented.to_botorch_posterior_transform()
                    )
                case _:
                    raise NotImplementedError(
                        f"The selected analytic acquisition "
                        f"'{self.acqf.__class__.__name__}' supports only affine "
                        f"target transformations. "
                    )
        else:
            # TODO: Enable once clarified:
            # https://github.com/pytorch/botorch/discussions/2849
            if isinstance(self.acqf, qNegIntegratedPosteriorVariance):
                # Type narrowing: qNIPV implies a single target
                assert isinstance(self.objective, SingleTargetObjective)

                if isinstance(
                    (target := self.objective._target), NumericalTarget
                ) and not isinstance(target.transformation, IdentityTransformation):
                    raise IncompatibilityError(
                        f"'{qNegIntegratedPosteriorVariance.__name__}' currently "
                        f"does not support any target transformations."
                    )
                else:
                    return

            self._args.objective = self.objective.to_botorch()

    def _set_best_f(self) -> None:
        """Set BoTorch's ``best_f`` argument."""
        if flds.best_f.name not in self._signature:
            return

        match self.objective:
            case SingleTargetObjective() | DesirabilityObjective():
                self._args.best_f = self._posterior_mean_comp.max().item()
            case _:
                raise NotImplementedError("This line should be impossible to reach.")

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
        self._args.mc_points = to_tensor(
            self.acqf.get_integration_points(self.searchspace)
        )

    def _set_partitioning(self) -> None:
        """Set BoTorch's ``partitioning`` argument."""
        if flds.partitioning.name not in self._signature:
            return

        assert isinstance(
            self.acqf,
            (qExpectedHypervolumeImprovement, qLogExpectedHypervolumeImprovement),
        )
        assert self._args.ref_point is not None

        self._args.partitioning = make_partitioning(
            self._posterior_mean_comp, self._args.ref_point, self.acqf.alpha
        )

    def _set_ref_point(self) -> None:
        """Set BoTorch's ``ref_point`` argument."""
        if flds.ref_point.name not in self._signature:
            return

        assert isinstance(self.acqf, _ExpectedHypervolumeImprovement)

        if isinstance(ref_point := self.acqf.reference_point, Iterable):
            self._args.ref_point = torch.tensor(ref_point)
        else:
            kwargs = {} if ref_point is None else {"factor": ref_point}
            self._args.ref_point = to_tensor(
                self.acqf.compute_ref_point(
                    self.objective.to_botorch()(to_tensor(self._target_configurations)),
                    **kwargs,
                )
            )

    def _set_X_baseline(self) -> None:
        """Set BoTorch's ``X_baseline`` argument."""
        if flds.X_baseline.name not in self._signature:
            return

        self._args.X_baseline = to_tensor(self._train_x)

    def _set_X_pending(self) -> None:
        """Set BoTorch's ``X_pending`` argument."""
        if self.pending_experiments is None:
            return

        pending_x = self.searchspace.transform(
            self.pending_experiments, allow_extra=True
        )
        self._args.X_pending = to_tensor(pending_x)
