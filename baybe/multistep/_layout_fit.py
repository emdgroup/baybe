"""Private compatibility shim for fitting Gaussian processes on factorized layouts.

The generic surrogate API is intentionally unaware of the product-free input
layout and requires a real :class:`~baybe.searchspace.core.SearchSpace`. This
module bridges the gap for
:class:`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate`, using
the same scaling, transformation, component resolution, and cache semantics as
the regular fit path. It is not a general search-space replacement: layout fits
accept only Gaussian-process surrogates.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Collection
from typing import TypeAlias, cast

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from joblib.hashing import hash

from baybe.exceptions import (
    DeprecationError,
    IncompatibilityError,
    IncompatibleSurrogateError,
    UnusedObjectWarning,
)
from baybe.objectives.base import Objective
from baybe.parameters.categorical import TaskParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import _IDENTITY_TRANSFORM
from baybe.surrogates.gaussian_process.components.kernel import ICMKernelFactory
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.boolean import strtobool
from baybe.utils.dataframe import handle_missing_values, to_tensor

from ._configuration_space import _ConfigurationSpace, _DiscreteParameterView

_FitInputContext: TypeAlias = SearchSpace | _ConfigurationSpace
"""The fit context of a surrogate: a real search space or a product-free layout."""


@define(frozen=True, slots=False)
class _LayoutSearchSpaceView:
    """Explicit, read-only search-space-shaped view of a product-free layout.

    All regular attributes are delegated to the real layout. The only added
    attribute is a discrete-parameter view, which the built-in EDBO preset accesses
    through ``searchspace.discrete.parameters``. No search-space object, candidate
    table, or parameter product is constructed.
    """

    layout: _ConfigurationSpace = field(validator=instance_of(_ConfigurationSpace))
    """The actual product-free input layout."""

    @property
    def discrete(self) -> _DiscreteParameterView:
        """The discrete-parameter view of the underlying layout."""
        return self.layout.discrete

    def _drop_parameters(self, names: Collection[str], /) -> _LayoutSearchSpaceView:
        """Return a view over the layout with the named parameters removed.

        The generic GP resolves a residual kernel by calling ``_drop_parameters`` and
        passing the result to the configured factory. Keeping the returned reduced
        layout behind this view preserves the ``discrete`` adaptation that the EDBO
        preset needs on that path.
        """
        return type(self)(self.layout._drop_parameters(names))

    def __getattr__(self, name: str):
        """Delegate every other attribute to the underlying layout."""
        return getattr(self.layout, name)


@define(frozen=True, slots=False)
class _LayoutModelContext:
    """The GP component context for a product-free input layout.

    It mirrors :class:`baybe.surrogates.gaussian_process.core._ModelContext`, but
    resolves its attributes through a :class:`_LayoutSearchSpaceView`.
    """

    searchspace: _LayoutSearchSpaceView = field(
        validator=instance_of(_LayoutSearchSpaceView)
    )
    """The read-only layout-backed model context."""

    objective: Objective = field(validator=instance_of(Objective))
    """The objective for which the model is trained."""

    measurements: pd.DataFrame = field(validator=instance_of(pd.DataFrame))
    """The training data in experimental representation."""

    @classmethod
    def from_layout(
        cls,
        layout: _ConfigurationSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> _LayoutModelContext:
        """Create the context for a product-free layout.

        Args:
            layout: The product-free input layout.
            objective: The objective for which the model is trained.
            measurements: The training data in experimental representation.

        Returns:
            The layout-backed model context.
        """
        return cls(_LayoutSearchSpaceView(layout), objective, measurements)

    @property
    def task_idx(self) -> int | None:
        """The computational column index of the task parameter, if available."""
        return self.searchspace.task_idx

    @property
    def is_multitask(self) -> bool:
        """Indicates if model is to be operated in a multi-task context."""
        return self.n_task_dimensions > 0

    @property
    def n_task_dimensions(self) -> int:
        """The number of task dimensions."""
        return 1 if self.task_idx is not None else 0

    @property
    def n_tasks(self) -> int:
        """The number of tasks."""
        return self.searchspace.n_tasks

    @property
    def parameter_bounds(self):
        """The layout parameter bounds in BoTorch format."""
        import torch

        return torch.from_numpy(self.searchspace.scaling_bounds.to_numpy(copy=True))

    @property
    def numerical_indices(self) -> list[int]:
        """The indices of the regular numerical model inputs."""
        return [
            i
            for i in range(len(self.searchspace.comp_rep_columns))
            if i != self.task_idx
        ]


def _make_model_context(
    context: _FitInputContext,
    objective: Objective,
    measurements: pd.DataFrame,
) -> object:
    """Create the GP component context for a real or product-free fit context.

    Args:
        context: The real search space or product-free layout.
        objective: The objective for which the model is trained.
        measurements: The training data in experimental representation.

    Returns:
        A layout-aware or regular model context, depending on the given context.
    """
    if isinstance(context, _ConfigurationSpace):
        return _LayoutModelContext.from_layout(context, objective, measurements)

    from baybe.surrogates.gaussian_process.core import _ModelContext

    return _ModelContext(cast(SearchSpace, context), objective, measurements)


def _validate_layout_surrogate(surrogate: object, context: _FitInputContext, /) -> None:
    """Reject non-Gaussian-process surrogates on a product-free layout.

    Args:
        surrogate: The configured or retained surrogate instance.
        context: The actual fit context.

    Raises:
        IncompatibilityError: If a layout context receives a surrogate that is not a
            :class:`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate`.
    """
    if not isinstance(context, _ConfigurationSpace):
        return
    if not isinstance(surrogate, GaussianProcessSurrogate):
        raise IncompatibilityError(
            f"Surrogate '{type(surrogate).__name__}' is not supported on the "
            "product-free input layout. Use a GaussianProcessSurrogate or the real "
            "search-space fit path."
        )


def _fit_gaussian_process_on_layout(
    surrogate: GaussianProcessSurrogate,
    layout: _ConfigurationSpace,
    objective: Objective,
    measurements: pd.DataFrame,
    train_x,
    train_y,
) -> None:
    """Fit a Gaussian process surrogate on a product-free layout.

    This mirrors ``GaussianProcessSurrogate._fit`` exactly, but resolves its model
    context through the layout instead of a real search space.

    Args:
        surrogate: The Gaussian process surrogate to fit.
        layout: The product-free input layout.
        objective: The objective for which the model is fitted.
        measurements: The training data in experimental representation.
        train_x: The scaled training inputs.
        train_y: The scaled training targets.
    """
    # Symmetry-aware architecture adjustment (planned for future implementation)
    if surrogate._symmetries:
        raise NotImplementedError(
            "Symmetry-aware surrogate architecture is not yet implemented."
        )

    context = _LayoutModelContext.from_layout(layout, objective, measurements)

    # Check for custom kernel + multi-task clash (only relevant when the task
    # parameter has no kernel override, since the override mechanism handles task
    # kernel attachment explicitly).
    task_param = layout._task_parameter
    has_task_override = (
        task_param is not None and task_param.override_kernel is not None
    )
    if (
        context.is_multitask
        and surrogate._custom_kernel
        and not has_task_override
        and not strtobool(os.getenv("BAYBE_DISABLE_CUSTOM_KERNEL_WARNING", "False"))
    ):
        raise DeprecationError(
            f"We noticed that you are using a custom kernel architecture on a "
            f"search space that includes a '{TaskParameter.__name__}'. Please note "
            f"that the kernel logic of '{GaussianProcessSurrogate.__name__}' has "
            f"changed: the task kernel is no longer automatically added and must "
            f"now be explicitly included in your kernel (factory). "
            f"The '{ICMKernelFactory.__name__}' provides a suitable interface "
            f"for this purpose. If you are aware of this breaking change and wish "
            f"to proceed with your current kernel architecture, you can disable "
            f"this error by setting the 'BAYBE_DISABLE_CUSTOM_KERNEL_WARNING' "
            f"environment variable to a truthy value."
        )

    kernel, mean, likelihood, criterion = surrogate._resolve_components(context)

    import botorch

    ### Input/output scaling
    # NOTE: For GPs, we let BoTorch handle scaling (see [Scaling Workaround] above)
    input_transform = surrogate._make_input_transform(context)
    outcome_transform = surrogate._make_outcome_transform(train_y)

    ### Model construction and fitting
    surrogate._model = botorch.models.SingleTaskGP(
        train_x,
        train_y,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
    )
    mll = criterion.to_gpytorch(surrogate._model.likelihood, surrogate._model)
    botorch.fit.fit_gpytorch_mll(mll)


def _fit_layout_surrogate(
    surrogate: object,
    layout: _ConfigurationSpace,
    objective: Objective,
    measurements: pd.DataFrame,
) -> None:
    """Train a surrogate on a product-free input layout.

    The training context is retained in the surrogate's standard
    ``_searchspace`` slot, so the regular posterior and cache machinery continue to
    operate without any generic-surrogate changes.

    Args:
        surrogate: The built-in Gaussian process surrogate or its classifier
            subclass.
        layout: The product-free input layout of the fitting context.
        objective: The objective to be optimized.
        measurements: The training data in experimental representation.

    Raises:
        IncompatibleSurrogateError: If the objective needs a multi-output model that
            the surrogate does not support.
        IncompatibilityError: If the surrogate or a component factory is not verified
            for the product-free input layout.
        ValueError: If the layout contains task parameters but the selected surrogate
            model type does not support transfer learning.
    """
    if objective._is_multi_model and not surrogate.supports_multi_output:
        raise IncompatibleSurrogateError(
            f"You attempted to train a single-output surrogate in a "
            f"{len(objective.targets)}-target multi-output context. Either use "
            f"a proper multi-output surrogate or consider explicitly "
            f"replicating the current surrogate model using its "
            f"'.{surrogate.replicate.__name__}' method."
        )

    # When the context is unchanged, no retraining is necessary
    if (
        layout == surrogate._searchspace
        and objective == surrogate._objective
        and hash(measurements) == surrogate._measurements_hash
    ):
        return

    # Check if transfer learning capabilities are needed
    if (layout.n_tasks > 1) and (not surrogate.supports_transfer_learning):
        raise ValueError(
            f"The search space contains task parameters but the selected "
            f"surrogate model type ({surrogate.__class__.__name__}) does not "
            f"support transfer learning."
        )

    if not surrogate.supports_kernel_overrides and (
        names := [p.name for p in layout.parameters if p.override_kernel]
    ):
        warnings.warn(
            f"The selected surrogate model ({surrogate.__class__.__name__}) does not "
            f"use the kernel overrides specified for parameters {names}.",
            UnusedObjectWarning,
        )

    # Block partial measurements
    handle_missing_values(measurements, [t.name for t in objective.targets])

    # Fail before any expensive computation if the context cannot be fitted.
    _validate_layout_surrogate(surrogate, layout)

    # The layout is retained as the surrogate's context, so the regular posterior
    # path and the fit cache see exactly what was trained.
    surrogate._searchspace = layout
    surrogate._objective = objective
    surrogate._measurements = measurements
    surrogate._measurements_hash = hash(measurements)

    try:
        # Create context-specific transformations
        surrogate._input_scaler = surrogate._make_input_scaler(layout)
        surrogate._output_scaler = surrogate._make_output_scaler(
            objective, measurements
        )

        # Transform and fit. The target pre-transformation and Torch objectives
        # match the regular surrogate fit path.
        pre_transformed = objective._pre_transform(measurements, allow_extra=True)
        train_x_comp_rep, train_y_tensor = to_tensor(
            layout.transform(measurements, allow_extra=True), pre_transformed
        )
        train_x = surrogate._input_scaler.transform(train_x_comp_rep)
        train_y = (
            train_y_tensor
            if surrogate._output_scaler is _IDENTITY_TRANSFORM
            else surrogate._output_scaler(train_y_tensor)[0]
        )

        if type(surrogate) is GaussianProcessSurrogate:
            _fit_gaussian_process_on_layout(
                cast(GaussianProcessSurrogate, surrogate),
                layout,
                objective,
                measurements,
                train_x,
                train_y,
            )
        else:
            surrogate._fit(train_x, train_y)
    except Exception:
        # Only this layout path is transactional: a failed fit must not leave a
        # cached context, scalers, or model state the surrogate was not trained on.
        surrogate._searchspace = None
        surrogate._objective = None
        surrogate._measurements = None
        surrogate._measurements_hash = None
        surrogate._input_scaler = None
        surrogate._output_scaler = None
        if hasattr(surrogate, "_model"):
            surrogate._model = None
        raise
