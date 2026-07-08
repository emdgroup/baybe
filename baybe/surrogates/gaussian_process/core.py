"""Gaussian process surrogates."""

from __future__ import annotations

import gc
import importlib
import os
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from attrs import Converter, define, field
from attrs.converters import optional as optional_c
from attrs.converters import pipe
from attrs.validators import instance_of, is_callable, optional
from typing_extensions import Self, override

from baybe.exceptions import (
    DeprecationError,
    IncompatibleOverrideError,
    IncompatibleSearchSpaceError,
    ModelNotTrainedError,
    UnsupportedSearchSpaceAttributeError,
)
from baybe.kernels.base import Kernel
from baybe.objectives.base import Objective
from baybe.parameters.base import Parameter
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.enum import TransferLearningMode
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.components.fit_criterion import (
    FitCriterion,
    FitCriterionFactoryProtocol,
)
from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentType,
    to_component_factory,
)
from baybe.surrogates.gaussian_process.components.kernel import (
    ICMKernelFactory,
    KernelFactoryProtocol,
)
from baybe.surrogates.gaussian_process.components.likelihood import (
    LikelihoodFactoryProtocol,
)
from baybe.surrogates.gaussian_process.components.mean import MeanFactoryProtocol
from baybe.surrogates.gaussian_process.presets import (
    GaussianProcessPreset,
)
from baybe.surrogates.gaussian_process.presets.baybe import (
    BayBEFitCriterionFactory,
    BayBEKernelFactory,
    BayBELikelihoodFactory,
    BayBEMeanFactory,
)
from baybe.utils.boolean import strtobool
from baybe.utils.conversion import to_string

if TYPE_CHECKING:
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import Posterior
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.means import Mean as GPyTorchMean
    from torch import Tensor


@define
class _ModelContext:
    """Model context for :class:`GaussianProcessSurrogate`."""

    searchspace: SearchSpace = field(validator=instance_of(SearchSpace))
    """The search space the model is trained on."""

    objective: Objective = field(validator=instance_of(Objective))
    """The objective for which the model is trained."""

    measurements: pd.DataFrame = field(validator=instance_of(pd.DataFrame))
    """The training data in experimental representation."""

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
        # TODO: Generalize to multiple task parameters
        return 1 if self.task_idx is not None else 0

    @property
    def n_tasks(self) -> int:
        """The number of tasks."""
        return self.searchspace.n_tasks

    @property
    def parameter_bounds(self) -> Tensor:
        """Get the search space parameter bounds in BoTorch Format."""
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


def _mark_custom_kernel(
    value: Kernel | KernelFactoryProtocol | None, self: GaussianProcessSurrogate
) -> Kernel | KernelFactoryProtocol | None:
    """Mark the surrogate as using a custom kernel (for deprecation purposes)."""
    if value is not None and type(value) is not BayBEKernelFactory:
        self._custom_kernel = True

    return value


def _strip_task_from_kernel(
    kernel: Kernel, task_name: str, non_task_names: tuple[str, ...]
) -> Kernel | None:
    """Remove the task parameter from the parameters a BayBE kernel acts on.

    Args:
        kernel: The kernel whose task dependence should be removed.
        task_name: The name of the task parameter to strip.
        non_task_names: The names of all non-task parameters, used when the kernel
            does not explicitly specify the parameters it acts on.

    Raises:
        IncompatibleOverrideError: If the kernel is a composite kernel other than a
            single-level scale kernel, which cannot be stripped unambiguously.

    Returns:
        The stripped kernel, or ``None`` if stripping leaves no non-task parameters
        (i.e., the kernel acted on the task parameter only).
    """
    from attrs import evolve

    from baybe.kernels.base import BasicKernel
    from baybe.kernels.composite import ScaleKernel

    if isinstance(kernel, BasicKernel):
        if kernel.parameter_names is None:
            return evolve(kernel, parameter_names=non_task_names)
        if task_name not in kernel.parameter_names:
            return kernel
        remaining = tuple(n for n in kernel.parameter_names if n != task_name)
        if not remaining:
            return None
        return evolve(kernel, parameter_names=remaining)

    if isinstance(kernel, ScaleKernel):
        stripped = _strip_task_from_kernel(
            kernel.base_kernel, task_name, non_task_names
        )
        if stripped is None:
            raise IncompatibleOverrideError(
                f"The scale kernel '{type(kernel).__name__}' acts only on the task "
                f"parameter '{task_name}', which cannot be combined with an "
                f"'override_transfer_learning_mode'."
            )
        return evolve(kernel, base_kernel=stripped)

    raise IncompatibleOverrideError(
        f"Composite kernel '{type(kernel).__name__}' cannot be combined with an "
        f"'override_transfer_learning_mode'. Only basic kernels and scaled basic "
        f"kernels are supported."
    )


@define
class GaussianProcessSurrogate(Surrogate):
    """A Gaussian process surrogate model."""

    # TODO: Enable multi-target support via batching

    # Note [Scaling Workaround]
    # -------------------------
    # For GPs, we deactivate the base class scaling and instead let the botorch
    # model internally handle input/output scaling. The reason is that we need to
    # make `to_botorch` expose the actual botorch GP object, instead of going
    # via the `AdapterModel`, because certain acquisition functions (like qNIPV)
    # require the capability to `fantasize`, which the `AdapterModel` does not support.
    # The base class scaling thus needs to be disabled since otherwise the botorch GP
    # object would be trained on pre-scaled input/output data. This would cause a
    # problem since the resulting `posterior` method of that object is exposed
    # to `optimize_acqf_*`, which is configured to be called on the original scale.
    # Moving the scaling operation into the botorch GP object avoids this conflict.

    supports_transfer_learning: ClassVar[bool] = True
    # See base class.

    _custom_kernel: bool = field(init=False, default=False, repr=False, eq=False)
    # For deprecation only!

    kernel_factory: KernelFactoryProtocol | None = field(
        alias="kernel_or_factory",
        converter=pipe(  # type: ignore[misc]
            Converter(_mark_custom_kernel, takes_self=True),  # type: ignore[call-overload]
            optional_c(
                partial(to_component_factory, component_type=GPComponentType.KERNEL)
            ),
        ),
        default=None,
        validator=optional(is_callable()),
    )
    """The kernel factory. ``None`` defers to context-dependent auto-selection.

    Accepts:
        * :class:`baybe.kernels.base.Kernel`
        * Any :class:`~.components.generic.GPComponentFactoryProtocol` returning a
          kernel
        * :class:`gpytorch.kernels.Kernel`
    """

    mean_factory: MeanFactoryProtocol | None = field(
        alias="mean_or_factory",
        default=None,
        converter=optional_c(
            partial(to_component_factory, component_type=GPComponentType.MEAN)  # type: ignore[misc]
        ),
        validator=optional(is_callable()),
    )
    """The mean factory. ``None`` defers to context-dependent auto-selection.

    Accepts:
        * Any :class:`~.components.generic.GPComponentFactoryProtocol` returning a mean
        * :class:`gpytorch.means.Mean`
    """

    likelihood_factory: LikelihoodFactoryProtocol | None = field(
        alias="likelihood_or_factory",
        default=None,
        converter=optional_c(
            partial(to_component_factory, component_type=GPComponentType.LIKELIHOOD)  # type: ignore[misc]
        ),
        validator=optional(is_callable()),
    )
    """The likelihood factory. ``None`` defers to context-dependent auto-selection.

    Accepts:
        * Any :class:`~.components.generic.GPComponentFactoryProtocol` returning a
          likelihood
        * :class:`gpytorch.likelihoods.Likelihood`
    """

    fit_criterion_factory: FitCriterionFactoryProtocol | None = field(
        alias="fit_criterion_or_factory",
        default=None,
        converter=optional_c(
            partial(to_component_factory, component_type=GPComponentType.CRITERION)  # type: ignore[misc]
        ),
        validator=optional(is_callable()),
    )
    """The fit criterion factory. ``None`` defers to context-dependent auto-selection.

    Accepts:
        * :class:`.components.fit_criterion.FitCriterion`
        * Any :class:`~.components.generic.GPComponentFactoryProtocol` returning a
          fit criterion
    """

    _model = field(init=False, default=None, eq=False)
    """The fitted BoTorch model."""

    @classmethod
    def from_preset(
        cls,
        preset: GaussianProcessPreset | str,
        kernel_or_factory: KernelFactoryProtocol
        | Kernel
        | GPyTorchKernel
        | None = None,
        mean_or_factory: MeanFactoryProtocol | GPyTorchMean | None = None,
        likelihood_or_factory: LikelihoodFactoryProtocol
        | GPyTorchLikelihood
        | None = None,
        fit_criterion_or_factory: FitCriterion
        | FitCriterionFactoryProtocol
        | None = None,
    ) -> Self:
        """Create a Gaussian process surrogate from one of the defined presets."""
        preset = GaussianProcessPreset(preset)

        module_name = (
            f"baybe.surrogates.gaussian_process.presets.{preset.value.lower()}"
        )
        module = importlib.import_module(module_name)

        kernel = kernel_or_factory or getattr(module, "KERNEL_FACTORY")
        mean = mean_or_factory or getattr(module, "MEAN_FACTORY")
        likelihood = likelihood_or_factory or getattr(module, "LIKELIHOOD_FACTORY")
        fit_criterion = fit_criterion_or_factory or getattr(
            module, "FIT_CRITERION_FACTORY"
        )

        gp = cls(kernel, mean, likelihood, fit_criterion)
        gp._custom_kernel = False  # preset are first-party features
        return gp

    @override
    def to_botorch(self) -> GPyTorchModel:
        if self._model is None:
            raise ModelNotTrainedError(
                "The surrogate must be trained before a BoTorch model can be created."
            )
        return self._model

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
        return None

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        assert self._model is not None
        return self._model.posterior(candidates_comp_scaled)

    def _resolve_kernel(self, context: _ModelContext) -> GPyTorchKernel:
        """Resolve the GP kernel, dispatching on task parameter overrides.

        Args:
            context: The model context providing searchspace information.

        Raises:
            IncompatibleOverrideError: If a transfer learning override is combined
                with a kernel or kernel factory that cannot be reduced to a task-free
                base kernel operating on parameter names.

        Returns:
            The constructed gpytorch kernel.
        """
        from baybe.kernels.basic import IndexKernel, PositiveIndexKernel
        from baybe.surrogates.gaussian_process.components.generic import (
            PlainGPComponentFactory,
        )

        searchspace = context.searchspace
        task_param = searchspace._task_parameter

        if (
            task_param is None
            or (tl_override := task_param.override_transfer_learning_mode) is None
        ):
            # No override: let the factory handle everything (default path)
            kernel_factory = self.kernel_factory or BayBEKernelFactory()
            kernel = kernel_factory(
                searchspace, context.objective, context.measurements
            )
            if isinstance(kernel, Kernel):
                kernel = kernel.to_gpytorch(searchspace=searchspace)
            return kernel

        # Override is set: build a task-free base kernel and attach the prescribed
        # task kernel manually.
        non_task_names = tuple(
            p.name for p in searchspace.parameters if p.name != task_param.name
        )
        effective_factory = self.kernel_factory or BayBEKernelFactory()
        incompatible_message = (
            f"The '{TaskParameter.__name__}' '{task_param.name}' specifies "
            f"'override_transfer_learning_mode={tl_override.name}', which requires a "
            f"kernel (factory) that yields a task-free BayBE kernel operating on "
            f"parameter names. The provided kernel factory "
            f"'{type(effective_factory).__name__}' does not satisfy this (e.g., it "
            f"returns a raw gpytorch kernel or already operates on the task "
            f"parameter)."
        )

        if isinstance(self.kernel_factory, PlainGPComponentFactory):
            # A fixed kernel was provided: strip the task parameter directly.
            component = self.kernel_factory.component
            if not isinstance(component, Kernel):
                raise IncompatibleOverrideError(incompatible_message)
            base_spec = _strip_task_from_kernel(
                component, task_param.name, non_task_names
            )
        else:
            # Call the factory on a reduced (task-free) searchspace so that it
            # produces only the base kernel. Factories that need computational
            # information unavailable on the reduced space, or that return a raw
            # gpytorch kernel, are not supported.
            reduced_searchspace = searchspace._drop_parameters({task_param.name})
            try:
                factory_kernel = effective_factory(
                    reduced_searchspace, context.objective, context.measurements
                )
            except (
                IncompatibleSearchSpaceError,
                UnsupportedSearchSpaceAttributeError,
            ) as ex:
                raise IncompatibleOverrideError(incompatible_message) from ex
            if not isinstance(factory_kernel, Kernel):
                raise IncompatibleOverrideError(incompatible_message)
            base_spec = factory_kernel

        # Convert the base kernel on the full searchspace so that parameter names
        # resolve to the correct computational column indices.
        base_kernel = (
            None
            if base_spec is None
            else base_spec.to_gpytorch(searchspace=searchspace)
        )

        # Create the task kernel based on the override mode
        n_tasks = searchspace.n_tasks
        if tl_override is TransferLearningMode.POSITIVE_INDEX_KERNEL:
            task_kernel_spec: Kernel = PositiveIndexKernel(
                num_tasks=n_tasks, rank=n_tasks, parameter_names=(task_param.name,)
            )
        elif tl_override is TransferLearningMode.INDEX_KERNEL:
            task_kernel_spec = IndexKernel(
                num_tasks=n_tasks, rank=n_tasks, parameter_names=(task_param.name,)
            )
        else:
            raise RuntimeError(
                f"Unhandled '{TransferLearningMode.__name__}' '{tl_override.name}'."
            )

        task_kernel = task_kernel_spec.to_gpytorch(searchspace=searchspace)
        return task_kernel if base_kernel is None else base_kernel * task_kernel

    def _resolve_components(
        self, context: _ModelContext
    ) -> tuple[GPyTorchKernel, GPyTorchMean, GPyTorchLikelihood, FitCriterion]:
        """Resolve factory fields to concrete GPyTorch components.

        Resolves ``None`` fields to BayBE defaults and calls the factories with
        the given context. This handles the standard resolution path.

        Args:
            context: The model context providing searchspace, objective, and
                measurements.

        Returns:
            A tuple of (kernel, mean, likelihood, criterion).
        """
        mean_factory = self.mean_factory or BayBEMeanFactory()
        likelihood_factory = self.likelihood_factory or BayBELikelihoodFactory()
        criterion_factory = self.fit_criterion_factory or BayBEFitCriterionFactory()

        kernel = self._resolve_kernel(context)

        mean = mean_factory(
            context.searchspace, context.objective, context.measurements
        )

        likelihood = likelihood_factory(
            context.searchspace, context.objective, context.measurements
        )

        criterion = criterion_factory(
            context.searchspace, context.objective, context.measurements
        )

        return kernel, mean, likelihood, criterion

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit a SingleTaskGP with resolved components.

        Args:
            train_x: Training inputs in computational representation.
            train_y: Training targets (pre-transformed).

        Raises:
            DeprecationError: If a custom kernel is used in a multi-task context.
        """
        assert self._searchspace is not None  # provided by base class
        assert self._objective is not None  # provided by base class
        assert self._measurements is not None  # provided by base class
        context = _ModelContext(self._searchspace, self._objective, self._measurements)

        # Check for custom kernel + multi-task clash (only relevant when no
        # override_transfer_learning_mode is set, since the override mechanism
        # handles task kernel attachment explicitly).
        task_param = context.searchspace._task_parameter
        has_tl_override = (
            task_param is not None
            and task_param.override_transfer_learning_mode is not None
        )
        if (
            context.is_multitask
            and self._custom_kernel
            and not has_tl_override
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

        kernel, mean, likelihood, criterion = self._resolve_components(context)

        import botorch
        from botorch.models.transforms import Normalize, Standardize

        input_transform = Normalize(
            train_x.shape[-1],
            bounds=context.parameter_bounds,
            indices=context.numerical_indices,
        )
        outcome_transform = Standardize(train_y.shape[-1])
        self._model = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean,
            covar_module=kernel,
            likelihood=likelihood,
        )
        mll = criterion.to_gpytorch(self._model.likelihood, self._model)
        botorch.fit.fit_gpytorch_mll(mll)

    @override
    def __str__(self) -> str:
        fields = [
            to_string(
                "Kernel factory", self.kernel_factory or "auto", single_line=True
            ),
            to_string("Mean factory", self.mean_factory or "auto", single_line=True),
            to_string(
                "Likelihood factory",
                self.likelihood_factory or "auto",
                single_line=True,
            ),
            to_string(
                "Fit criterion factory",
                self.fit_criterion_factory or "auto",
                single_line=True,
            ),
        ]
        return to_string(super().__str__(), *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
