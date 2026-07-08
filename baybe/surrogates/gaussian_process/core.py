"""Gaussian process surrogates."""

from __future__ import annotations

import gc
import importlib
import os
import warnings
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
from baybe.symmetries.base import Symmetry
from baybe.utils.boolean import strtobool
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import to_tensor

if TYPE_CHECKING:
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.models.transforms.input import InputTransform, Normalize
    from botorch.models.transforms.outcome import OutcomeTransform, Standardize
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
    """The factory used to create the kernel for the Gaussian process.

    Accepts:
        * :class:`baybe.kernels.base.Kernel`
        * :obj:`.components.kernel.KernelFactoryProtocol`
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
    """The factory used to create the mean function for the Gaussian process.

    Accepts:
        * :obj:`.components.mean.MeanFactoryProtocol`
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
    """The factory used to create the likelihood for the Gaussian process.

    Accepts:
        * :obj:`.components.likelihood.LikelihoodFactoryProtocol`
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
    """The fitting criterion for Gaussian process hyperparameter optimization.

    Accepts:
        * :class:`.components.fit_criterion.FitCriterion`
        * :obj:`.components.fit_criterion.FitCriterionFactoryProtocol`
    """

    _symmetries: tuple[Symmetry, ...] = field(factory=tuple, init=False, eq=False)
    """Symmetries for future architecture adjustments (e.g., invariant kernels)."""

    # TODO: type should be SingleTaskGP | None but is currently omitted due to:
    #   https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The fitted BoTorch model."""

    @staticmethod
    def _make_input_transform(context: _ModelContext) -> Normalize:
        """Create the input transform for the Gaussian process."""
        from botorch.models.transforms.input import Normalize

        return Normalize(
            len(context.searchspace.comp_rep_columns),
            bounds=context.parameter_bounds,
            indices=context.numerical_indices,
        )

    @staticmethod
    def _make_outcome_transform(train_y: Tensor) -> Standardize:
        """Create the (unfitted) outcome transform for the Gaussian process."""
        from botorch.models.transforms.outcome import Standardize

        outcome_transform = Standardize(m=train_y.shape[-1])
        outcome_transform(train_y)
        return outcome_transform

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
        """Create a Gaussian process surrogate from one of the defined presets.

        Unlike the regular constructor, where a ``None`` value for a factory argument
        defers to context-dependent auto-selection at fit time, a ``None`` value here
        falls back to the corresponding default of the chosen preset.

        Args:
            preset: The preset to use.
            kernel_or_factory: The kernel (factory) to use.
            mean_or_factory: The mean (factory) to use.
            likelihood_or_factory: The likelihood (factory) to use.
            fit_criterion_or_factory: The fit criterion (factory) to use.

        Returns:
            The Gaussian process surrogate configured according to the preset.
        """
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

    def posterior_mean_function(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> GPyTorchMean:
        """Create a GPyTorch mean module representing the surrogate's posterior mean.

        The method can be used to create the mean for a new
        :class:`GaussianProcessSurrogate` in two ways:

        * **Eagerly:** By calling the method and passing the returned module to a GP.
        * **Lazily:** By passing the bound method itself, without eagerly calling it.
            This works because the method signature complies with
            :class:`~.components.mean.MeanFactoryProtocol`, i.e., the new GP will use it
            as a factory and call it automatically at fit time.

        If the mean-providing GP has not been fitted at call time, its prior mean module
        is returned instead (which coincides with the posterior in this case) and a
        :class:`UserWarning` is emitted.

        Args:
            searchspace: The search space of the *new* GP.
            objective: The objective of the *new* GP.
            measurements: The training data of the *new* GP.

        Returns:
            A mean module ready to be used as the mean of a new
            :class:`GaussianProcessSurrogate`.
        """
        if self._model is None:
            warnings.warn(
                f"'{self.__class__.__name__}' has not been fitted yet. "
                f"Therefore, the prior mean is returned (which coincides with the "
                f"posterior in this case).",
                UserWarning,
            )
            mean_factory = self.mean_factory or BayBEMeanFactory()
            return mean_factory(searchspace, objective, measurements)

        context = _ModelContext(searchspace, objective, measurements)

        train_y = to_tensor(objective._pre_transform(measurements, allow_extra=True))
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)

        input_transform = self._make_input_transform(context)
        input_transform.eval()

        outcome_transform = self._make_outcome_transform(train_y)
        outcome_transform.eval()

        return _make_posterior_mean_module(
            self._model, input_transform, outcome_transform
        )

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
        # Model being fit is guaranteed by the call in `posterior`
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

        # Override is set: assemble the prescribed task kernel.
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

        # Default factory: reuse the ICM machinery on the full searchspace, which
        # builds the task-excluded base kernel and combines it with the prescribed
        # task kernel. This avoids the reduced searchspace, on which the default
        # factory's numerical kernel cannot resolve its active dimensions.
        if type(self.kernel_factory) is BayBEKernelFactory:
            icm = ICMKernelFactory(task_kernel_or_factory=task_kernel_spec)
            return icm(searchspace, context.objective, context.measurements)

        # Otherwise, build a task-free base kernel and attach the prescribed task
        # kernel manually.
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

        task_kernel = task_kernel_spec.to_gpytorch(searchspace=searchspace)
        return task_kernel if base_kernel is None else base_kernel * task_kernel

    def _resolve_components(
        self, context: _ModelContext
    ) -> tuple[GPyTorchKernel, GPyTorchMean, GPyTorchLikelihood, FitCriterion]:
        """Resolve factory fields to concrete components.

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
        assert self._searchspace is not None  # ensured by base class
        assert self._objective is not None  # ensured by base class
        assert self._measurements is not None  # ensured by base class

        # Symmetry-aware architecture adjustment (planned for future implementation)
        if self._symmetries:
            raise NotImplementedError(
                "Symmetry-aware surrogate architecture is not yet implemented."
            )
            for s in self._symmetries:
                s.validate_searchspace_context(self._searchspace)

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

        ### Input/output scaling
        # NOTE: For GPs, we let BoTorch handle scaling (see [Scaling Workaround] above)
        input_transform = self._make_input_transform(context)
        outcome_transform = self._make_outcome_transform(train_y)

        ### Model construction and fitting
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


def _make_posterior_mean_module(
    model: GPyTorchModel,
    input_transform: Normalize,
    outcome_transform: Standardize,
) -> GPyTorchMean:
    """Make a :class:`~gpytorch.means.Mean` that represents the posterior mean of a GP.

    Computationally, this is achieved by wrapping a deep copy of the provided GP with
    all parameters frozen, so that training a new GP consuming the module cannot alter
    the pretrained one.

    Transformations are applied to automatically align the spaces of the providing and
    consuming model, bridging the gap between their different modeling contexts:

    * **Input un-normalization**:
      When the new GP calls the produced module during training or inference, the inputs
      have already been normalized by the new GP's input transform. For this purpose,
      the inputs are un-normalized before passed to the pretrained GP. Without this
      step, the pretrained GP would receive inputs on the wrong scale and return
      meaningless predictions.

    * Output **un-standardization:**
      The produced mean values are expected in the original GP's output space, i.e.,
      the prior mean of the consuming GP should exactly match the posterior mean of the
      pretrained GP. However, the consuming GP transforms the output values of the
      mean module into its own scale, which would corrupt the result. To cancel out
      this effect, the inverse of this transformation is applied to the pretrained
      module output before passing it to the consuming GP.

    Args:
        model: The fitted GP whose posterior mean is to be extracted.
        input_transform: The new GP's input transform, used to un-normalize inputs.
        outcome_transform: The new GP's outcome transform, used to un-standardize
            outputs.

    Returns:
        A mean module ready for use in a new GP.
    """
    from copy import deepcopy

    import gpytorch

    frozen_model = deepcopy(model)
    for param in frozen_model.parameters():
        param.requires_grad = False
    frozen_model.eval()

    class _PosteriorMean(gpytorch.means.Mean):
        """GPyTorch mean wrapping a frozen GP's posterior."""

        def __init__(self) -> None:
            super().__init__()
            self.gp = frozen_model

        @override
        def forward(self, x: Tensor) -> Tensor:
            """Compute the prior mean in the new GP's standardized output space."""
            x_raw = input_transform.untransform(x)
            posterior_mean = self.gp.posterior(x_raw).mean
            standardized, _ = outcome_transform(posterior_mean)
            return standardized.squeeze(-1)

    return _PosteriorMean()


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
