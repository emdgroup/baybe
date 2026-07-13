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

from baybe.exceptions import DeprecationError, ModelNotTrainedError
from baybe.kernels.base import Kernel
from baybe.objectives.base import Objective
from baybe.parameters.base import Parameter
from baybe.parameters.categorical import TaskParameter
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
    from botorch.models import SingleTaskGP
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


@define
class _GaussianProcessSurrogate:
    """Internal model builder for :class:`GaussianProcessSurrogate`.

    Receives already-resolved GPyTorch components and performs straightforward
    :class:`~botorch.models.SingleTaskGP` construction with no factory calls,
    no context inspection, and no branching. Never exposed to users or the
    serialization layer.
    """

    kernel: GPyTorchKernel = field(eq=False)
    """The resolved GPyTorch kernel."""

    mean: GPyTorchMean = field(eq=False)
    """The resolved GPyTorch mean function."""

    likelihood: GPyTorchLikelihood = field(eq=False)
    """The resolved GPyTorch likelihood."""

    criterion: FitCriterion = field()
    """The resolved fitting criterion."""

    _model: SingleTaskGP | None = field(init=False, default=None, eq=False)
    """The fitted BoTorch model."""

    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Fit the GP from BayBE objects, satisfying :class:`SurrogateProtocol`.

        Args:
            searchspace: The search space the model is trained on.
            objective: The objective for which the model is trained.
            measurements: The training data in experimental representation.
        """
        from baybe.utils.dataframe import to_tensor

        context = _ModelContext(searchspace, objective, measurements)
        pre_transformed = objective._pre_transform(measurements, allow_extra=True)
        train_x, train_y = to_tensor(
            searchspace.transform(measurements, allow_extra=True), pre_transformed
        )
        self._fit_from_tensors(train_x, train_y, context)

    def _fit_from_tensors(
        self,
        train_x: Tensor,
        train_y: Tensor,
        context: _ModelContext,
    ) -> None:
        """Build and fit the SingleTaskGP from pre-processed tensors.

        This method exists as a performance optimisation for the case where the caller
        (i.e. :class:`GaussianProcessSurrogate`) has already computed ``train_x`` and
        ``train_y`` through the standard :class:`~baybe.surrogates.base.Surrogate`
        pipeline and can pass them directly, avoiding a redundant second pass through
        :meth:`~baybe.searchspace.core.SearchSpace.transform` and
        :func:`~baybe.utils.dataframe.to_tensor` that would otherwise occur if
        :meth:`fit` were called with the original BayBE objects instead.

        Args:
            train_x: Training inputs in computational representation.
            train_y: Training targets (pre-transformed).
            context: The model context providing bounds and index information.
        """
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
            mean_module=self.mean,
            covar_module=self.kernel,
            likelihood=self.likelihood,
        )
        mll = self.criterion.to_gpytorch(self._model.likelihood, self._model)
        botorch.fit.fit_gpytorch_mll(mll)

    def to_botorch(self) -> GPyTorchModel:
        """Return the fitted BoTorch model, satisfying :class:`SurrogateProtocol`."""
        if self._model is None:
            raise RuntimeError(f"'{self.__class__.__name__}' has not been fitted yet.")
        return self._model


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

    _kernel_factory: KernelFactoryProtocol | None = field(
        alias="kernel_or_factory",
        converter=pipe(  # type: ignore[misc]
            Converter(_mark_custom_kernel, takes_self=True),  # type: ignore[call-overload]
            optional_c(
                partial(to_component_factory, component_type=GPComponentType.KERNEL)
            ),
        ),
        default=None,
        repr=False,
        validator=optional(is_callable()),
    )
    """The used kernel factory. ``None`` defers to the BayBE default.

    Accepts a :class:`baybe.kernels.base.Kernel`, a ``KernelFactoryProtocol``, or a
    :class:`gpytorch.kernels.Kernel`.
    """

    _mean_factory: MeanFactoryProtocol | None = field(
        alias="mean_or_factory",
        default=None,
        converter=optional_c(
            partial(to_component_factory, component_type=GPComponentType.MEAN)  # type: ignore[misc]
        ),
        repr=False,
        validator=optional(is_callable()),
    )
    """The used mean factory. ``None`` defers to the BayBE default.

    Accepts a ``MeanFactoryProtocol`` or a :class:`gpytorch.means.Mean`.
    """

    _likelihood_factory: LikelihoodFactoryProtocol | None = field(
        alias="likelihood_or_factory",
        default=None,
        converter=optional_c(
            partial(to_component_factory, component_type=GPComponentType.LIKELIHOOD)  # type: ignore[misc]
        ),
        repr=False,
        validator=optional(is_callable()),
    )
    """The used likelihood factory. ``None`` defers to the BayBE default.

    Accepts a ``LikelihoodFactoryProtocol`` or a
    :class:`gpytorch.likelihoods.Likelihood`.
    """

    _fit_criterion_factory: FitCriterionFactoryProtocol | None = field(
        alias="fit_criterion_or_factory",
        default=None,
        converter=optional_c(
            partial(to_component_factory, component_type=GPComponentType.CRITERION)  # type: ignore[misc]
        ),
        repr=False,
        validator=optional(is_callable()),
    )
    """The used fit criterion factory. ``None`` defers to the BayBE default.

    Accepts a :class:`.components.fit_criterion.FitCriterion` or a
    ``FitCriterionFactoryProtocol``.
    """

    _inner: _GaussianProcessSurrogate | None = field(init=False, default=None, eq=False)
    """The fitted internal model instance. Available after fitting."""

    @property
    def fit_criterion_factory(self) -> FitCriterionFactoryProtocol:
        """The fit criterion factory used during model fitting."""
        return self._fit_criterion_factory or BayBEFitCriterionFactory()

    @property
    def kernel_factory(self) -> KernelFactoryProtocol:
        """The kernel factory used during model fitting."""
        return self._kernel_factory or BayBEKernelFactory()

    @property
    def likelihood_factory(self) -> LikelihoodFactoryProtocol:
        """The likelihood factory used during model fitting."""
        return self._likelihood_factory or BayBELikelihoodFactory()

    @property
    def mean_factory(self) -> MeanFactoryProtocol:
        """The mean factory used during model fitting."""
        return self._mean_factory or BayBEMeanFactory()

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
        if self._inner is None:
            raise ModelNotTrainedError(
                "The surrogate must be trained before a BoTorch model can be created."
            )
        return self._inner.to_botorch()

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
        assert self._inner is not None
        return self._inner.to_botorch().posterior(candidates_comp_scaled)

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        assert self._searchspace is not None  # provided by base class
        assert self._objective is not None  # provided by base class
        assert self._measurements is not None  # provided by base class
        context = _ModelContext(self._searchspace, self._objective, self._measurements)

        if (
            context.is_multitask
            and self._custom_kernel
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

        ### Component resolution
        mean = self.mean_factory(
            context.searchspace, context.objective, context.measurements
        )

        kernel = self.kernel_factory(
            context.searchspace, context.objective, context.measurements
        )
        if isinstance(kernel, Kernel):
            kernel = kernel.to_gpytorch(searchspace=context.searchspace)

        likelihood = self.likelihood_factory(
            context.searchspace, context.objective, context.measurements
        )

        criterion = self.fit_criterion_factory(
            context.searchspace, context.objective, context.measurements
        )

        ### Create internal instance with resolved components and delegate fitting
        inner = _GaussianProcessSurrogate(
            kernel=kernel, mean=mean, likelihood=likelihood, criterion=criterion
        )
        inner._fit_from_tensors(train_x, train_y, context)
        self._inner = inner

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Kernel factory", self.kernel_factory, single_line=True),
            to_string("Mean factory", self.mean_factory, single_line=True),
            to_string("Likelihood factory", self.likelihood_factory, single_line=True),
            to_string(
                "Fit criterion factory", self.fit_criterion_factory, single_line=True
            ),
        ]
        return to_string(super().__str__(), *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
