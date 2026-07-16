"""Gaussian process surrogates."""

from __future__ import annotations

import gc
import importlib
import os
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from attrs import Converter, define, field
from attrs.converters import pipe
from attrs.validators import instance_of, is_callable
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
    value: Kernel | KernelFactoryProtocol, self: GaussianProcessSurrogate
) -> Kernel | KernelFactoryProtocol:
    """Mark the surrogate as using a custom kernel (for deprecation purposes)."""
    if type(value) is not BayBEKernelFactory:
        self._custom_kernel = True

    return value


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

    kernel_factory: KernelFactoryProtocol = field(
        alias="kernel_or_factory",
        converter=pipe(  # type: ignore[misc]
            Converter(_mark_custom_kernel, takes_self=True),  # type: ignore[call-overload]
            partial(to_component_factory, component_type=GPComponentType.KERNEL),
        ),
        factory=BayBEKernelFactory,
        validator=is_callable(),
    )
    """The factory used to create the kernel for the Gaussian process.

    Accepts:
        * :class:`baybe.kernels.base.Kernel`
        * :class:`.components.kernel.KernelFactory`
        * :class:`gpytorch.kernels.Kernel`
    """

    mean_factory: MeanFactoryProtocol = field(
        alias="mean_or_factory",
        factory=BayBEMeanFactory,
        converter=partial(to_component_factory, component_type=GPComponentType.MEAN),  # type: ignore[misc]
        validator=is_callable(),
    )
    """The factory used to create the mean function for the Gaussian process.

    Accepts:
        * :class:`.components.mean.MeanFactory`
        * :class:`gpytorch.means.Mean`
    """

    likelihood_factory: LikelihoodFactoryProtocol = field(
        alias="likelihood_or_factory",
        factory=BayBELikelihoodFactory,
        converter=partial(  # type: ignore[misc]
            to_component_factory, component_type=GPComponentType.LIKELIHOOD
        ),
        validator=is_callable(),
    )
    """The factory used to create the likelihood for the Gaussian process.

    Accepts:
        * :class:`.components.likelihood.LikelihoodFactory`
        * :class:`gpytorch.likelihoods.Likelihood`
    """

    fit_criterion_factory: FitCriterionFactoryProtocol = field(
        alias="fit_criterion_or_factory",
        factory=BayBEFitCriterionFactory,
        converter=partial(  # type: ignore[misc]
            to_component_factory, component_type=GPComponentType.CRITERION
        ),
        validator=is_callable(),
    )
    """The fitting criterion for Gaussian process hyperparameter optimization.

    Accepts:
        * :class:`.components.fit_criterion.FitCriterion`
        * :class:`.components.fit_criterion.FitCriterionFactoryProtocol`
    """

    # TODO: type should be Optional[botorch.models.SingleTaskGP] but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

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

        return Standardize(m=train_y.shape[-1])

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

    def posterior_mean_function(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> GPyTorchMean:
        """Return a GPyTorch mean module representing the surrogate's posterior mean.

        The returned mean module wraps a frozen copy of this surrogate's fitted GP.
        Its posterior mean is evaluated in the raw input space and mapped into the
        normalized input/output space defined by the given ``searchspace``,
        ``objective``, and ``measurements``.

        The module is intended and designed to be used as the prior mean of another
        :class:`GaussianProcessSurrogate`, so that this surrogate's posterior mean
        acts as that GP's prior mean. To this end, the bound method satisfies
        :class:`~baybe.surrogates.gaussian_process.components.mean.MeanFactoryProtocol`
        and can be passed to it directly.

        Args:
            searchspace: The search space defining the module's input space.
            objective: The objective defining the module's output space.
            measurements: The data defining the module's input/output normalization.

        Returns:
            The posterior mean module.

        Raises:
            ModelNotTrainedError: If the surrogate has not been fitted yet.
        """
        if self._model is None:
            raise ModelNotTrainedError(
                f"'{self.__class__.__name__}' must be fitted before its "
                f"'{self.posterior_mean_function.__name__}' can be used as a "
                f"mean function."
            )

        context = _ModelContext(searchspace, objective, measurements)

        train_y = to_tensor(objective._pre_transform(measurements, allow_extra=True))
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)

        input_transform = self._make_input_transform(context)
        input_transform.eval()

        outcome_transform = self._make_outcome_transform(train_y)
        outcome_transform(train_y)  # fit means/stds; nothing else fits it here
        outcome_transform.eval()

        return _make_posterior_mean_module(
            self._model, input_transform, outcome_transform
        )

    @override
    def to_botorch(self) -> GPyTorchModel:
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
        return self._model.posterior(candidates_comp_scaled)

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        import botorch

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

        ### Input/output scaling
        # NOTE: For GPs, we let BoTorch handle scaling (see [Scaling Workaround] above)
        input_transform = self._make_input_transform(context)
        outcome_transform = self._make_outcome_transform(train_y)

        ### Mean
        mean = self.mean_factory(
            context.searchspace, context.objective, context.measurements
        )

        ### Kernel
        kernel = self.kernel_factory(
            context.searchspace, context.objective, context.measurements
        )
        if isinstance(kernel, Kernel):
            kernel = kernel.to_gpytorch(searchspace=context.searchspace)

        ### Likelihood
        likelihood = self.likelihood_factory(
            context.searchspace, context.objective, context.measurements
        )

        ### Criterion
        criterion = self.fit_criterion_factory(
            context.searchspace, context.objective, context.measurements
        )

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
            to_string("Kernel factory", self.kernel_factory, single_line=True),
            to_string("Mean factory", self.mean_factory, single_line=True),
            to_string("Likelihood factory", self.likelihood_factory, single_line=True),
            to_string(
                "Fit criterion factory", self.fit_criterion_factory, single_line=True
            ),
        ]
        return to_string(super().__str__(), *fields)


def _make_posterior_mean_module(
    model: GPyTorchModel,
    input_transform: Normalize,
    outcome_transform: Standardize,
) -> GPyTorchMean:
    """Make a :class:`gpytorch.means.Mean` wrapping a frozen copy of a GP.

    Args:
        model: The fitted GP whose posterior mean becomes the new GP's prior mean.
            It is deep-copied and frozen so that fitting the new GP cannot alter it.
        input_transform: The new GP's input transform; used to un-normalize the
            inputs ``x`` arriving at ``forward`` so the wrapped GP sees raw inputs.
        outcome_transform: The new GP's outcome transform; used to standardize the
            wrapped GP's raw-space predictions into the new GP's output space.

    Returns:
        A mean module suitable for use as ``mean_module`` of the new GP.
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
        def train(self, mode: bool = True) -> _PosteriorMean:
            """Set training mode without propagating to the frozen inner GP.

            The inner GP stays in eval mode so ``posterior(x)`` always returns
            predictive outputs regardless of the outer GP's training state.
            """
            self.training = mode
            return self

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
