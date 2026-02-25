"""Gaussian process surrogates."""

from __future__ import annotations

import gc
import importlib
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import Self, override

from baybe.kernels.base import Kernel
from baybe.parameters.base import Parameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.components.generic import to_component_factory
from baybe.surrogates.gaussian_process.components.kernel import (
    KernelFactory,
)
from baybe.surrogates.gaussian_process.components.likelihood import LikelihoodFactory
from baybe.surrogates.gaussian_process.components.mean import MeanFactory
from baybe.surrogates.gaussian_process.presets import (
    GaussianProcessPreset,
)
from baybe.surrogates.gaussian_process.presets.baybe import (
    DefaultKernelFactory,
    DefaultLikelihoodFactory,
    DefaultMeanFactory,
)
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

        return torch.from_numpy(self.searchspace.scaling_bounds.values)

    @property
    def numerical_indices(self) -> list[int]:
        """The indices of the regular numerical model inputs."""
        return [
            i
            for i in range(len(self.searchspace.comp_rep_columns))
            if i != self.task_idx
        ]


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

    kernel_factory: KernelFactory = field(
        alias="kernel_or_factory",
        factory=DefaultKernelFactory,
        converter=to_component_factory,
    )
    """The factory used to create the kernel for the Gaussian process.

    Accepts:
        * :class:`baybe.kernels.base.Kernel`
        * :class:`.kernel_factory.KernelFactory`
        * :class:`gpytorch.kernels.Kernel`
    """

    mean_factory: MeanFactory = field(
        alias="mean_or_factory",
        factory=DefaultMeanFactory,
        converter=to_component_factory,
    )
    """The factory used to create the mean function for the Gaussian process.

    Accepts:
        * :class:`.mean_factory.MeanFactory`
        * :class:`gpytorch.means.Mean`
    """

    likelihood_factory: LikelihoodFactory = field(
        alias="likelihood_or_factory",
        factory=DefaultLikelihoodFactory,
        converter=to_component_factory,
    )
    """The factory used to create the likelihood for the Gaussian process.

    Accepts:
        * :class:`.likelihood_factory.LikelihoodFactory`
        * :class:`gpytorch.likelihoods.Likelihood`
    """

    # TODO: type should be Optional[botorch.models.SingleTaskGP] but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @classmethod
    def from_preset(
        cls,
        preset: GaussianProcessPreset | str,
        kernel_or_factory: KernelFactory | Kernel | GPyTorchKernel | None = None,
        mean_or_factory: MeanFactory | GPyTorchMean | None = None,
        likelihood_or_factory: LikelihoodFactory | GPyTorchLikelihood | None = None,
    ) -> Self:
        """Create a Gaussian process surrogate from one of the defined presets."""
        preset = GaussianProcessPreset(preset)

        module_name = (
            f"baybe.surrogates.gaussian_process.presets.{preset.value.lower()}"
        )
        module = importlib.import_module(module_name)

        kernel = kernel_or_factory or getattr(module, "PresetKernelFactory")()
        mean = mean_or_factory or getattr(module, "PresetMeanFactory")()
        likelihood = (
            likelihood_or_factory or getattr(module, "PresetLikelihoodFactory")()
        )

        return cls(kernel, mean, likelihood)

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
        import gpytorch

        assert self._searchspace is not None  # provided by base class
        context = _ModelContext(self._searchspace)

        ### Input/output scaling
        # NOTE: For GPs, we let BoTorch handle scaling (see [Scaling Workaround] above)
        input_transform = botorch.models.transforms.Normalize(  # type: ignore[attr-defined]
            train_x.shape[-1],
            bounds=context.parameter_bounds,
            indices=context.numerical_indices,
        )
        outcome_transform = botorch.models.transforms.Standardize(train_y.shape[-1])  # type: ignore[attr-defined]

        ### Mean
        mean = self.mean_factory(context.searchspace, train_x, train_y)

        ### Kernel
        kernel = self.kernel_factory(context.searchspace, train_x, train_y)
        if isinstance(kernel, Kernel):
            kernel_num_dims = train_x.shape[-1] - context.n_task_dimensions
            kernel = kernel.to_gpytorch(
                ard_num_dims=kernel_num_dims,
                active_dims=context.numerical_indices,
            )

        if context.is_multitask:
            task_kernel = gpytorch.kernels.IndexKernel(
                num_tasks=context.n_tasks,
                active_dims=context.task_idx,
                rank=context.n_tasks,  # TODO: make controllable
            )
            kernel = kernel * task_kernel

        ### Likelihood
        likelihood = self.likelihood_factory(context.searchspace, train_x, train_y)

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

        # TODO: This is still a temporary workaround to avoid overfitting seen in
        #  low-dimensional TL cases. More robust settings are being researched.
        if context.n_task_dimensions > 0:
            mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(
                self._model.likelihood, self._model
            )
        else:
            mll = gpytorch.ExactMarginalLogLikelihood(
                self._model.likelihood, self._model
            )

        botorch.fit.fit_gpytorch_mll(mll)

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Kernel factory", self.kernel_factory, single_line=True),
        ]
        return to_string(super().__str__(), *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
