"""Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar, Literal

from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.parameters.base import Parameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.kernel_factory import (
    KernelFactory,
    to_kernel_factory,
)
from baybe.surrogates.gaussian_process.presets import (
    GaussianProcessPreset,
    make_gp_from_preset,
)
from baybe.surrogates.gaussian_process.presets.default import (
    DefaultKernelFactory,
    _default_noise_factory,
)
from baybe.surrogates.gaussian_process.prior_modules import PriorMean, PriorKernel
from baybe.utils.conversion import to_string

if TYPE_CHECKING:
    from botorch.models.model import Model
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import Posterior
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

    def get_numerical_indices(self, n_inputs: int) -> tuple[int, ...]:
        """Get the indices of the regular numerical model inputs."""
        return tuple(i for i in range(n_inputs) if i != self.task_idx)


@define
class GaussianProcessSurrogate(Surrogate):
    """A Gaussian process surrogate model."""

    # TODO: Enable multi-target support via batching

    # Note [Scaling Workaround]
    # -------------------------
    # For GPs, we deactivate the base class scaling and instead let the botorch
    # model internally handle input/output scaling. The reasons is that we need to
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
        converter=to_kernel_factory,
    )
    """The factory used to create the kernel of the Gaussian process.

    Accepts either a :class:`baybe.kernels.base.Kernel` or a
    :class:`.kernel_factory.KernelFactory`.
    When passing a :class:`baybe.kernels.base.Kernel`, it gets automatically wrapped
    into a :class:`.kernel_factory.PlainKernelFactory`."""

    # TODO: type should be Optional[botorch.models.SingleTaskGP] but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    # Transfer learning fields
    _prior_gp = field(init=False, default=None, eq=False)
    """Prior GP to extract mean/covariance from for transfer learning."""

    _transfer_mode: Literal["mean", "kernel"] | None = field(init=False, default=None, eq=False)
    """Transfer learning mode: 'mean' uses prior as mean function, 'kernel' uses prior covariance."""

    @staticmethod
    def from_preset(preset: GaussianProcessPreset) -> GaussianProcessSurrogate:
        """Create a Gaussian process surrogate from one of the defined presets."""
        return make_gp_from_preset(preset)

    @classmethod
    def from_prior_gp(
        cls,
        prior_gp,
        transfer_mode: Literal["mean", "kernel"] = "mean",
        kernel_factory: KernelFactory | None = None,
        **kwargs
    ) -> GaussianProcessSurrogate:
        """Create a GP surrogate from a prior GP.

        Args:
            prior_gp: Fitted SingleTaskGP to use as prior 
            transfer_mode: "mean" extracts posterior mean as prior mean,
                          "kernel" uses prior's covariance
            kernel_factory: Kernel factory for new covariance (required for mean mode,
                           ignored for kernel mode)
            **kwargs: Additional arguments passed to GaussianProcessSurrogate constructor

        Returns:
            New GaussianProcessSurrogate instance with prior mean or covariance

        Raises:
            ValueError: If prior_gp is not fitted or configuration is invalid
        """
        from copy import deepcopy
        from botorch.models import SingleTaskGP

        # Validate prior GP is fitted
        if not isinstance(prior_gp, SingleTaskGP):
            raise ValueError("prior_gp must be a fitted SingleTaskGP instance")
        if not hasattr(prior_gp, 'train_inputs') or prior_gp.train_inputs is None:
            raise ValueError("Prior GP must be fitted (have train_inputs) before use")

        # Validate transfer mode configuration
        if transfer_mode not in ["mean", "kernel"]:
            raise ValueError("transfer_mode must be 'mean' or 'kernel'")

        if transfer_mode == "mean" and kernel_factory is None:
            raise ValueError("kernel_factory is required for mean transfer mode")

        # For kernel transfer, kernel_factory is ignored (we use prior's kernel)
        if transfer_mode == "kernel":
            kernel_factory = kernel_factory or DefaultKernelFactory()

        # Create new surrogate instance
        instance = cls(kernel_or_factory=kernel_factory, **kwargs)

        # Configure for transfer learning
        instance._prior_gp = deepcopy(prior_gp)
        instance._transfer_mode = transfer_mode

        return instance

    @override
    def to_botorch(self) -> Model:
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

    def _initialize_model(
        self,
        train_x: Tensor,
        train_y: Tensor,
        context: _ModelContext,
        batch_shape,
    ) -> None:
        """Initialize the GP model with appropriate mean and covariance modules.

        Handles both standard GP creation and creation of GP from given prior.

        Args:
            train_x: Training input data
            train_y: Training target data
            context: Model context containing searchspace information
            batch_shape: Batch shape for the training data
        """
        import botorch
        import gpytorch
        import torch

        numerical_idxs = context.get_numerical_indices(train_x.shape[-1])

        # Configure input/output transforms
        if self._prior_gp is not None:
            # Use prior's transforms for consistency in transfer learning
            input_transform = self._prior_gp.input_transform
            outcome_transform = self._prior_gp.outcome_transform
        else:
            # Standard transform setup
            input_transform = botorch.models.transforms.Normalize(
                train_x.shape[-1], bounds=context.parameter_bounds, indices=numerical_idxs
            )
            outcome_transform = botorch.models.transforms.Standardize(train_y.shape[-1])

        # Configure mean module
        if self._prior_gp is not None and self._transfer_mode == "mean":
            mean_module = PriorMean(self._prior_gp, batch_shape=batch_shape)
        else:
            mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        # Configure base covariance module
        if self._prior_gp is not None and self._transfer_mode == "kernel":
            base_covar_module = PriorKernel(self._prior_gp.covar_module)
        else:
            # Use kernel factory
            base_covar_module = self.kernel_factory(
                context.searchspace, train_x, train_y
            ).to_gpytorch(
                ard_num_dims=train_x.shape[-1] - context.n_task_dimensions,
                active_dims=numerical_idxs,
                batch_shape=batch_shape,
            )

        # Handle multi-task covariance (keep existing logic)
        if not context.is_multitask:
            covar_module = base_covar_module
        else:
            task_covar_module = gpytorch.kernels.IndexKernel(
                num_tasks=context.n_tasks,
                active_dims=context.task_idx,
                rank=context.n_tasks,
            )
            covar_module = base_covar_module * task_covar_module

        # Configure likelihood (keep existing logic)
        noise_prior = _default_noise_factory(context.searchspace, train_x, train_y)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior[0].to_gpytorch(), batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # Create the model
        self._model = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        import botorch
        import gpytorch

        # FIXME[typing]: It seems there is currently no better way to inform the type
        #   checker that the attribute is available at the time of the function call
        assert self._searchspace is not None

        context = _ModelContext(self._searchspace)
        batch_shape = train_x.shape[:-2]

        # Initialize model
        self._initialize_model(train_x, train_y, context, batch_shape)

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
