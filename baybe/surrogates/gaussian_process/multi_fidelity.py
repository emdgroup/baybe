"""Multi-fidelity Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from typing_extensions import override

from baybe.parameters.base import Parameter
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.core import (
    _ModelContext,
)
from baybe.surrogates.gaussian_process.kernel_factory import (
    DiscreteFidelityKernelFactory,
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
from baybe.surrogates.gaussian_process.presets.fidelity import (
    DefaultFidelityKernelFactory,
)
from baybe.utils.conversion import to_string

if TYPE_CHECKING:
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import Posterior
    from torch import Tensor


@define
class MultiFidelityGaussianProcessSurrogate(Surrogate):
    """Multi fidelity Gaussian process with customisable kernel."""

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    supports_multi_fidelity: ClassVar[bool] = True
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

    fidelity_kernel_factory: DiscreteFidelityKernelFactory = field(
        alias="fidelity_kernel_or_factory",
        factory=DefaultFidelityKernelFactory,
        converter=to_kernel_factory,
    )
    """The factory used to create the fidelity kernel of the Gaussian process.
    Accepts either a :class:`baybe.kernels.base.Kernel` or a
    :class:`.kernel_factory.KernelFactory`.
    When passing a :class:`baybe.kernels.base.Kernel`, it gets automatically wrapped
    into a :class:`.kernel_factory.PlainKernelFactory`."""

    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @staticmethod
    def from_preset(preset: GaussianProcessPreset) -> Surrogate:
        """Create a Gaussian process surrogate from one of the defined presets."""
        return make_gp_from_preset(preset)

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
        import torch
        from botorch.models.transforms import Normalize, Standardize

        # FIXME[typing]: It seems there is currently no better way to inform the type
        #   checker that the attribute is available at the time of the function call
        assert self._searchspace is not None

        context = _ModelContext(self._searchspace)

        numerical_idxs = context.get_numerical_indices(train_x.shape[-1])

        numerical_design_idxs = tuple(
            idx for idx in numerical_idxs if idx != context.fidelity_idx
        )

        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
        input_transform = Normalize(
            train_x.shape[-1],
            bounds=context.parameter_bounds,
            indices=list(numerical_idxs),
        )
        outcome_transform = Standardize(train_y.shape[-1])

        # extract the batch shape of the training data
        batch_shape = train_x.shape[:-2]

        # create GP mean
        mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
        input_transform = botorch.models.transforms.Normalize(
            train_x.shape[-1],
            bounds=context.parameter_bounds,
            indices=list(numerical_design_idxs),
        )
        outcome_transform = botorch.models.transforms.Standardize(train_y.shape[-1])

        base_covar_module = self.kernel_factory(
            context.searchspace, train_x, train_y
        ).to_gpytorch(
            ard_num_dims=train_x.shape[-1] - context.n_fidelity_dimensions,
            active_dims=numerical_design_idxs,
            batch_shape=batch_shape,
        )

        fidelity_covar_module = self.fidelity_kernel_factory(
            searchspace=self._searchspace
        ).to_gpytorch(
            ard_num_dims=1,
            active_dims=None
            if context.fidelity_idx is None
            else (context.fidelity_idx,),
            batch_shape=batch_shape,
        )

        covar_module = base_covar_module * fidelity_covar_module

        # create GP likelihood
        noise_prior = _default_noise_factory(context.searchspace, train_x, train_y)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior[0].to_gpytorch(), batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # construct and fit the Gaussian process
        self._model = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )

        mll = gpytorch.ExactMarginalLogLikelihood(self._model.likelihood, self._model)

        botorch.fit.fit_gpytorch_mll(mll)

    @override
    def __str__(self) -> str:
        fields = [
            to_string(
                to_string("Kernel factory", self.kernel_factory, single_line=True),
                "Fidelity kernel factory",
                self.fidelity_kernel_factory,
                single_line=True,
            ),
        ]
        return to_string(super().__str__(), *fields)


@define
class GaussianProcessSurrogateSTMF(Surrogate):
    """Botorch's single task multi fidelity Gaussian process."""

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    supports_multi_fidelity: ClassVar[bool] = True
    # See base class.

    kernel_factory: KernelFactory = field(init=False, default=None)
    """Design kernel is set to Matern within SingleTaskMultiFidelityGP."""

    # TODO: type should be Optional[botorch.models.SingleTaskGP] but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @staticmethod
    def from_preset(preset: GaussianProcessPreset) -> Surrogate:
        """Create a Gaussian process surrogate from one of the defined presets."""
        return make_gp_from_preset(preset)

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

        assert self._searchspace is not None

        context = _ModelContext(self._searchspace)

        numerical_design_idxs = context.get_numerical_indices(train_x.shape[-1])

        assert context.is_multi_fidelity, (
            "GaussianProcessSurrogateSTMF can only "
            "be fit on multi fidelity searchspaces."
        )

        if context.is_multi_fidelity:
            numerical_design_idxs = tuple(
                idx for idx in numerical_design_idxs if idx != context.fidelity_idx
            )

        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
        input_transform = botorch.models.transforms.Normalize(
            train_x.shape[-1],
            bounds=context.parameter_bounds,
            indices=list(numerical_design_idxs),
        )
        outcome_transform = botorch.models.transforms.Standardize(train_y.shape[-1])

        # construct and fit the Gaussian process
        self._model = botorch.models.SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            data_fidelities=None
            if context.fidelity_idx is None
            else (context.fidelity_idx,),
        )

        mll = gpytorch.ExactMarginalLogLikelihood(self._model.likelihood, self._model)

        botorch.fit.fit_gpytorch_mll(mll)

    @override
    def __str__(self) -> str:
        return "SingleTaskMultiFidelityGP with Botorch defaults."


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
