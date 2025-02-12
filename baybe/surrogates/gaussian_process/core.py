"""Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.parameters import TaskParameter
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
from baybe.utils.plotting import to_string

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

        return torch.from_numpy(self.searchspace.comp_rep_bounds.values)

    def get_numerical_indices(self, n_inputs: int) -> tuple[int, ...]:
        """Get the indices of the regular numerical model inputs."""
        return tuple(i for i in range(n_inputs) if i != self.task_idx)


@define
class GaussianProcessSurrogate(Surrogate):
    """A Gaussian process surrogate model."""

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

    _task_stratified_outtransform: bool = field(default=False)
    """Should task-stratified output transform be used for multi-task model.

    This is experimental and may be removed before merging to main.
    Also, the StratifiedStandardise would need to be adapted to work
    with multi-output models.
    """

    @staticmethod
    def from_preset(preset: GaussianProcessPreset) -> GaussianProcessSurrogate:
        """Create a Gaussian process surrogate from one of the defined presets."""
        return make_gp_from_preset(preset)

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

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        import botorch
        import gpytorch
        import torch

        # FIXME[typing]: It seems there is currently no better way to inform the type
        #   checker that the attribute is available at the time of the function call
        assert self._searchspace is not None

        context = _ModelContext(self._searchspace)

        numerical_idxs = context.get_numerical_indices(train_x.shape[-1])

        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
        input_transform = botorch.models.transforms.Normalize(
            train_x.shape[-1], bounds=context.parameter_bounds, indices=numerical_idxs
        )

        if context.is_multitask and self._task_stratified_outtransform:
            # TODO See https://github.com/pytorch/botorch/issues/2739
            if train_y.shape[-1] != 1:
                raise NotImplementedError(
                    "Task-stratified output transform currently does not support"
                    + "multiple outputs."
                )
            outcome_transform = botorch.models.transforms.outcome.StratifiedStandardize(
                task_values=train_x[..., context.task_idx].unique().to(torch.long),
                stratification_idx=context.task_idx,
            )
        else:
            outcome_transform = botorch.models.transforms.Standardize(train_y.shape[-1])

        # extract the batch shape of the training data
        batch_shape = train_x.shape[:-2]

        # create GP mean
        mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        # define the covariance module for the numeric dimensions
        base_covar_module = self.kernel_factory(
            context.searchspace, train_x, train_y
        ).to_gpytorch(
            ard_num_dims=train_x.shape[-1] - context.n_task_dimensions,
            batch_shape=batch_shape,
            # The active_dims parameter is omitted as it is not needed for both
            # - single-task SingleTaskGP: all features are used
            # - multi-task MultiTaskGP: the model splits task and non-task features
            #   before passing them to the covariance kernel
        )

        # create GP likelihood
        noise_prior = _default_noise_factory(context.searchspace, train_x, train_y)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior[0].to_gpytorch(), batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # Whether to use multi- or single-task model
        if not context.is_multitask:
            model_cls = botorch.models.SingleTaskGP
            model_kwargs = {}
        else:
            model_cls = botorch.models.MultiTaskGP
            # TODO
            #  It is assumed that there is only one task parameter with only
            #  one active value.
            #  One active task value is required for MultiTaskGP as else
            #  one posterior per task would be returned:
            #  https://github.com/pytorch/botorch/blob/a018a5ffbcbface6229d6c39f7ac6ef9baf5765e/botorch/models/gpytorch.py#L951
            # TODO
            #  The below code implicitly assumes there is single task parameter,
            #  which is already checked in the SearchSpace.
            task_param = [
                p
                for p in context.searchspace.discrete.parameters
                if isinstance(p, TaskParameter)
            ][0]
            if len(task_param.active_values) > 1:
                raise NotImplementedError(
                    "Does not support multiple active task values."
                )
            model_kwargs = {
                "task_feature": context.task_idx,
                "output_tasks": [
                    task_param.comp_df.at[task_param.active_values[0], task_param.name]
                ],
                "rank": context.n_tasks,
                "task_covar_prior": None,
                "all_tasks": task_param.comp_df[task_param.name].astype(int).to_list(),
            }

        # construct and fit the Gaussian process
        self._model = model_cls(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=base_covar_module,
            likelihood=likelihood,
            **model_kwargs,
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
