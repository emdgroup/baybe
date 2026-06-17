"""Gaussian process surrogates."""

from __future__ import annotations

import gc
import importlib
import os
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

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
    def _make_outcome_transform(context: _ModelContext) -> Standardize:
        """Create the outcome transform for the Gaussian process."""
        from botorch.models.transforms.outcome import Standardize

        train_y = to_tensor(
            context.objective._pre_transform(context.measurements, allow_extra=True)
        )
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)
        transform = Standardize(m=train_y.shape[-1])
        transform(train_y)  # fits means/stdvs; GP will re-fit in train mode
        return transform

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
        *,
        anchors: Literal["pretrained", "new", "combined"] = "pretrained",
        mean_kernel_init: Literal["freeze", "warmstart", "discard"] = "freeze",
    ) -> GPyTorchMean:
        """Return a GPyTorch mean module representing the surrogate's posterior mean.

        The bound method satisfies
        :class:`~baybe.surrogates.gaussian_process.components.mean.MeanFactoryProtocol`
        and can be passed directly to a new :class:`GaussianProcessSurrogate`.

        The returned mean module wraps an *inner* GP whose anchor inputs/targets are
        selected by ``anchors`` and whose kernel/mean/likelihood hyperparameters are
        controlled by ``mean_kernel_init``. ``mean_kernel_init`` affects only the
        inner kernel inside the prior-mean module — the new GP's outer kernel is
        always learned from default initialization.

        Args:
            searchspace: The search space of the new GP being fitted.
            objective: The objective of the new GP being fitted.
            measurements: The training data of the new GP being fitted.
            anchors: Which inputs/targets to use as anchors for the inner GP.

                * ``"pretrained"``: the pretrained GP's training data (current default).
                * ``"new"``: the new GP's training data (the pretrained GP's targets
                  are not used; only its kernel structure is reused).
                * ``"combined"``: the concatenation of both. Beware that overlapping
                  or near-duplicate inputs across the two datasets make the inner
                  kernel matrix ``K_XX + sigma^2 I`` ill-conditioned (and, at very low
                  likelihood noise, effectively singular), which can destabilize the
                  inner posterior solve. Prefer non-overlapping anchor data or a
                  non-negligible likelihood noise when using this option.
            mean_kernel_init: How to initialize the inner kernel/likelihood/mean.

                * ``"freeze"``: deepcopy the pretrained components and freeze them
                  (the prior mean is a fixed function of the inputs).
                * ``"warmstart"``: deepcopy the pretrained components but leave them
                  trainable so the outer MLL can adjust them.
                * ``"discard"``: build fresh components via the surrogate's configured
                  factories (no pretrained hyperparameters are transferred).

        Returns:
            The posterior mean module.

        Raises:
            ModelNotTrainedError: If the surrogate has not been fitted yet.
            ValueError: If ``anchors="new"`` is combined with
                ``mean_kernel_init="discard"`` (no information from the pretrained GP
                would be transferred).
        """
        if self._model is None:
            raise ModelNotTrainedError(
                f"'{self.__class__.__name__}' must be fitted before its "
                f"'{self.posterior_mean_function.__name__}' can be used as a "
                f"mean function."
            )

        if anchors == "new" and mean_kernel_init == "discard":
            raise ValueError(
                f"The combination 'anchors=\"new\"' and "
                f"'mean_kernel_init=\"discard\"' would not transfer any information "
                f"from the pretrained '{self.__class__.__name__}'."
            )

        # Dangerous combinations: when the inner anchors include the *new* GP's
        # training targets (``anchors in {"new", "combined"}``) *and* the inner
        # hyperparameters are trainable (``warmstart``), the same target values drive
        # both the inner prior mean and the outer likelihood. The outer MLL can then
        # explain the data almost entirely through the (now flexible) prior mean,
        # pushing the outer residual -- and hence the outer noise -- toward zero, i.e.
        # overfitting. The even more degenerate ``new``+``discard`` case transfers no
        # pretrained information at all and is rejected above; here we only warn
        # because ``combined`` still carries pretrained anchor information.
        if mean_kernel_init == "warmstart" and anchors in ("new", "combined"):
            warnings.warn(
                f"Using 'mean_kernel_init=\"warmstart\"' with 'anchors=\"{anchors}\"' "
                f"causes the new GP's training targets to influence both the inner "
                f"prior mean and the outer likelihood; the marginal log-likelihood "
                f"optimizer may collapse the outer noise toward zero.",
                UserWarning,
                stacklevel=2,
            )

        context = _ModelContext(searchspace, objective, measurements)
        new_input_transform = self._make_input_transform(context)
        new_input_transform.eval()
        new_outcome_transform = self._make_outcome_transform(context)
        new_outcome_transform.eval()

        anchor_x_raw, anchor_y_raw = _resolve_anchors(anchors, self._model, context)
        inner_gp = _build_mean_transfer_gp(
            anchor_x_raw,
            anchor_y_raw,
            mean_kernel_init=mean_kernel_init,
            pretrained_model=self._model,
            mean_factory=self.mean_factory,
            kernel_factory=self.kernel_factory,
            likelihood_factory=self.likelihood_factory,
            context=context,
        )

        return _build_posterior_mean_module(
            inner_gp=inner_gp,
            new_input_transform=new_input_transform,
            new_outcome_transform=new_outcome_transform,
            trainable=mean_kernel_init != "freeze",
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
        outcome_transform = self._make_outcome_transform(context)

        ### Mean
        mean = self.mean_factory(
            context.searchspace, context.objective, context.measurements
        )
        # A posterior-mean module transferred from another GP bakes in the input/
        # output transforms of the context it was built for. Fitting the new GP on a
        # different context would silently evaluate the transferred mean at the wrong
        # physical inputs/outputs, so we verify the contexts agree.
        _validate_transferred_mean_context(mean, input_transform, outcome_transform)

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


def _extract_raw_training_data(
    model: Any,
) -> tuple[Tensor, Tensor]:
    """Return a fitted GP's training data in raw (un-transformed) space.

    Args:
        model: A fitted BoTorch ``SingleTaskGP`` whose ``train_inputs`` /
            ``train_targets`` are read. Typed as ``Any`` because gpytorch's
            registered-buffer attributes are not statically resolvable.

    Returns:
        A tuple ``(X_raw, Y_raw)`` of detached, cloned tensors with shapes
        ``(N, d)`` and ``(N, 1)`` respectively.
    """
    x_norm = model.train_inputs[0]
    y_std = model.train_targets
    if y_std.ndim == 1:
        y_std = y_std.unsqueeze(-1)
    x_raw = model.input_transform.untransform(x_norm).detach().clone()
    y_raw, _ = model.outcome_transform.untransform(y_std)
    return x_raw, y_raw.detach().clone()


def _resolve_anchors(
    anchors: Literal["pretrained", "new", "combined"],
    pretrained_model: Any,
    context: _ModelContext,
) -> tuple[Tensor, Tensor]:
    """Resolve anchor inputs/targets for the inner GP in raw space.

    Args:
        anchors: Which dataset(s) the inner GP should be conditioned on.
        pretrained_model: The fitted pretrained GP whose data may be reused.
        context: Bundles the new GP's search space, objective and measurements.

    Returns:
        A tuple ``(X_raw, Y_raw)`` of the selected anchor inputs/targets in raw
        space.
    """
    import torch

    if anchors == "pretrained":
        return _extract_raw_training_data(pretrained_model)

    x_new_raw = to_tensor(
        context.searchspace.transform(context.measurements, allow_extra=True)
    )
    y_new_raw = to_tensor(
        context.objective._pre_transform(context.measurements, allow_extra=True)
    )
    if y_new_raw.ndim == 1:
        y_new_raw = y_new_raw.unsqueeze(-1)

    if anchors == "new":
        return x_new_raw, y_new_raw

    x_old_raw, y_old_raw = _extract_raw_training_data(pretrained_model)
    return (
        torch.cat([x_old_raw, x_new_raw], dim=0),
        torch.cat([y_old_raw, y_new_raw], dim=0),
    )


def _build_mean_transfer_gp(
    anchor_x_raw: Tensor,
    anchor_y_raw: Tensor,
    *,
    mean_kernel_init: Literal["freeze", "warmstart", "discard"],
    pretrained_model: Any,
    mean_factory: MeanFactoryProtocol,
    kernel_factory: KernelFactoryProtocol,
    likelihood_factory: LikelihoodFactoryProtocol,
    context: _ModelContext,
) -> GPyTorchModel:
    """Construct the inner GP wrapped by the posterior-mean module.

    Args:
        anchor_x_raw: Anchor inputs in raw parameter space.
        anchor_y_raw: Anchor targets in raw output space.
        mean_kernel_init: How to initialize the inner mean/kernel/likelihood. See
            :meth:`GaussianProcessSurrogate.posterior_mean_function` for details.
        pretrained_model: The fitted pretrained GP whose transforms and (for
            ``freeze``/``warmstart``) hyperparameters are reused.
        mean_factory: Factory used to build a fresh mean module for ``discard``.
        kernel_factory: Factory used to build a fresh kernel for ``discard``.
        likelihood_factory: Factory used to build a fresh likelihood for ``discard``.
        context: Bundles the new GP's search space, objective and measurements;
            passed to the factories for ``discard``.

    Returns:
        A BoTorch ``SingleTaskGP`` in eval mode with the appropriate parameters
        frozen when ``mean_kernel_init='freeze'``.
    """
    from copy import deepcopy

    import botorch

    # The pretrained GP's transforms define the space the kernel/likelihood
    # hyperparameters were learned in. Reusing them keeps inner predictions
    # consistent with the pretrained scales; the new GP's transforms convert
    # to/from the new GP's standardized output space inside `forward`.
    input_transform = deepcopy(pretrained_model.input_transform)
    input_transform.eval()
    outcome_transform = deepcopy(pretrained_model.outcome_transform)
    outcome_transform.eval()

    # TODO: Decide whether the inner outcome transform should be refit on the anchor
    #   data instead of reusing the pretrained one. For ``discard`` (fresh inner
    #   hyperparameters) -- and arguably ``warmstart``, where the inner
    #   hyperparameters also change during outer optimization -- the inner kernel ends
    #   up living in a space scaled by the *pretrained* standardization statistics,
    #   which is a mild inconsistency. Reusing the pretrained transform keeps
    #   ``freeze``/``pretrained`` numerically exact; refitting would be more
    #   principled for ``discard``. The right trade-off (and whether to also refit for
    #   ``warmstart``) is currently undecided.

    if mean_kernel_init == "discard":
        mean = mean_factory(
            context.searchspace, context.objective, context.measurements
        )
        kernel = kernel_factory(
            context.searchspace, context.objective, context.measurements
        )
        if isinstance(kernel, Kernel):
            kernel = kernel.to_gpytorch(searchspace=context.searchspace)
        likelihood = likelihood_factory(
            context.searchspace, context.objective, context.measurements
        )
    else:
        # We transfer all three modules because the posterior mean depends on
        # mean + kernel + likelihood noise, not only on the mean module.
        mean = deepcopy(pretrained_model.mean_module)
        kernel = deepcopy(pretrained_model.covar_module)
        likelihood = deepcopy(pretrained_model.likelihood)

    inner_gp = botorch.models.SingleTaskGP(
        anchor_x_raw,
        anchor_y_raw,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
    )

    if mean_kernel_init == "freeze":
        for param in inner_gp.parameters():
            param.requires_grad = False
    inner_gp.eval()
    return inner_gp


def _build_posterior_mean_module(
    inner_gp: GPyTorchModel,
    new_input_transform: Normalize,
    new_outcome_transform: Standardize,
    trainable: bool,
) -> GPyTorchMean:
    """Build a :class:`gpytorch.means.Mean` wrapping the given inner GP.

    Args:
        inner_gp: The GP whose posterior mean is used as the prior mean. Already
            constructed with the desired anchors, hyperparameters, and transforms.
        new_input_transform: The new GP's input transform; used to un-normalize the
            inputs ``x`` arriving at ``forward`` so the inner GP sees raw inputs.
        new_outcome_transform: The new GP's outcome transform; used to standardize
            the inner GP's raw-space predictions into the new GP's output space.
        trainable: If ``True``, the inner GP's cached prediction strategy is
            invalidated on every forward call and gradients flow through
            ``(K_XX + sigma^2 I)^{-1} y`` for joint optimization with the outer MLL.

    Returns:
        A mean module suitable for use as ``mean_module`` of the new GP.
    """
    import gpytorch

    class _PosteriorMean(gpytorch.means.Mean):
        """GPyTorch mean wrapping a GP's posterior."""

        def __init__(self) -> None:
            super().__init__()
            self.gp = inner_gp
            # Stored as a plain tuple (not as direct module attributes) so that
            # ``torch.nn.Module`` does not register the transforms as submodules of
            # this mean. They are read only by ``_validate_transferred_mean_context``
            # to check that the new GP is fitted on the context they were built for.
            self._transferred_context = (new_input_transform, new_outcome_transform)

        @override
        def train(self, mode: bool = True) -> _PosteriorMean:
            """Set training mode without propagating to children.

            The inner GP stays in eval so ``posterior(x)`` returns predictive
            (not training) outputs. Trainable parameters keep ``requires_grad``
            and the outer MLL still updates them via the autograd graph.
            """
            # Keep this wrapper's flag in sync with the outer model while
            # intentionally not toggling child module modes.
            #
            # Why: we always use the inner GP as a posterior provider, but we
            # may still optimize its parameters via autograd in warmstart/
            # discard modes.
            self.training = mode
            return self

        @override
        def forward(self, x: Tensor) -> Tensor:
            """Compute the prior mean in the new GP's standardized output space."""
            x_raw = new_input_transform.untransform(x)
            if trainable:
                # The pretrained cache holds (K_XX + sigma^2 I)^{-1} y for the
                # values the hyperparameters had at the last evaluation. Discard
                # it so the next call recomputes against the current values and
                # keep its solve attached to the autograd graph.
                self.gp.prediction_strategy = None  # type: ignore[assignment]  # gpytorch's typed attribute is intentionally cleared to force recomputation
                # Keep test-time caches attached so gradients can flow into
                # inner GP parameters during outer MLL optimization.
                with gpytorch.settings.detach_test_caches(False):
                    posterior_mean = self.gp.posterior(x_raw).mean
            else:
                # Frozen inner GP: only the posterior mean is consumed, so neither
                # gradient-through-inner machinery nor predictive-variance
                # acceleration (``fast_pred_var`` only speeds up *variances*) is
                # needed here.
                posterior_mean = self.gp.posterior(x_raw).mean
            standardized, _ = new_outcome_transform(posterior_mean)
            return standardized.squeeze(-1)

    return _PosteriorMean()


def _validate_transferred_mean_context(
    mean: Any,
    input_transform: Normalize,
    outcome_transform: Standardize,
) -> None:
    """Validate a transferred posterior-mean module against the new GP's context.

    A posterior-mean module created by
    :meth:`GaussianProcessSurrogate.posterior_mean_function` bakes in the input/output
    transforms of the context it was built for. When such a module is reused as the
    mean of a new GP, the new GP must be fitted on a matching context; otherwise the
    transforms disagree and the transferred prior mean is evaluated at the wrong
    physical inputs/outputs.

    Args:
        mean: The mean module of the new GP. Only modules produced by
            ``posterior_mean_function`` carry the context marker; anything else is
            ignored. Typed as ``Any`` because the marker attribute is not part of the
            static ``Mean`` interface.
        input_transform: The new GP's input transform.
        outcome_transform: The new GP's (fitted) outcome transform.

    Raises:
        ValueError: If the transferred module's transforms do not match the new GP's.
    """
    import torch

    transferred = getattr(mean, "_transferred_context", None)
    if transferred is None:
        return

    inner_input_transform, inner_outcome_transform = transferred
    if (
        torch.equal(inner_input_transform.coefficient, input_transform.coefficient)
        and torch.equal(inner_input_transform.offset, input_transform.offset)
        and torch.equal(inner_outcome_transform.means, outcome_transform.means)
        and torch.equal(inner_outcome_transform.stdvs, outcome_transform.stdvs)
    ):
        return

    raise ValueError(
        "The transferred posterior-mean module was created for a different "
        "search space / objective / measurements context than the one the new "
        "Gaussian process is being fitted on. Rebuild the mean function via "
        "'posterior_mean_function' using the same context that you pass to 'fit'."
    )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
