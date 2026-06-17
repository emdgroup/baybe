"""Mean-transfer transfer-learning surrogate."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar, Literal

from attrs import define, field
from attrs.validators import in_
from typing_extensions import override

from baybe.exceptions import IncompatibleSurrogateError
from baybe.parameters.categorical import TaskParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.conversion import to_string

if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import Posterior
    from torch import Tensor

    from baybe.parameters.base import Parameter

_ANCHOR_OPTIONS = ("pretrained", "new", "combined")
"""Allowed values for the anchor selection."""

_MEAN_KERNEL_INIT_OPTIONS = ("freeze", "warmstart", "discard")
"""Allowed values for the inner mean/kernel initialization."""


@define
class MeanTransferSurrogate(Surrogate):
    """A transfer-learning surrogate that transfers a source GP's posterior mean.

    The surrogate operates on a search space that contains a single
    :class:`~baybe.parameters.categorical.TaskParameter` with exactly one active
    (target) value. It splits the training data into source and target data, fits a
    :class:`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate` on the
    source data in a task-free search space, and uses that model's posterior mean as
    the prior mean of a second Gaussian process trained on the target data. Predictions
    are served by the target model after stripping the task column from the inputs.

    The two degenerate data regimes are handled gracefully:

    * **No source data:** There is nothing to transfer, so the surrogate behaves like a
      plain :class:`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate`
      trained on the target data only.
    * **No target data:** There is nothing to condition on, so the surrogate behaves
      like a Gaussian process whose prior mean equals the source model's posterior
      mean (i.e., the source model is used directly for predictions).
    """

    supports_transfer_learning: ClassVar[bool] = True
    # See base class.

    anchors: Literal["pretrained", "new", "combined"] = field(
        default="pretrained", validator=in_(_ANCHOR_OPTIONS)
    )
    """The anchor selection forwarded to
    :meth:`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate.posterior_mean_function`."""

    mean_kernel_init: Literal["freeze", "warmstart", "discard"] = field(
        default="freeze", validator=in_(_MEAN_KERNEL_INIT_OPTIONS)
    )
    """The inner mean/kernel initialization forwarded to
    :meth:`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate.posterior_mean_function`."""

    # TODO: type omitted analogous to `GaussianProcessSurrogate._model` due to:
    #   https://github.com/python-attrs/cattrs/issues/531
    _target_model = field(init=False, default=None, eq=False)
    """The target Gaussian process operating on the task-free search space."""

    _keep_indices: tuple[int, ...] | None = field(init=False, default=None, eq=False)
    """The computational column indices retained when stripping the task column."""

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        # The inner Gaussian processes handle their own scaling. See the
        # [Scaling Workaround] note in `GaussianProcessSurrogate`.
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # The inner Gaussian processes handle their own scaling. See the
        # [Scaling Workaround] note in `GaussianProcessSurrogate`.
        return None

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        import torch

        assert self._target_model is not None
        assert self._keep_indices is not None
        index = torch.tensor(self._keep_indices, device=candidates_comp_scaled.device)
        candidates_without_task = candidates_comp_scaled.index_select(-1, index)
        return self._target_model.to_botorch().posterior(candidates_without_task)

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        assert self._searchspace is not None  # provided by base class
        assert self._objective is not None  # provided by base class
        assert self._measurements is not None  # provided by base class
        searchspace = self._searchspace
        objective = self._objective
        measurements = self._measurements

        task_parameter = self._get_task_parameter(searchspace)
        target_task = task_parameter.active_values[0]
        task_name = task_parameter.name

        target_mask = measurements[task_name] == target_task
        target_data = measurements[target_mask].drop(columns=[task_name])
        source_data = measurements[~target_mask].drop(columns=[task_name])

        task_less_searchspace = SearchSpace.from_product(
            parameters=[
                p for p in searchspace.parameters if not isinstance(p, TaskParameter)
            ]
        )
        self._validate_column_order(searchspace, task_less_searchspace)

        task_idx = searchspace.task_idx
        self._keep_indices = tuple(
            i for i in range(len(searchspace.comp_rep_columns)) if i != task_idx
        )

        # Without source data there is nothing to transfer, so we degrade to a plain
        # Gaussian process trained on the target data only.
        if len(source_data) == 0:
            target_model = GaussianProcessSurrogate()
            target_model.fit(task_less_searchspace, objective, target_data)
            self._target_model = target_model
            return

        source_tasks = measurements.loc[~target_mask, task_name].unique()
        if len(source_tasks) != 1:
            raise IncompatibleSurrogateError(
                f"'{self.__class__.__name__}' requires exactly one source task, but "
                f"found {len(source_tasks)} ({sorted(source_tasks)}). Restrict the "
                f"measurements to a single source task."
            )

        source_model = GaussianProcessSurrogate()
        source_model.fit(task_less_searchspace, objective, source_data)

        # Without target data there is nothing to condition on, so the surrogate acts
        # as a Gaussian process whose (prior) mean equals the source model's posterior
        # mean. The source model itself already represents exactly this object, so we
        # use it directly as the prediction model.
        if len(target_data) == 0:
            self._target_model = source_model
            return

        mean = source_model.posterior_mean_function(
            task_less_searchspace,
            objective,
            target_data,
            anchors=self.anchors,
            mean_kernel_init=self.mean_kernel_init,
        )

        target_model = GaussianProcessSurrogate(mean_or_factory=mean)
        target_model.fit(task_less_searchspace, objective, target_data)
        self._target_model = target_model

    def _get_task_parameter(self, searchspace: SearchSpace) -> TaskParameter:
        """Return the single active-value task parameter of the search space.

        Args:
            searchspace: The search space the surrogate is fitted on.

        Returns:
            The task parameter with a single active value.

        Raises:
            IncompatibleSurrogateError: If the search space does not contain exactly
                one task parameter, or if that parameter does not have exactly one
                active value.
        """
        task_parameters = [
            p for p in searchspace.parameters if isinstance(p, TaskParameter)
        ]
        if len(task_parameters) != 1:
            raise IncompatibleSurrogateError(
                f"'{self.__class__.__name__}' requires a search space with exactly "
                f"one '{TaskParameter.__name__}', but found {len(task_parameters)}."
            )
        task_parameter = task_parameters[0]
        if len(task_parameter.active_values) != 1:
            raise IncompatibleSurrogateError(
                f"'{self.__class__.__name__}' requires the '{TaskParameter.__name__}' "
                f"to have exactly one active value, but "
                f"'{task_parameter.name}' has {len(task_parameter.active_values)} "
                f"({list(task_parameter.active_values)})."
            )
        return task_parameter

    def _validate_column_order(
        self, searchspace: SearchSpace, task_less_searchspace: SearchSpace
    ) -> None:
        """Validate that dropping the task column matches the task-free layout.

        Args:
            searchspace: The original (task-bearing) search space.
            task_less_searchspace: The reconstructed search space without the task
                parameter.

        Raises:
            IncompatibleSurrogateError: If the original computational columns with the
                task column removed do not match the task-free columns, which would
                make the task-column stripping during prediction incorrect.
        """
        task_idx = searchspace.task_idx
        expected = tuple(
            col for i, col in enumerate(searchspace.comp_rep_columns) if i != task_idx
        )
        if expected != task_less_searchspace.comp_rep_columns:
            raise IncompatibleSurrogateError(
                f"'{self.__class__.__name__}' cannot map the task-bearing search "
                f"space onto its task-free counterpart because the computational "
                f"column layouts are incompatible: {expected} vs. "
                f"{task_less_searchspace.comp_rep_columns}."
            )

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Anchors", self.anchors, single_line=True),
            to_string("Mean kernel init", self.mean_kernel_init, single_line=True),
        ]
        return to_string(super().__str__(), *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
