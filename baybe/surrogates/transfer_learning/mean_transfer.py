"""Mean-transfer Gaussian process surrogate for transfer learning."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar

from attrs import define, evolve, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate

if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import Posterior
    from torch import Tensor

    from baybe.parameters.base import Parameter


@define
class MeanTransferSurrogate(Surrogate):
    """A transfer learning surrogate that transfers a source model's posterior mean.

    The surrogate splits the training data by task into a single source and a single
    target subset. A single-task Gaussian process is fitted on the source subset over
    the reduced (task-free) search space. Its posterior mean is then used as the prior
    mean of a second single-task Gaussian process fitted on the target subset, thereby
    transferring the source knowledge to the target.

    Predictions are made for target points: the task column is stripped from the
    incoming candidates so that they match the reduced space of the target model, whose
    posterior is then returned.

    Note:
        Only a single source and a single target task are currently supported.
    """

    supports_transfer_learning: ClassVar[bool] = True
    # See base class.

    base_surrogate: GaussianProcessSurrogate = field(
        factory=GaussianProcessSurrogate,
        validator=instance_of(GaussianProcessSurrogate),
    )
    """The Gaussian process configuration used for both the source and target models."""

    _source_gp: GaussianProcessSurrogate | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The single-task GP trained on the source data. Available after fitting."""

    _target_gp: GaussianProcessSurrogate | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The single-task GP trained on the target data. Available after fitting."""

    _numerical_indices: list[int] | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The comp-rep column indices of the non-task inputs. Available after fitting."""

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Split the data by task and train source and target single-task GPs.

        The measurements are partitioned by task parameter value into a source and a
        target subset. A single-task source GP is trained on the source subset over the
        reduced (task-free) search space. A single-task target GP is then trained on the
        target subset, using the source GP's posterior mean as its prior mean.

        Args:
            train_x: Computational-representation inputs prepared by the base class.
                Not used directly, since the inner GPs are refitted from the stored
                measurements over the reduced search space.
            train_y: Target values prepared by the base class. Not used directly, for
                the same reason as ``train_x``.

        Raises:
            IncompatibleSearchSpaceError: If the search space has no task parameter,
                does not describe exactly one source and one target task, or lacks
                measurements for the source or the target task.
        """
        assert self._searchspace is not None  # provided by base class
        assert self._objective is not None  # provided by base class
        assert self._measurements is not None  # provided by base class

        searchspace = self._searchspace
        objective = self._objective
        measurements = self._measurements

        task_param = searchspace._task_parameter
        if task_param is None:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires a search space that contains a "
                f"task parameter."
            )

        active_values = set(task_param.active_values)
        source_values = set(task_param.values) - active_values
        if len(active_values) != 1 or len(source_values) != 1:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' currently supports exactly one source "
                f"and one target task, but the task parameter describes "
                f"{len(source_values)} source and {len(active_values)} target value(s)."
            )
        (target_value,) = active_values
        (source_value,) = source_values

        task_name = task_param.name
        source_measurements = measurements[measurements[task_name] == source_value]
        target_measurements = measurements[measurements[task_name] == target_value]
        if source_measurements.empty or target_measurements.empty:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires measurements for both the "
                f"source and the target task, but received "
                f"{len(source_measurements)} source and "
                f"{len(target_measurements)} target measurement(s)."
            )

        reduced_searchspace = searchspace._drop_parameters({task_name})

        self._source_gp = evolve(self.base_surrogate)
        self._source_gp.fit(reduced_searchspace, objective, source_measurements)

        self._target_gp = evolve(
            self.base_surrogate,
            mean_or_factory=self._source_gp.posterior_mean_function,
        )
        self._target_gp.fit(reduced_searchspace, objective, target_measurements)

        self._numerical_indices = [
            i
            for i in range(len(searchspace.comp_rep_columns))
            if i != searchspace.task_idx
        ]

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        """Disable base-class input scaling; the inner GPs handle scaling themselves.

        Args:
            parameter: The parameter for which a scaler would otherwise be created.

        Returns:
            Always ``None`` to signal that no base-class input scaling is applied.
        """
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        """Disable base-class output scaling; the inner GPs handle scaling themselves.

        Returns:
            Always ``None`` to signal that no base-class output scaling is applied.
        """
        return None

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Compute the posterior by stripping the task column and querying the target GP.

        The incoming candidates are in the computational representation of the full
        (task-aware) search space. The task column is removed so that the resulting
        tensor matches the reduced single-task space on which the target GP was trained.

        Args:
            candidates_comp_scaled: Candidate points in the computational representation
                of the full search space (including the task column).

        Returns:
            The posterior of the target Gaussian process at the given candidates.
        """
        import torch

        assert self._numerical_indices is not None  # set during fitting
        assert self._target_gp is not None  # set during fitting

        indices = torch.tensor(self._numerical_indices, dtype=torch.long)
        reduced_candidates = candidates_comp_scaled.index_select(-1, indices)
        return self._target_gp._posterior(reduced_candidates)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
