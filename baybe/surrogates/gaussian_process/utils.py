"""Gaussian process utilities."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from attrs import define, field
from attrs.validators import instance_of

from baybe.searchspace.core import SearchSpace

if TYPE_CHECKING:
    from torch import Tensor


@define
class _ModelContext:
    """Model context for Gaussian process surrogates."""

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
    def n_fidelity_dimensions(self) -> int:
        """The number of fidelity dimensions."""
        # TODO: Generalize to multiple fidelity parameters
        return 1 if self.searchspace.fidelity_idx is not None else 0

    @property
    def fidelity_idx(self) -> int | None:
        """The computational column index of the fidelity parameter, if available."""
        return self.searchspace.fidelity_idx

    @property
    def n_fidelities(self) -> int:
        """The number of fidelities."""
        return self.searchspace.n_fidelities

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
            if i not in (self.task_idx, self.fidelity_idx)
        ]


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
