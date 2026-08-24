"""Resolution of the transfer-learning kernel override."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import assert_never

from baybe.kernels.base import Kernel
from baybe.parameters.enum import TransferLearningMode

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel

    from baybe.surrogates.gaussian_process.core import _ModelContext


def extract_transfer_learning_overrides(
    context: _ModelContext,
) -> list[tuple[str, GPyTorchKernel]]:
    """Extract the transfer-learning kernel override, if any.

    Args:
        context: The model context providing the task parameter and override mode.

    Returns:
        A single ``(task_name, kernel)`` pair if an override is set, else an empty
        list.
    """
    task_param = context.searchspace._task_parameter
    if task_param is None or context.tl_override is None:
        return []
    return [(task_param.name, make_transfer_learning_override_kernel(context))]


def make_transfer_learning_override_kernel(
    context: _ModelContext,
) -> GPyTorchKernel:
    """Create the task kernel factor requested by a transfer-learning override.

    Args:
        context: The model context providing the task parameter and override mode.

    Returns:
        The GPyTorch task kernel for the requested transfer-learning mode.
    """
    from baybe.kernels.basic import IndexKernel, PositiveIndexKernel

    task_param = context.searchspace._task_parameter
    override = context.tl_override
    assert task_param is not None and override is not None
    n_tasks, names = context.n_tasks, (task_param.name,)
    match override:
        case TransferLearningMode.POSITIVE_INDEX_KERNEL:
            spec: Kernel = PositiveIndexKernel(
                num_tasks=n_tasks, rank=n_tasks, parameter_names=names
            )
        case TransferLearningMode.INDEX_KERNEL:
            spec = IndexKernel(num_tasks=n_tasks, rank=n_tasks, parameter_names=names)
        case _:
            assert_never(override)
    return spec.to_gpytorch(context.searchspace)
