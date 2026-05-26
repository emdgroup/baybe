"""Tests for kernel factories."""

from contextlib import nullcontext

import pytest
import torch
from pytest import param

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.kernels.basic import IndexKernel, PositiveIndexKernel
from baybe.parameters.categorical import (
    CategoricalParameter,
    TaskParameter,
    TransferLearningMode,
)
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.presets.baybe import (
    BayBEKernelFactory,
    BayBENumericalKernelFactory,
    BayBETaskKernelFactory,
)

# A selector that accepts all parameters
_SELECT_ALL = lambda parameter: True  # noqa: E731


@pytest.mark.parametrize(
    ("factory", "parameters", "error"),
    [
        param(
            BayBENumericalKernelFactory(parameter_selector=_SELECT_ALL),
            [TaskParameter("task", ["t1", "t2"])],
            IncompatibleSearchSpaceError,
            id="regular_rejects_task",
        ),
        param(
            BayBETaskKernelFactory(parameter_selector=_SELECT_ALL),
            [CategoricalParameter("cat", ["a", "b"])],
            IncompatibleSearchSpaceError,
            id="task_rejects_categorical",
        ),
        param(
            BayBETaskKernelFactory(parameter_selector=_SELECT_ALL),
            [NumericalDiscreteParameter("num", [1, 2, 3])],
            IncompatibleSearchSpaceError,
            id="task_rejects_numerical_discrete",
        ),
        param(
            BayBETaskKernelFactory(parameter_selector=_SELECT_ALL),
            [NumericalContinuousParameter("cont", (0, 1))],
            IncompatibleSearchSpaceError,
            id="task_rejects_numerical_continuous",
        ),
        param(
            BayBEKernelFactory(),
            [
                NumericalContinuousParameter("cont", (0, 1)),
                TaskParameter("task", ["t1", "t2"]),
            ],
            None,
            id="combined_accepts_both",
        ),
    ],
)
def test_factory_parameter_kind_validation(factory, parameters, error):
    """Factories reject unsupported parameter kinds and accept supported ones."""
    ss = SearchSpace.from_product(parameters)
    train_x = torch.zeros(2, len(ss.comp_rep_columns))
    train_y = torch.zeros(2, 1)

    with (
        nullcontext()
        if error is None
        else pytest.raises(error, match="does not support")
    ):
        factory(ss, train_x, train_y)


@pytest.mark.parametrize(
    ("transfer_learning_mode", "expected_kernel_type"),
    [
        param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            PositiveIndexKernel,
            id="positive_index_kernel",
        ),
        param(
            TransferLearningMode.INDEX_KERNEL,
            IndexKernel,
            id="index_kernel",
        ),
    ],
)
def test_task_kernel_factory_dispatching(transfer_learning_mode, expected_kernel_type):
    """BayBETaskKernelFactory dispatches to the correct kernel type."""
    parameters = [
        NumericalDiscreteParameter(name="x", values=[1.0, 2.0, 3.0]),
        TaskParameter(
            name="task",
            values=["source", "target"],
            active_values=["target"],
            transfer_learning_mode=transfer_learning_mode,
        ),
    ]
    ss = SearchSpace.from_product(parameters)
    train_x = torch.zeros(2, len(ss.comp_rep_columns))
    train_y = torch.zeros(2, 1)

    kernel = BayBETaskKernelFactory()(ss, train_x, train_y)
    assert isinstance(kernel, expected_kernel_type)
