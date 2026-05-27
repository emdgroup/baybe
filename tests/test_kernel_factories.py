"""Tests for kernel factories."""

from contextlib import nullcontext

import gpytorch
import pytest
import torch
from pytest import param

from baybe.exceptions import IncompatibleKernelError, IncompatibleSearchSpaceError
from baybe.kernels.basic import IndexKernel, MaternKernel, PositiveIndexKernel
from baybe.parameters.categorical import (
    CategoricalParameter,
    TaskParameter,
)
from baybe.parameters.enum import TransferLearningMode
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.components.kernel import ICMKernelFactory
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


def test_task_kernel_factory_always_returns_positive_index_kernel():
    """BayBETaskKernelFactory always returns PositiveIndexKernel."""
    parameters = [
        NumericalDiscreteParameter(name="x", values=[1.0, 2.0, 3.0]),
        TaskParameter(
            name="task",
            values=["source", "target"],
            active_values=["target"],
        ),
    ]
    ss = SearchSpace.from_product(parameters)
    train_x = torch.zeros(2, len(ss.comp_rep_columns))
    train_y = torch.zeros(2, 1)

    kernel = BayBETaskKernelFactory()(ss, train_x, train_y)
    assert isinstance(kernel, PositiveIndexKernel)


def _make_dispatch_context(override_mode):
    """Build a `_ModelContext` with a numerical + task parameter for dispatch tests."""
    from baybe.surrogates.gaussian_process.core import _ModelContext

    task_param = TaskParameter(
        "Task",
        ["A", "B", "C"],
        active_values=["A"],
        override_transfer_learning_mode=override_mode,
    )
    num_param = NumericalDiscreteParameter("x", [1, 2, 3, 4, 5])
    searchspace = SearchSpace.from_product([num_param, task_param])
    train_x = torch.zeros(2, len(searchspace.comp_rep_columns))
    train_y = torch.zeros(2, 1)
    return _ModelContext(searchspace), train_x, train_y


@pytest.mark.parametrize(
    ("override_mode", "kernel_or_factory", "expected_task_kernel_cls"),
    [
        param(
            None,
            None,
            "PositiveIndexKernel",
            id="no_override+default_factory",
        ),
        param(
            None,
            ICMKernelFactory(
                task_kernel_or_factory=IndexKernel(
                    num_tasks=3, rank=3, parameter_names=("Task",)
                )
            ),
            "IndexKernel",
            id="no_override+custom_icm_index_kernel_escape_hatch",
        ),
        param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            None,
            "PositiveIndexKernel",
            id="positive_index_override+default_factory",
        ),
        param(
            TransferLearningMode.INDEX_KERNEL,
            None,
            "IndexKernel",
            id="index_override+default_factory",
        ),
        param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            MaternKernel(),
            "PositiveIndexKernel",
            id="positive_index_override+bare_baybe_matern",
        ),
        param(
            TransferLearningMode.INDEX_KERNEL,
            gpytorch.kernels.MaternKernel(nu=2.5),
            "IndexKernel",
            id="index_override+bare_gpytorch_matern",
        ),
    ],
)
def test_resolve_kernel_dispatch(
    monkeypatch, override_mode, kernel_or_factory, expected_task_kernel_cls
):
    """`_resolve_kernel` produces the expected task kernel for each dispatch branch."""
    monkeypatch.setenv("BAYBE_DISABLE_CUSTOM_KERNEL_WARNING", "True")

    context, train_x, train_y = _make_dispatch_context(override_mode)
    kwargs = (
        {} if kernel_or_factory is None else {"kernel_or_factory": kernel_or_factory}
    )
    surrogate = GaussianProcessSurrogate(**kwargs)

    kernel = surrogate._resolve_kernel(context, train_x, train_y)

    # The resolved kernel is a product of base * task kernel
    base_kernel, task_kernel = kernel.kernels
    assert type(task_kernel).__name__ == expected_task_kernel_cls

    # Override branches must partition active dims so the task column isn't
    # consumed by both halves of the product kernel.
    if override_mode is not None:
        assert base_kernel.active_dims is not None
        assert set(base_kernel.active_dims.tolist()) == set(context.numerical_indices)


@pytest.mark.parametrize(
    "override_mode",
    [
        param(TransferLearningMode.POSITIVE_INDEX_KERNEL, id="positive_index"),
        param(TransferLearningMode.INDEX_KERNEL, id="index"),
    ],
)
def test_resolve_kernel_overspecification_raises(monkeypatch, override_mode):
    """Override + task-aware kernel factory raises `IncompatibleKernelError`."""
    monkeypatch.setenv("BAYBE_DISABLE_CUSTOM_KERNEL_WARNING", "True")

    context, train_x, train_y = _make_dispatch_context(override_mode)
    surrogate = GaussianProcessSurrogate(
        kernel_or_factory=ICMKernelFactory(
            task_kernel_or_factory=IndexKernel(
                num_tasks=3, rank=3, parameter_names=("Task",)
            )
        )
    )

    with pytest.raises(IncompatibleKernelError, match="overspecification"):
        surrogate._resolve_kernel(context, train_x, train_y)
