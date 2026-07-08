"""Tests for kernel factories."""

from contextlib import nullcontext

import gpytorch
import pandas as pd
import pytest
from pytest import param

from baybe.exceptions import IncompatibleOverrideError, IncompatibleSearchSpaceError
from baybe.kernels.basic import IndexKernel, MaternKernel, PositiveIndexKernel
from baybe.kernels.composite import ScaleKernel
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
    _BayBENumericalKernelFactory,
    _BayBETaskKernelFactory,
)
from baybe.targets.numerical import NumericalTarget

# A selector that accepts all parameters
_SELECT_ALL = lambda parameter: True  # noqa: E731


@pytest.mark.parametrize(
    ("factory", "parameters", "error"),
    [
        param(
            _BayBENumericalKernelFactory(parameter_selector=_SELECT_ALL),
            [TaskParameter("task", ["t1", "t2"])],
            IncompatibleSearchSpaceError,
            id="regular_rejects_task",
        ),
        param(
            _BayBETaskKernelFactory(parameter_selector=_SELECT_ALL),
            [CategoricalParameter("cat", ["a", "b"])],
            IncompatibleSearchSpaceError,
            id="task_rejects_categorical",
        ),
        param(
            _BayBETaskKernelFactory(parameter_selector=_SELECT_ALL),
            [NumericalDiscreteParameter("num", [1, 2, 3])],
            IncompatibleSearchSpaceError,
            id="task_rejects_numerical_discrete",
        ),
        param(
            _BayBETaskKernelFactory(parameter_selector=_SELECT_ALL),
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
    searchspace = SearchSpace.from_product(parameters)
    objective = NumericalTarget("y").to_objective()
    measurements = pd.DataFrame()

    with (
        nullcontext()
        if error is None
        else pytest.raises(error, match="does not support")
    ):
        factory(searchspace, objective, measurements)


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
    searchspace = SearchSpace.from_product(parameters)
    objective = NumericalTarget("y").to_objective()
    measurements = pd.DataFrame()

    kernel = _BayBETaskKernelFactory()(searchspace, objective, measurements)
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
    objective = NumericalTarget("y").to_objective()
    measurements = pd.DataFrame()
    return _ModelContext(searchspace, objective, measurements)


@pytest.mark.parametrize(
    ("override_mode", "kernel_or_factory", "expected_task_kernel_cls", "has_base"),
    [
        param(
            None,
            None,
            "PositiveIndexKernel",
            True,
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
            True,
            id="no_override+custom_icm_index_kernel_escape_hatch",
        ),
        param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            MaternKernel(),
            "PositiveIndexKernel",
            True,
            id="positive_index_override+bare_baybe_matern",
        ),
        param(
            TransferLearningMode.INDEX_KERNEL,
            MaternKernel(),
            "IndexKernel",
            True,
            id="index_override+bare_baybe_matern",
        ),
        param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            MaternKernel(parameter_names=("x", "Task")),
            "PositiveIndexKernel",
            True,
            id="positive_index_override+baybe_matern_with_task_name",
        ),
        param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            ScaleKernel(MaternKernel()),
            "PositiveIndexKernel",
            True,
            id="positive_index_override+scaled_baybe_matern",
        ),
        param(
            TransferLearningMode.INDEX_KERNEL,
            IndexKernel(num_tasks=3, rank=3, parameter_names=("Task",)),
            "IndexKernel",
            False,
            id="index_override+task_only_index_kernel",
        ),
        param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            None,
            "PositiveIndexKernel",
            True,
            id="positive_index_override+default_factory",
        ),
        param(
            TransferLearningMode.INDEX_KERNEL,
            None,
            "IndexKernel",
            True,
            id="index_override+default_factory",
        ),
    ],
)
def test_resolve_kernel_dispatch_success(
    monkeypatch, override_mode, kernel_or_factory, expected_task_kernel_cls, has_base
):
    """`_resolve_kernel` produces the expected task kernel for supported inputs."""
    monkeypatch.setenv("BAYBE_DISABLE_CUSTOM_KERNEL_WARNING", "True")

    context = _make_dispatch_context(override_mode)
    kwargs = (
        {} if kernel_or_factory is None else {"kernel_or_factory": kernel_or_factory}
    )
    surrogate = GaussianProcessSurrogate(**kwargs)

    kernel = surrogate._resolve_kernel(context)

    if has_base:
        # The resolved kernel is a product of base * task kernel
        _, task_kernel = kernel.kernels
    else:
        # Stripping left no non-task parameters -> only the task kernel remains
        task_kernel = kernel
    assert type(task_kernel).__name__ == expected_task_kernel_cls

    # Override branches must partition active dims so the task kernel acts exactly
    # on the task column.
    if override_mode is not None:
        assert task_kernel.active_dims is not None
        assert set(task_kernel.active_dims.tolist()) == {context.task_idx}


@pytest.mark.parametrize(
    ("override_mode", "kernel_or_factory"),
    [
        param(
            TransferLearningMode.INDEX_KERNEL,
            gpytorch.kernels.MaternKernel(nu=2.5),
            id="index_override+bare_gpytorch_matern",
        ),
        param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            MaternKernel(parameter_names=("x",))
            * IndexKernel(num_tasks=3, rank=3, parameter_names=("Task",)),
            id="override+product_kernel",
        ),
        param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            ICMKernelFactory(
                task_kernel_or_factory=IndexKernel(
                    num_tasks=3, rank=3, parameter_names=("Task",)
                )
            ),
            id="override+task_aware_factory",
        ),
    ],
)
def test_resolve_kernel_dispatch_raises(monkeypatch, override_mode, kernel_or_factory):
    """`_resolve_kernel` raises for inputs incompatible with an override.

    This covers raw gpytorch kernels, composite kernels, and task-aware factories.
    """
    monkeypatch.setenv("BAYBE_DISABLE_CUSTOM_KERNEL_WARNING", "True")

    context = _make_dispatch_context(override_mode)
    kwargs = (
        {} if kernel_or_factory is None else {"kernel_or_factory": kernel_or_factory}
    )
    surrogate = GaussianProcessSurrogate(**kwargs)

    with pytest.raises(IncompatibleOverrideError):
        surrogate._resolve_kernel(context)
