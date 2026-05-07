"""Tests for kernel factories."""

from contextlib import nullcontext

import pytest
import torch
from pytest import param

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.parameters.categorical import CategoricalParameter, TaskParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets.baybe import (
    BayBEKernelFactory,
    _BayBENumericalKernelFactory,
    _BayBETaskKernelFactory,
)
from baybe.surrogates.gaussian_process.presets.chen import CHENKernelFactory
from baybe.surrogates.gaussian_process.presets.edbo import EDBOKernelFactory
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOKernelFactory,
)

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
    "factory",
    [
        param(BayBEKernelFactory(), id="BayBEKernelFactory"),
        param(SmoothedEDBOKernelFactory(), id="SmoothedEDBOKernelFactory"),
        param(EDBOKernelFactory(), id="EDBOKernelFactory"),
        param(CHENKernelFactory(), id="CHENKernelFactory"),
    ],
)
def test_kernel_factory_serialization_roundtrip(factory):
    """Kernel factories survive a serialization roundtrip via a GP surrogate."""
    gp = GaussianProcessSurrogate(kernel_or_factory=factory)
    json_str = gp.to_json()
    gp_roundtrip = GaussianProcessSurrogate.from_json(json_str)
    assert type(gp.kernel_factory) is type(gp_roundtrip.kernel_factory)
    assert gp == gp_roundtrip
