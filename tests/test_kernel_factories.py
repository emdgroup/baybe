"""Tests for kernel factories."""

from contextlib import nullcontext

import pandas as pd
import pytest
from pytest import param

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.parameters.categorical import CategoricalParameter, TaskParameter
from baybe.parameters.fidelity import (
    CategoricalFidelityParameter,
    NumericalDiscreteFidelityParameter,
)
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.kernel import (
    ICMKernelFactory,
    _enable_index_kernel,
)
from baybe.surrogates.gaussian_process.presets.baybe import (
    BayBEKernelFactory,
    _BayBEIndexKernelFactory,
    _BayBENumericalKernelFactory,
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
            _BayBEIndexKernelFactory(parameter_selector=_SELECT_ALL),
            [CategoricalParameter("cat", ["a", "b"])],
            IncompatibleSearchSpaceError,
            id="task_rejects_categorical",
        ),
        param(
            _BayBEIndexKernelFactory(parameter_selector=_SELECT_ALL),
            [NumericalDiscreteParameter("num", [1, 2, 3])],
            IncompatibleSearchSpaceError,
            id="task_rejects_numerical_discrete",
        ),
        param(
            _BayBEIndexKernelFactory(parameter_selector=_SELECT_ALL),
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
            id="combined_accepts_task",
        ),
        param(
            BayBEKernelFactory(),
            [
                NumericalContinuousParameter("cont", (0, 1)),
                CategoricalFidelityParameter(
                    "fid", ["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
            ],
            None,
            id="combined_accepts_fidelity",
        ),
        param(
            _BayBENumericalKernelFactory(parameter_selector=_SELECT_ALL),
            [
                CategoricalFidelityParameter(
                    "fid", ["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                )
            ],
            IncompatibleSearchSpaceError,
            id="regular_rejects_fidelity",
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


def test_enable_index_kernel_guard():
    """_enable_index_kernel raises when the factory already supports task/fidelity."""
    with pytest.raises(TypeError, match="already supports task or fidelity"):
        _enable_index_kernel(_BayBEIndexKernelFactory)


_icm_guard_message = "contains a 'TaskParameter' or a 'CategoricalFidelityParameter'"


@pytest.mark.parametrize(
    "parameters",
    [
        param([NumericalContinuousParameter("x", (0, 1))], id="plain"),
        param(
            [
                NumericalContinuousParameter("x", (0, 1)),
                NumericalDiscreteFidelityParameter(
                    "fid", values=[0.5, 1.0], costs=[1.0, 10.0]
                ),
            ],
            id="numerical_fidelity",
        ),
    ],
)
def test_icm_guard_rejects_non_index_space(parameters):
    """ICMKernelFactory raises for spaces without an index parameter."""
    searchspace = SearchSpace.from_product(parameters)
    with pytest.raises(IncompatibleSearchSpaceError, match=_icm_guard_message):
        ICMKernelFactory()(
            searchspace, NumericalTarget("y").to_objective(), pd.DataFrame()
        )
