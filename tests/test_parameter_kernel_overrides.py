"""Tests for parameter-specific kernel overrides."""

from contextlib import nullcontext
from functools import partial
from unittest.mock import Mock

import pandas as pd
import pytest
from attrs import evolve
from gpytorch import kernels as gk
from gpytorch import means as gm
from pytest import param

from baybe.kernels import MaternKernel, RBFKernel
from baybe.kernels.base import Kernel
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel
from baybe.parameters import (
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.parameters.base import _iter_basic_kernels

_continuous = partial(NumericalContinuousParameter, "x", (0, 1))
_discrete = partial(NumericalDiscreteParameter, "x", (0, 1, 2))
_categorical = partial(CategoricalParameter, "x", ("a", "b", "c"))
_custom = partial(
    CustomDiscreteParameter,
    "x",
    pd.DataFrame({"d1": [1.0, 2.0], "d2": [3.0, 1.0]}, index=["a", "b"]),
)
_task = partial(TaskParameter, "x", ("a", "b"))

_SCOPE_ERROR = "may only act on the parameter itself"
_ACTIVE_DIMS_ERROR = "must not specify 'active_dims'"
_TYPE_ERROR = "must be a BayBE or GPyTorch kernel"
_TRAVERSE_ERROR = "Cannot traverse kernel"


@pytest.mark.parametrize(
    ("constructor", "override", "error", "match"),
    [
        # Valid BayBE overrides
        param(_continuous, RBFKernel(), None, None, id="baybe-unscoped"),
        param(
            _discrete,
            RBFKernel(parameter_names=("x",)),
            None,
            None,
            id="baybe-scoped-to-own-parameter",
        ),
        param(_categorical, ScaleKernel(RBFKernel()), None, None, id="baybe-scaled"),
        param(_custom, RBFKernel() + MaternKernel(), None, None, id="baybe-sum"),
        param(
            _continuous, RBFKernel() * MaternKernel(), None, None, id="baybe-product"
        ),
        param(
            _discrete,
            ScaleKernel(
                AdditiveKernel([RBFKernel(), ProductKernel([RBFKernel()] * 2)])
            ),
            None,
            None,
            id="baybe-nested",
        ),
        # Valid GPyTorch overrides
        param(_continuous, gk.RBFKernel(), None, None, id="gpytorch-plain"),
        param(
            _categorical, gk.RBFKernel(ard_num_dims=3), None, None, id="gpytorch-ard"
        ),
        param(
            _discrete, gk.ScaleKernel(gk.RBFKernel()), None, None, id="gpytorch-nested"
        ),
        # BayBE overrides acting on other parameters
        param(
            _continuous,
            RBFKernel(parameter_names=("other",)),
            ValueError,
            _SCOPE_ERROR,
            id="baybe-foreign-scope",
        ),
        param(
            _discrete,
            RBFKernel(parameter_names=("x", "other")),
            ValueError,
            _SCOPE_ERROR,
            id="baybe-multiple-names",
        ),
        param(
            _categorical,
            ScaleKernel(RBFKernel(parameter_names=("other",))),
            ValueError,
            _SCOPE_ERROR,
            id="baybe-nested-foreign-scope",
        ),
        # GPyTorch overrides with explicit active dimensions
        param(
            _continuous,
            gk.RBFKernel(active_dims=(0,)),
            ValueError,
            _ACTIVE_DIMS_ERROR,
            id="gpytorch-active-dims",
        ),
        param(
            _custom,
            gk.ScaleKernel(gk.RBFKernel(active_dims=(0,))),
            ValueError,
            _ACTIVE_DIMS_ERROR,
            id="gpytorch-nested-active-dims",
        ),
        # Objects that are not supported kernels
        param(_continuous, 1, TypeError, _TYPE_ERROR, id="no-kernel"),
        param(_discrete, gm.ConstantMean(), TypeError, _TYPE_ERROR, id="gpytorch-mean"),
        param(
            _continuous,
            Mock(spec=Kernel),
            TypeError,
            _TRAVERSE_ERROR,
            id="unknown-kernel-type",
        ),
        param(
            _categorical,
            ScaleKernel(Mock(spec=Kernel)),
            TypeError,
            _TRAVERSE_ERROR,
            id="unknown-nested-kernel-type",
        ),
        # Task parameters derive their override from the transfer learning mode
        param(
            _task,
            RBFKernel(),
            TypeError,
            "unexpected keyword argument 'override_kernel'",
            id="task-parameter",
        ),
    ],
)
def test_override_assignment(constructor, override, error, match):
    """Assigning a kernel override to a parameter succeeds or raises as expected."""
    with nullcontext() if error is None else pytest.raises(error, match=match):
        parameter = constructor(override_kernel=override)

    # Invalid overrides: nothing more to check
    if error is not None:
        return

    # GPyTorch overrides are returned unchanged
    if not isinstance(override, Kernel):
        assert parameter.override_kernel is override
        return

    # BayBE overrides are scoped to their parameter, also after renaming
    renamed = evolve(parameter, name="renamed")
    for p in (parameter, renamed):
        kernels = _iter_basic_kernels(p.override_kernel)
        assert all(k.parameter_names == (p.name,) for k in kernels)

    # Unscoped and owner-scoped input yield the same parameter
    assert parameter == constructor(override_kernel=override._scope_to_parameter("x"))
