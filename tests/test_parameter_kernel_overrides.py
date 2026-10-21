"""Tests for parameter-specific kernel overrides."""

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from attrs import evolve
from botorch.models.kernels.positive_index import (
    PositiveIndexKernel as BoPositiveIndexKernel,
)
from gpytorch import kernels as gk
from gpytorch import means as gm
from gpytorch.priors import GammaPrior as GPyTorchGammaPrior
from pytest import param

from baybe.exceptions import IncompatibleOverrideError
from baybe.kernels import MaternKernel, RBFKernel
from baybe.kernels.base import Kernel
from baybe.kernels.basic import IndexKernel
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel
from baybe.parameters import (
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.parameters.base import _iter_basic_kernels
from baybe.parameters.enum import TransferLearningMode
from baybe.parameters.selectors import NameSelector
from baybe.priors import GammaPrior
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.components.kernel import ICMKernelFactory
from baybe.surrogates.gaussian_process.core import _ModelContext
from baybe.surrogates.gaussian_process.presets import BayBEKernelFactory
from baybe.utils.dataframe import create_fake_input
from tests.test_kernel_factories import (
    _gpytorch_returning_factory,
    _matern_kernel_factory,
    _unnamed_matern_factory,
)

_PRIOR = GammaPrior(3.0, 6.0)

_SCOPE_ERROR = "may only act on the parameter itself"
_ACTIVE_DIMS_ERROR = "must not specify 'active_dims'"
_TYPE_ERROR = "must be a BayBE or GPyTorch kernel"
_TRAVERSE_ERROR = "Cannot traverse kernel"

_INDEX = TransferLearningMode.INDEX_KERNEL
_POSITIVE_INDEX = TransferLearningMode.POSITIVE_INDEX_KERNEL
_IDENTITY = TransferLearningMode.IDENTITY


def _x(name, override_kernel=None):
    """Create a continuous parameter on [0, 1], optionally with a kernel override."""
    return NumericalContinuousParameter(name, (0, 1), override_kernel=override_kernel)


def _task(mode=None, *, name="task", values=("a", "b", "c"), **kwargs):
    """Create a task parameter, optionally with a transfer learning mode."""
    return TaskParameter(name, values, override_transfer_learning_mode=mode, **kwargs)


_continuous = partial(_x, "x")
_discrete = partial(NumericalDiscreteParameter, "x", (0, 1, 2))
_categorical = partial(CategoricalParameter, "x", ("a", "b", "c"))
_custom = partial(
    CustomDiscreteParameter,
    "x",
    pd.DataFrame({"d1": [1.0, 2.0], "d2": [3.0, 1.0]}, index=["a", "b"]),
)


def _gpytorch_overrides(parameters):
    """Map parameter names to their user-provided GPyTorch kernel overrides."""
    return {
        p.name: p.override_kernel
        for p in parameters
        if p.override_kernel is not None and not isinstance(p.override_kernel, Kernel)
    }


def _basic_kernels(kernel, searchspace):
    """List each basic kernel with the parameters and columns it acts on.

    Basic kernels compute on columns (e.g. RBF, Matern), as opposed to wrappers such
    as Scale, Additive, or Product kernels. As in GPyTorch, ``active_dims`` of a
    kernel apply to everything nested inside it.
    """

    def walk(k, columns):
        if k.active_dims is not None:
            columns = set(k.active_dims.tolist())
        children = [c for c in k.children() if isinstance(c, gk.Kernel)]
        children += [
            m for c in k.children() if isinstance(c, torch.nn.ModuleList) for m in c
        ]
        if not children:
            yield k, columns
        for child in children:
            yield from walk(child, columns)

    all_columns = set(range(len(searchspace.comp_rep_columns)))
    result = []
    for k, columns in walk(kernel, all_columns):
        names = tuple(
            p.name
            for p in searchspace.parameters
            if columns & set(searchspace.get_comp_rep_parameter_indices(p.name))
        )
        result.append((k, names, columns))
    return result


def _factors_by_parameter(kernel, searchspace):
    """Map the parameters each basic kernel acts on to the basic kernel classes.

    Example: ``{("x1", "x3"): [MaternKernel], ("x2",): [RBFKernel]}``
    """
    factors: dict[tuple[str, ...], list[type]] = {}
    for k, names, _ in _basic_kernels(kernel, searchspace):
        factors.setdefault(names, []).append(type(k))
    return factors


def _needs_full_searchspace(searchspace, objective, measurements):
    """Return a kernel after accessing information unavailable on reduced spaces."""
    searchspace.task_idx
    return MaternKernel()


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
            partial(_task, name="x", values=("a", "b")),
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


_X2_RBF = [_x("x1"), _x("x2", RBFKernel()), _x("x3")]
_X1_SELECTOR = BayBEKernelFactory(parameter_selector=NameSelector(("x1",), regex=False))
_TASK_KERNEL = IndexKernel(num_tasks=3, rank=3, parameter_names=("task",))
_EXCLUDE_ERROR = "can exclude these parameters"
_ARD_ERROR = "specifies 2 ARD dimensions, but the parameter has 3"


@pytest.mark.parametrize(
    ("parameters", "kernel_or_factory", "expected"),
    [
        # No overrides: the surrogate kernel is used unchanged
        param(
            [_x("x1"), _x("x2")],
            None,
            {("x1", "x2"): [gk.MaternKernel]},
            id="no-override-default",
        ),
        param(
            [_x("x1"), _x("x2")],
            gk.MaternKernel(),
            {("x1", "x2"): [gk.MaternKernel]},
            id="no-override-gpytorch",
        ),
        # One override, the surrogate kernel acts on the remaining parameters
        param(
            _X2_RBF,
            None,
            {("x1", "x3"): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="default",
        ),
        param(
            _X2_RBF,
            MaternKernel(),
            {("x1", "x3"): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="fixed",
        ),
        param(
            _X2_RBF,
            MaternKernel(parameter_names=("x1", "x2")),
            {("x1",): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="fixed-naming-overridden-parameter",
        ),
        param(
            _X2_RBF,
            ScaleKernel(MaternKernel()),
            {("x1", "x3"): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="scaled",
        ),
        param(
            _X2_RBF,
            MaternKernel() + RBFKernel(),
            {("x1", "x3"): [gk.MaternKernel, gk.RBFKernel], ("x2",): [gk.RBFKernel]},
            id="sum",
        ),
        param(
            _X2_RBF,
            MaternKernel() * RBFKernel(),
            {("x1", "x3"): [gk.MaternKernel, gk.RBFKernel], ("x2",): [gk.RBFKernel]},
            id="product",
        ),
        param(
            _X2_RBF,
            _matern_kernel_factory,
            {("x1", "x3"): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="factory",
        ),
        param(
            _X2_RBF,
            _X1_SELECTOR,
            {("x1",): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="selector",
        ),
        # Override shapes
        param(
            [
                _x("x1"),
                CategoricalParameter("c", ("a", "b", "c"), override_kernel=RBFKernel()),
            ],
            None,
            {("x1",): [gk.MaternKernel], ("c",): [gk.RBFKernel]},
            id="multi-column-parameter",
        ),
        param(
            [_x("x1"), _x("x2", RBFKernel()), _x("x3", MaternKernel())],
            None,
            {
                ("x1",): [gk.MaternKernel],
                ("x2",): [gk.RBFKernel],
                ("x3",): [gk.MaternKernel],
            },
            id="multiple-overrides",
        ),
        param(
            [_x("x1", RBFKernel()), _x("x2", MaternKernel())],
            MaternKernel(),
            {("x1",): [gk.RBFKernel], ("x2",): [gk.MaternKernel]},
            id="all-overridden",
        ),
        param(
            [_x("x1"), _x("x2", gk.RBFKernel())],
            None,
            {("x1",): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="gpytorch-override",
        ),
        param(
            [
                _x("x1"),
                CategoricalParameter(
                    "c", ("a", "b", "c"), override_kernel=gk.RBFKernel(ard_num_dims=3)
                ),
            ],
            None,
            {("x1",): [gk.MaternKernel], ("c",): [gk.RBFKernel]},
            id="gpytorch-override-ard",
        ),
        param(
            [_x("x1"), _x("x2", gk.ScaleKernel(gk.RBFKernel()))],
            None,
            {("x1",): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="gpytorch-override-nested",
        ),
        # Task parameters: the transfer learning mode defines the task kernel
        param(
            [_x("x1"), _task()],
            None,
            {("x1",): [gk.MaternKernel], ("task",): [BoPositiveIndexKernel]},
            id="task-without-mode",
        ),
        param(
            [_x("x1"), _task(_INDEX)],
            None,
            {("x1",): [gk.MaternKernel], ("task",): [gk.IndexKernel]},
            id="task-index",
        ),
        param(
            [_x("x1"), _task(_POSITIVE_INDEX)],
            None,
            {("x1",): [gk.MaternKernel], ("task",): [BoPositiveIndexKernel]},
            id="task-positive-index",
        ),
        param(
            [_x("x1"), _task(_IDENTITY)],
            None,
            {("x1",): [gk.MaternKernel], ("task",): [gk.ConstantKernel]},
            id="task-identity",
        ),
        param(
            [_x("x1"), _task(_INDEX)],
            MaternKernel(parameter_names=("x1", "task")),
            {("x1",): [gk.MaternKernel], ("task",): [gk.IndexKernel]},
            id="task-index+kernel-naming-task",
        ),
        param(
            [_x("x1"), _task(_INDEX)],
            MaternKernel(parameter_names=("x1",)) * _TASK_KERNEL,
            {("x1",): [gk.MaternKernel], ("task",): [gk.IndexKernel]},
            id="task-index+product-with-task-kernel",
        ),
        param(
            [_x("x1"), _task(_INDEX)],
            _TASK_KERNEL,
            {("task",): [gk.IndexKernel]},
            id="task-index+task-only-kernel",
        ),
        param(
            [_x("x1"), _task(_INDEX)],
            _unnamed_matern_factory,
            {("x1",): [gk.MaternKernel], ("task",): [gk.IndexKernel]},
            id="task-index+factory",
        ),
        param(
            [_x("x1"), _x("x2", RBFKernel()), _task(_POSITIVE_INDEX)],
            None,
            {
                ("x1",): [gk.MaternKernel],
                ("task",): [BoPositiveIndexKernel],
                ("x2",): [gk.RBFKernel],
            },
            id="task-positive-index+parameter-override",
        ),
        param(
            [_x("x1"), _x("x2", RBFKernel()), _task()],
            _X1_SELECTOR,
            {
                ("x1",): [gk.MaternKernel],
                ("task",): [BoPositiveIndexKernel],
                ("x2",): [gk.RBFKernel],
            },
            id="task-without-mode+selector",
        ),
        # Invalid configurations detected at kernel construction
        param(
            [
                _x("x1"),
                CategoricalParameter(
                    "c", ("a", "b", "c"), override_kernel=gk.RBFKernel(ard_num_dims=2)
                ),
            ],
            None,
            (IncompatibleOverrideError, _ARD_ERROR),
            id="gpytorch-ard-mismatch",
        ),
        param(
            [
                _x("x1"),
                CategoricalParameter(
                    "c",
                    ("a", "b", "c"),
                    override_kernel=gk.ScaleKernel(gk.RBFKernel(ard_num_dims=2)),
                ),
            ],
            None,
            (IncompatibleOverrideError, _ARD_ERROR),
            id="gpytorch-nested-ard-mismatch",
        ),
        param(
            _X2_RBF,
            gk.MaternKernel(),
            (IncompatibleOverrideError, _EXCLUDE_ERROR),
            id="gpytorch-surrogate-kernel",
        ),
        param(
            _X2_RBF,
            _gpytorch_returning_factory,
            (IncompatibleOverrideError, _EXCLUDE_ERROR),
            id="factory-returning-gpytorch-kernel",
        ),
        param(
            _X2_RBF,
            _needs_full_searchspace,
            (IncompatibleOverrideError, _EXCLUDE_ERROR),
            id="factory-needing-full-searchspace",
        ),
        param(
            [_x("x1"), _task(_INDEX)],
            ICMKernelFactory(task_kernel_or_factory=_TASK_KERNEL),
            (IncompatibleOverrideError, _EXCLUDE_ERROR),
            id="task-index+icm-factory",
        ),
    ],
)
def test_override_kernel_construction(
    searchspace, objective, kernel_or_factory, expected
):
    """Kernel construction with overrides yields the expected kernel or error."""
    context = _ModelContext(searchspace, objective, pd.DataFrame())
    surrogate = GaussianProcessSurrogate(kernel_or_factory=kernel_or_factory)

    # Invalid configurations raise during construction
    if isinstance(expected, tuple):
        error, match = expected
        with pytest.raises(error, match=match):
            surrogate._resolve_kernel(context)
        return

    # Each parameter is covered by the expected kernel classes
    kernel = surrogate._resolve_kernel(context)
    assert _factors_by_parameter(kernel, searchspace) == expected

    # User-provided GPyTorch overrides are copied, leaving the input unbound
    gpytorch_overrides = _gpytorch_overrides(searchspace.parameters)
    for override in gpytorch_overrides.values():
        assert all(k.active_dims is None for k in (override, *override.sub_kernels()))

    # BayBE-created kernels use one ARD dimension per column, user-provided GPyTorch
    # kernels keep their setting
    user_kernel = isinstance(kernel_or_factory, gk.Kernel)
    for k, names, columns in _basic_kernels(kernel, searchspace):
        if user_kernel or set(names) & gpytorch_overrides.keys():
            assert k.ard_num_dims in (None, len(columns))
        elif isinstance(k, gk.ConstantKernel):
            # The identity task kernel is constant and carries no ARD dimensions
            assert k.ard_num_dims is None
        else:
            assert k.ard_num_dims == len(columns)


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        param(
            [_x("x1"), _x("x2", RBFKernel(lengthscale_prior=_PRIOR))],
            {("x1",): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="parameter-override",
        ),
        param(
            [
                _x("x1"),
                CategoricalParameter(
                    "c",
                    ("a", "b", "c"),
                    override_kernel=RBFKernel(lengthscale_prior=_PRIOR),
                ),
                _task(_INDEX),
            ],
            {
                ("x1",): [gk.MaternKernel],
                ("c",): [gk.RBFKernel],
                ("task",): [gk.IndexKernel],
            },
            id="task-index+parameter-override",
        ),
        param(
            [
                _x("x1", RBFKernel(lengthscale_prior=_PRIOR)),
                _x("x2", MaternKernel(lengthscale_prior=_PRIOR)),
            ],
            {("x1",): [gk.RBFKernel], ("x2",): [gk.MaternKernel]},
            id="all-overridden",
        ),
        param(
            [
                _x("x1"),
                _x(
                    "x2",
                    gk.RBFKernel(lengthscale_prior=GPyTorchGammaPrior(3.0, 6.0)),
                ),
            ],
            {("x1",): [gk.MaternKernel], ("x2",): [gk.RBFKernel]},
            id="gpytorch-override",
        ),
    ],
)
def test_fitted_override_kernel(searchspace, objective, expected):
    """The fitted model uses the kernel constructed from the overrides."""
    measurements = create_fake_input(searchspace.parameters, objective.targets, 6)

    # Snapshot user-provided GPyTorch overrides to detect mutations by training
    gpytorch_overrides = list(_gpytorch_overrides(searchspace.parameters).values())
    snapshots = [deepcopy(o.state_dict()) for o in gpytorch_overrides]

    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    # The model was fitted on data using the expected kernel structure
    assert surrogate._model is not None
    kernel = surrogate._model.covar_module
    assert _factors_by_parameter(kernel, searchspace) == expected

    # Training did not touch the user-provided GPyTorch overrides
    for override, snapshot in zip(gpytorch_overrides, snapshots):
        assert all(k.active_dims is None for k in (override, *override.sub_kernels()))
        assert all(
            torch.equal(v, override.state_dict()[k]) for k, v in snapshot.items()
        )
