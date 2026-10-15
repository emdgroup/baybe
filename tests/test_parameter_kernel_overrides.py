"""Tests for parameter-specific kernel overrides."""

from contextlib import nullcontext
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
from gpytorch.constraints import Interval
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
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.components.kernel import ICMKernelFactory
from baybe.surrogates.gaussian_process.core import _ModelContext
from baybe.surrogates.gaussian_process.presets import BayBEKernelFactory
from baybe.targets import NumericalTarget

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

_INDEX = TransferLearningMode.INDEX_KERNEL
_POSITIVE_INDEX = TransferLearningMode.POSITIVE_INDEX_KERNEL


def _rbf(lengthscale, *, frozen=False):
    """Create a GPyTorch RBF kernel with a given, optionally frozen, lengthscale."""
    kernel = gk.RBFKernel()
    kernel.lengthscale = lengthscale
    kernel.raw_lengthscale.requires_grad_(not frozen)
    return kernel


def _pair(override, other_override, other_bounds=(0, 1)):
    """Create two differently named continuous parameters with given overrides."""
    return (
        NumericalContinuousParameter("p1", (0, 1), override_kernel=override),
        NumericalContinuousParameter(
            "p2", other_bounds, override_kernel=other_override
        ),
    )


def _task_pair(values, mode, other_values, other_mode):
    """Create two differently named task parameters."""
    return (
        TaskParameter("t1", values, override_transfer_learning_mode=mode),
        TaskParameter("t2", other_values, override_transfer_learning_mode=other_mode),
    )


def _x(name, override=None):
    """Create a continuous parameter on [0, 1], optionally with a kernel override."""
    return NumericalContinuousParameter(name, (0, 1), override_kernel=override)


def _tl_task(mode=None):
    """Create a three-valued task parameter, optionally with a TL mode."""
    return TaskParameter("task", ("a", "b", "c"), override_transfer_learning_mode=mode)


def _basic_kernels(kernel, columns):
    """Yield each basic kernel of a GPyTorch kernel with the columns it acts on.

    Basic kernels compute on columns (e.g. RBF, Matern), as opposed to wrappers such
    as Scale, Additive, or Product kernels. As in GPyTorch, ``active_dims`` of a
    kernel apply to everything nested inside it.
    """
    if kernel.active_dims is not None:
        columns = set(kernel.active_dims.tolist())
    children = [c for c in kernel.children() if isinstance(c, gk.Kernel)]
    children += [
        m for c in kernel.children() if isinstance(c, torch.nn.ModuleList) for m in c
    ]
    if not children:
        yield kernel, columns
    for child in children:
        yield from _basic_kernels(child, columns)


def _factors_by_parameter(kernel, searchspace):
    """Map the parameters each basic kernel acts on to the basic kernel classes.

    Example: ``{("x1", "x3"): [MaternKernel], ("x2",): [RBFKernel]}``
    """
    factors: dict[tuple[str, ...], list[type]] = {}
    for k, names, _ in _basic_kernels_by_parameter(kernel, searchspace):
        factors.setdefault(names, []).append(type(k))
    return factors


def _basic_kernels_by_parameter(kernel, searchspace):
    """Yield each basic kernel with the parameters and columns it acts on."""
    all_columns = set(range(len(searchspace.comp_rep_columns)))
    for k, columns in _basic_kernels(kernel, all_columns):
        names = tuple(
            p.name
            for p in searchspace.parameters
            if columns & set(searchspace.get_comp_rep_parameter_indices(p.name))
        )
        yield k, names, columns


def _matern_naming_all(searchspace, objective, measurements):
    """Return a Matern kernel naming all parameters of the given search space."""
    return MaternKernel(parameter_names=searchspace.parameter_names)


def _unnamed_matern(searchspace, objective, measurements):
    """Return a Matern kernel without parameter names."""
    return MaternKernel()


def _raw_matern(searchspace, objective, measurements):
    """Return a raw GPyTorch kernel, which cannot exclude overridden parameters."""
    return gk.MaternKernel()


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


_SAME_GPYTORCH_KERNEL = gk.RBFKernel()


@pytest.mark.parametrize(
    ("p1", "p2", "expected"),
    [
        # Equivalent: only the name differs
        param(*_pair(None, None), True, id="no-override"),
        param(*_pair(RBFKernel(), RBFKernel()), True, id="baybe"),
        param(
            *_pair(RBFKernel(), RBFKernel(parameter_names=("p2",))),
            True,
            id="baybe-unscoped-vs-scoped",
        ),
        param(
            *_pair(
                ScaleKernel(RBFKernel() + MaternKernel()),
                ScaleKernel(RBFKernel() + MaternKernel()),
            ),
            True,
            id="baybe-composite",
        ),
        param(*_pair(gk.MaternKernel(), gk.MaternKernel()), True, id="gpytorch"),
        param(
            *_pair(
                gk.ScaleKernel(
                    gk.RBFKernel(lengthscale_prior=GPyTorchGammaPrior(3.0, 6.0))
                ),
                gk.ScaleKernel(
                    gk.RBFKernel(lengthscale_prior=GPyTorchGammaPrior(3.0, 6.0))
                ),
            ),
            True,
            id="gpytorch-nested",
        ),
        param(
            *_pair(gk.IndexKernel(num_tasks=2), gk.IndexKernel(num_tasks=2)),
            True,
            id="gpytorch-random-init",
        ),
        param(*_pair(_rbf(2.0), _rbf(3.0)), True, id="gpytorch-initial-value"),
        param(
            *_pair(_rbf(2.0, frozen=True), _rbf(2.0, frozen=True)),
            True,
            id="gpytorch-same-frozen-value",
        ),
        param(
            *_pair(_SAME_GPYTORCH_KERNEL, _SAME_GPYTORCH_KERNEL),
            True,
            id="gpytorch-same-instance",
        ),
        param(
            *_task_pair(("a", "b"), _INDEX, ("a", "b"), _INDEX),
            True,
            id="task-same-mode",
        ),
        param(*_task_pair(("a", "b"), None, ("a", "b"), None), True, id="task-no-mode"),
        # Not equivalent: specification differs
        param(
            NumericalContinuousParameter("p1", (0, 1), override_kernel=RBFKernel()),
            NumericalDiscreteParameter("p2", (0, 1), override_kernel=RBFKernel()),
            False,
            id="parameter-type",
        ),
        param(
            *_pair(RBFKernel(), RBFKernel(), other_bounds=(0, 2)),
            False,
            id="parameter-spec",
        ),
        param(*_pair(RBFKernel(), None), False, id="override-vs-none"),
        param(*_pair(RBFKernel(), MaternKernel()), False, id="baybe-class"),
        param(
            *_pair(RBFKernel(), RBFKernel(lengthscale_prior=GammaPrior(3.0, 6.0))),
            False,
            id="baybe-setting",
        ),
        param(*_pair(RBFKernel(), gk.RBFKernel()), False, id="baybe-vs-gpytorch"),
        param(*_pair(gk.RBFKernel(), gk.MaternKernel()), False, id="gpytorch-class"),
        param(
            *_pair(gk.MaternKernel(nu=0.5), gk.MaternKernel()),
            False,
            id="gpytorch-attribute",
        ),
        param(
            *_pair(
                gk.RBFKernel(lengthscale_prior=GPyTorchGammaPrior(3.0, 6.0)),
                gk.RBFKernel(lengthscale_prior=GPyTorchGammaPrior(2.0, 6.0)),
            ),
            False,
            id="gpytorch-prior",
        ),
        param(
            *_pair(
                gk.RBFKernel(lengthscale_constraint=Interval(0.1, 2.0)),
                gk.RBFKernel(lengthscale_constraint=Interval(0.1, 3.0)),
            ),
            False,
            id="gpytorch-constraint",
        ),
        param(
            *_pair(_rbf(2.0, frozen=True), _rbf(3.0, frozen=True)),
            False,
            id="gpytorch-frozen-value",
        ),
        param(
            *_pair(_rbf(2.0, frozen=True), _rbf(2.0)),
            False,
            id="gpytorch-frozen-vs-learnable",
        ),
        param(
            *_pair(
                gk.IndexKernel(num_tasks=2, rank=1), gk.IndexKernel(num_tasks=2, rank=2)
            ),
            False,
            id="gpytorch-shape",
        ),
        param(
            *_pair(
                gk.RBFKernel() + gk.MaternKernel(), gk.MaternKernel() + gk.RBFKernel()
            ),
            False,
            id="gpytorch-summand-order",
        ),
        param(
            *_task_pair(("a", "b"), _INDEX, ("a", "b"), _POSITIVE_INDEX),
            False,
            id="task-mode",
        ),
        param(
            *_task_pair(("a", "b"), None, ("a", "b", "c"), None),
            False,
            id="task-values",
        ),
        param(
            *_task_pair(("a", "b"), _INDEX, ("a", "b"), None),
            False,
            id="task-mode-vs-none",
        ),
    ],
)
def test_parameter_equivalence(p1, p2, expected):
    """Parameter equivalence with kernel overrides produces the expected result."""
    assert p1.is_equivalent(p2) == expected
    assert p2.is_equivalent(p1) == expected


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
            _matern_naming_all,
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
            [_x("x1"), _tl_task()],
            None,
            {("x1",): [gk.MaternKernel], ("task",): [BoPositiveIndexKernel]},
            id="task-without-mode",
        ),
        param(
            [_x("x1"), _tl_task(_INDEX)],
            None,
            {("x1",): [gk.MaternKernel], ("task",): [gk.IndexKernel]},
            id="task-index",
        ),
        param(
            [_x("x1"), _tl_task(_POSITIVE_INDEX)],
            None,
            {("x1",): [gk.MaternKernel], ("task",): [BoPositiveIndexKernel]},
            id="task-positive-index",
        ),
        param(
            [_x("x1"), _tl_task(_INDEX)],
            MaternKernel(parameter_names=("x1", "task")),
            {("x1",): [gk.MaternKernel], ("task",): [gk.IndexKernel]},
            id="task-index+kernel-naming-task",
        ),
        param(
            [_x("x1"), _tl_task(_INDEX)],
            MaternKernel(parameter_names=("x1",)) * _TASK_KERNEL,
            {("x1",): [gk.MaternKernel], ("task",): [gk.IndexKernel]},
            id="task-index+product-with-task-kernel",
        ),
        param(
            [_x("x1"), _tl_task(_INDEX)],
            _TASK_KERNEL,
            {("task",): [gk.IndexKernel]},
            id="task-index+task-only-kernel",
        ),
        param(
            [_x("x1"), _tl_task(_INDEX)],
            _unnamed_matern,
            {("x1",): [gk.MaternKernel], ("task",): [gk.IndexKernel]},
            id="task-index+factory",
        ),
        param(
            [_x("x1"), _x("x2", RBFKernel()), _tl_task(_POSITIVE_INDEX)],
            None,
            {
                ("x1",): [gk.MaternKernel],
                ("task",): [BoPositiveIndexKernel],
                ("x2",): [gk.RBFKernel],
            },
            id="task-positive-index+parameter-override",
        ),
        param(
            [_x("x1"), _x("x2", RBFKernel()), _tl_task()],
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
            _raw_matern,
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
            [_x("x1"), _tl_task(_INDEX)],
            ICMKernelFactory(task_kernel_or_factory=_TASK_KERNEL),
            (IncompatibleOverrideError, _EXCLUDE_ERROR),
            id="task-index+icm-factory",
        ),
    ],
)
def test_override_kernel_construction(parameters, kernel_or_factory, expected):
    """Kernel construction with overrides yields the expected kernel or error."""
    searchspace = SearchSpace.from_product(parameters)
    context = _ModelContext(
        searchspace, NumericalTarget("y").to_objective(), pd.DataFrame()
    )
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
    gpytorch_overrides = {
        p.name: p.override_kernel
        for p in parameters
        if p.override_kernel is not None and not isinstance(p.override_kernel, Kernel)
    }
    for override in gpytorch_overrides.values():
        assert all(k.active_dims is None for k in (override, *override.sub_kernels()))

    # BayBE-created kernels use one ARD dimension per column, user-provided GPyTorch
    # kernels keep their setting
    user_kernel = isinstance(kernel_or_factory, gk.Kernel)
    for k, names, columns in _basic_kernels_by_parameter(kernel, searchspace):
        if user_kernel or set(names) & gpytorch_overrides.keys():
            assert k.ard_num_dims in (None, len(columns))
        else:
            assert k.ard_num_dims == len(columns)
