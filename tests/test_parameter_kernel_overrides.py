"""Functional tests for parameter-specific kernel overrides.

These tests resolve or fit kernels to check binding and composition behavior.
"""

from copy import deepcopy

import pandas as pd
import pytest
import torch
from attrs import evolve
from botorch.models.kernels.positive_index import PositiveIndexKernel
from gpytorch import kernels as gk
from gpytorch.priors import GammaPrior
from pytest import param

from baybe.exceptions import IncompatibleOverrideError
from baybe.kernels import MaternKernel, RBFKernel
from baybe.kernels.basic import IndexKernel
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.parameters.enum import TransferLearningMode
from baybe.parameters.selectors import NameSelector
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.components.kernel import ICMKernelFactory
from baybe.surrogates.gaussian_process.core import _ModelContext
from baybe.surrogates.gaussian_process.presets import BayBEKernelFactory
from baybe.targets import NumericalTarget


def _resolve(parameters, kernel_or_factory=None):
    """Resolve the default GP kernel for the given parameters."""
    searchspace = SearchSpace.from_product(parameters)
    context = _ModelContext(
        searchspace, NumericalTarget("y").to_objective(), pd.DataFrame()
    )
    surrogate = GaussianProcessSurrogate(kernel_or_factory=kernel_or_factory)
    return surrogate._resolve_kernel(context), searchspace


def _leaf_kernels(kernel):
    """Return the leaf kernels of a GPyTorch kernel tree."""
    children = tuple(kernel.sub_kernels())
    return (
        (kernel,)
        if not children
        else tuple(k for c in children for k in _leaf_kernels(c))
    )


def _rbf_kernel(lengthscale, *, frozen=False):
    """Create a GPyTorch RBF kernel with a given, optionally frozen, lengthscale."""
    kernel = gk.RBFKernel()
    kernel.lengthscale = lengthscale
    kernel.raw_lengthscale.requires_grad_(not frozen)
    return kernel


@pytest.mark.parametrize(
    "mode",
    [None, *TransferLearningMode],
    ids=lambda mode: mode.name if mode else "no-task",
)
def test_selector_does_not_exclude_overridden_parameters(mode):
    """A selector narrows the surrogate kernel, but overrides apply regardless.

    The factory selects only ``x1``, so ``omitted`` contributes no kernel factor.
    ``x2`` is likewise unselected, yet its override still yields its own factor.
    """
    parameters = [
        NumericalContinuousParameter("x1", (0, 1)),
        NumericalContinuousParameter("x2", (0, 1), override_kernel=RBFKernel()),
        NumericalContinuousParameter("omitted", (0, 1)),
    ]
    expected_names = ["x1", "x2"]
    if mode is not None:
        parameters.append(
            TaskParameter("task", ["a", "b"], override_transfer_learning_mode=mode)
        )
        expected_names.append("task")

    kernel, searchspace = _resolve(
        parameters,
        BayBEKernelFactory(parameter_selector=NameSelector(("x1",), regex=False)),
    )

    actual_dimensions = {tuple(k.active_dims.tolist()) for k in _leaf_kernels(kernel)}
    expected_dimensions = {
        searchspace.get_comp_rep_parameter_indices(name) for name in expected_names
    }
    assert actual_dimensions == expected_dimensions


@pytest.mark.parametrize(
    ("kernel_or_factory", "expected_residual_dims"),
    [
        param(MaternKernel(), (0, 2), id="unnamed"),
        param(MaternKernel(parameter_names=("x1", "x2")), (0,), id="named"),
        param(ScaleKernel(MaternKernel()), (0, 2), id="scaled"),
        param(
            lambda space, _, __: MaternKernel(parameter_names=space.parameter_names),
            (0, 2),
            id="callable",
        ),
    ],
)
def test_nondefault_residual_indices(kernel_or_factory, expected_residual_dims):
    """Removing a middle parameter preserves the original computational indices."""
    kernel, _ = _resolve(
        [
            NumericalContinuousParameter("x1", (0, 1)),
            NumericalContinuousParameter("x2", (0, 1), override_kernel=RBFKernel()),
            NumericalContinuousParameter("x3", (0, 1)),
        ],
        kernel_or_factory,
    )
    residual, override = kernel.kernels
    assert tuple(residual.active_dims.tolist()) == expected_residual_dims
    assert tuple(override.active_dims.tolist()) == (1,)
    assert isinstance(override, gk.RBFKernel)


@pytest.mark.parametrize(
    ("override_parameter", "base_override", "task_parameter"),
    [
        (NumericalDiscreteParameter("override", [0, 1, 2]), None, None),
        (CategoricalParameter("override", ["a", "b", "c"]), None, None),
        (CategoricalParameter("override", ["a", "b"]), MaternKernel(), None),
        (
            CategoricalParameter("override", ["a", "b"]),
            None,
            TaskParameter("task", ["a", "b"]),
        ),
        *[
            (
                CategoricalParameter("override", ["a", "b", "c"]),
                None,
                TaskParameter("task", ["a", "b"], override_transfer_learning_mode=mode),
            )
            for mode in TransferLearningMode
        ],
    ],
    ids=[
        "single-dim",
        "multi-dim",
        "all-overridden",
        "task-without-tl-override",
        *(mode.name for mode in TransferLearningMode),
    ],
)
def test_fitted_model_uses_parameter_kernel_overrides(
    override_parameter, base_override, task_parameter
):
    """The fitted model uses each configured kernel on the intended dimensions."""
    parameters = [
        CategoricalParameter("base", ["a", "b"], override_kernel=base_override),
        evolve(override_parameter, override_kernel=RBFKernel()),
    ]
    expected = {"base": gk.MaternKernel, "override": gk.RBFKernel}
    if task_parameter is not None:
        parameters.append(task_parameter)
        expected["task"] = (
            gk.IndexKernel
            if task_parameter.override_transfer_learning_mode
            == TransferLearningMode.INDEX_KERNEL
            else PositiveIndexKernel
        )
    searchspace = SearchSpace.from_product(parameters)
    measurements = pd.DataFrame(
        [
            {**{p.name: p.values[i] for p in parameters}, "y": y}
            for i, y in [(0, 0.0), (-1, 1.0)]
        ]
    )
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, NumericalTarget("y").to_objective(), measurements)
    assert surrogate._model is not None
    kernel = surrogate._model.covar_module
    leaves = _leaf_kernels(kernel)

    assert isinstance(kernel, gk.ProductKernel)
    assert len(leaves) == len(expected)
    assert {tuple(k.active_dims.tolist()): type(k) for k in leaves} == {
        searchspace.get_comp_rep_parameter_indices(name): cls
        for name, cls in expected.items()
    }
    assert all(k.ard_num_dims == len(k.active_dims) for k in leaves)


@pytest.mark.parametrize(
    "override",
    [
        param(RBFKernel(), id="unscoped"),
        param(RBFKernel(parameter_names=("override",)), id="owner-scoped"),
        param(ScaleKernel(RBFKernel()), id="scale"),
        param(AdditiveKernel([RBFKernel(), MaternKernel()]), id="additive"),
        param(ProductKernel([RBFKernel(), MaternKernel()]), id="product"),
    ],
)
def test_valid_baybe_kernel_override(override):
    """Valid BayBE overrides bind all their leaves to the owning parameter."""
    parameters = [
        CategoricalParameter("base", ["a", "b"]),
        CategoricalParameter("override", ["a", "b", "c"], override_kernel=override),
    ]
    kernel, searchspace = _resolve(parameters)
    expected = set(searchspace.get_comp_rep_parameter_indices("override"))
    base_dims = set(searchspace.get_comp_rep_parameter_indices("base"))

    override_leaves = [
        k for k in _leaf_kernels(kernel) if set(k.active_dims.tolist()) != base_dims
    ]
    assert override_leaves
    assert all(set(k.active_dims.tolist()) == expected for k in override_leaves)


@pytest.mark.parametrize(
    ("categories", "override"),
    [
        param(["a", "b"], gk.RBFKernel(), id="no-ard"),
        param(["a", "b"], gk.RBFKernel(ard_num_dims=2), id="ard"),
        param(["a", "b", "c"], gk.RBFKernel(ard_num_dims=3), id="multi-column-ard"),
        param(["a", "b"], gk.ScaleKernel(gk.RBFKernel()), id="nested"),
    ],
)
def test_valid_gpytorch_kernel_override(categories, override):
    """Valid GPyTorch overrides are bound to the owner without mutating the input."""
    snapshot = deepcopy(override.state_dict())
    base = NumericalContinuousParameter("base", (0, 1))
    kernel, searchspace = _resolve(
        [base, CategoricalParameter("override", categories, override_kernel=override)]
    )
    expected = set(searchspace.get_comp_rep_parameter_indices("override"))

    # The override factor is the product factor acting on the override dimensions.
    factor = next(k for k in kernel.kernels if set(k.active_dims.tolist()) == expected)
    assert type(factor) is type(override)

    # The provided kernel is copied, not mutated.
    assert override.active_dims is None
    assert all(torch.equal(snapshot[k], override.state_dict()[k]) for k in snapshot)


def test_gpytorch_ard_mismatch_rejected():
    """A GPyTorch override with mismatched ARD size is rejected during resolution."""
    override = gk.RBFKernel(ard_num_dims=2)
    parameter = CategoricalParameter("p", ["a", "b", "c"], override_kernel=override)
    with pytest.raises(IncompatibleOverrideError, match="has 3 computational"):
        _resolve([parameter])


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        param(None, None, True, id="none"),
        param(RBFKernel(), RBFKernel(), True, id="baybe"),
        param(gk.MaternKernel(), gk.MaternKernel(), True, id="gpytorch"),
        param(
            gk.ScaleKernel(gk.RBFKernel(lengthscale_prior=GammaPrior(3.0, 6.0))),
            gk.ScaleKernel(gk.RBFKernel(lengthscale_prior=GammaPrior(3.0, 6.0))),
            True,
            id="gpytorch-nested",
        ),
        param(
            gk.IndexKernel(num_tasks=2),
            gk.IndexKernel(num_tasks=2),
            True,
            id="gpytorch-random-init",
        ),
        param(_rbf_kernel(2.0), _rbf_kernel(3.0), True, id="gpytorch-init-value"),
        param(RBFKernel(), MaternKernel(), False, id="baybe-class"),
        param(gk.MaternKernel(nu=0.5), gk.MaternKernel(), False, id="gpytorch-attr"),
        param(
            gk.RBFKernel(lengthscale_prior=GammaPrior(3.0, 6.0)),
            gk.RBFKernel(lengthscale_prior=GammaPrior(2.0, 6.0)),
            False,
            id="gpytorch-prior",
        ),
        param(
            _rbf_kernel(2.0, frozen=True),
            _rbf_kernel(3.0, frozen=True),
            False,
            id="gpytorch-frozen-value",
        ),
        param(RBFKernel(), gk.RBFKernel(), False, id="baybe-vs-gpytorch"),
        param(gk.RBFKernel(), None, False, id="gpytorch-vs-none"),
    ],
)
def test_parameter_equivalence(left, right, expected):
    """Parameter equivalence with kernel overrides produces the expected result."""
    p1 = NumericalContinuousParameter("p1", (0, 1), override_kernel=left)
    p2 = NumericalContinuousParameter("p2", (0, 1), override_kernel=right)
    assert p1.is_equivalent(p2) == expected


@pytest.mark.parametrize(
    ("values", "mode", "expected"),
    [
        param(["a", "b"], TransferLearningMode.INDEX_KERNEL, True, id="same"),
        param(["a", "b"], TransferLearningMode.POSITIVE_INDEX_KERNEL, False, id="mode"),
        param(["a", "b", "c"], TransferLearningMode.INDEX_KERNEL, False, id="values"),
    ],
)
def test_task_parameter_equivalence(values, mode, expected):
    """Task parameter equivalence produces the expected result."""
    mode_ref = TransferLearningMode.INDEX_KERNEL
    p1 = TaskParameter("t1", ["a", "b"], override_transfer_learning_mode=mode_ref)
    p2 = TaskParameter("t2", values, override_transfer_learning_mode=mode)
    assert p1.is_equivalent(p2) == expected


@pytest.mark.parametrize(
    "kernel_or_factory",
    [
        param(gk.MaternKernel(), id="raw-gpytorch-kernel"),
        param(lambda s, o, m: gk.MaternKernel(), id="factory-returning-raw-kernel"),
    ],
)
def test_incompatible_surrogate_kernel_is_rejected(kernel_or_factory):
    """Surrogate kernels that cannot exclude overridden dimensions are rejected."""
    parameters = [
        NumericalContinuousParameter("x1", (0, 1)),
        NumericalContinuousParameter("x2", (0, 1), override_kernel=RBFKernel()),
    ]

    with pytest.raises(IncompatibleOverrideError, match="can exclude these parameters"):
        _resolve(parameters, kernel_or_factory)


@pytest.mark.parametrize(
    ("override_kind", "kernel_cls", "active_dims"),
    [
        param("tl", MaternKernel, None, id="tl-unrestricted-residual"),
        param("tl", MaternKernel, (0, 1), id="tl-overlapping-residual"),
        param("tl", IndexKernel, None, id="unrestricted-task"),
        param("tl", IndexKernel, (0, 1), id="overlapping-task"),
        param("tl", IndexKernel, (), id="empty-task"),
        param("regular", MaternKernel, None, id="regular-unrestricted-residual"),
        param("regular", RBFKernel, None, id="unrestricted-override"),
    ],
)
def test_partition_validation(monkeypatch, override_kind, kernel_cls, active_dims):
    """Reject misbound regular and TL factors, matching ICM for TL partitions."""
    parameters = [
        NumericalContinuousParameter("x", (0, 1)),
        TaskParameter(
            "task",
            ["a", "b"],
            override_transfer_learning_mode=TransferLearningMode.INDEX_KERNEL,
        )
        if override_kind == "tl"
        else NumericalContinuousParameter(
            "override", (0, 1), override_kernel=RBFKernel()
        ),
    ]
    searchspace = SearchSpace.from_product(parameters)
    objective = NumericalTarget("y").to_objective()
    measurements = pd.DataFrame()

    def misbound_kernel(self, searchspace):
        return gk.RBFKernel(active_dims=active_dims)

    monkeypatch.setattr(kernel_cls, "to_gpytorch", misbound_kernel)
    with pytest.raises(ValueError, match="active_dims"):
        GaussianProcessSurrogate(kernel_or_factory=MaternKernel())._resolve_kernel(
            _ModelContext(searchspace, objective, measurements)
        )
    if override_kind == "tl":
        with pytest.raises(ValueError, match="active_dims"):
            ICMKernelFactory(
                base_kernel_or_factory=MaternKernel(parameter_names=("x",)),
                task_kernel_or_factory=IndexKernel(
                    num_tasks=2, rank=2, parameter_names=("task",)
                ),
            )(searchspace, objective, measurements)


@pytest.mark.parametrize("as_factory", [False, True], ids=["fixed", "callable"])
def test_raw_surrogate_kernel_without_overrides(as_factory):
    """Preserve raw kernels and the factory's original searchspace without overrides."""
    raw = gk.RBFKernel(active_dims=(0,))
    searchspace = SearchSpace.from_product([NumericalContinuousParameter("x", (0, 1))])

    def factory(received_searchspace, objective, measurements):
        assert received_searchspace is searchspace
        return raw

    surrogate = GaussianProcessSurrogate(
        kernel_or_factory=factory if as_factory else raw
    )
    context = _ModelContext(
        searchspace, NumericalTarget("y").to_objective(), pd.DataFrame()
    )
    assert surrogate._resolve_kernel(context) is raw
