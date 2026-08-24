"""Functional tests for parameter-specific kernel overrides.

These tests resolve or fit kernels to check binding and composition behavior.
Construction-time input validation lives in ``tests/validation``.
"""

from copy import deepcopy

import pandas as pd
import pytest
import torch
from botorch.models.kernels.positive_index import PositiveIndexKernel
from gpytorch import kernels as gk
from pytest import param

from baybe.exceptions import IncompatibleOverrideError
from baybe.kernels import MaternKernel, RBFKernel
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


def _fit(parameters):
    """Fit a GP and return its installed covariance module and searchspace."""
    searchspace = SearchSpace.from_product(parameters)
    measurements = pd.DataFrame(
        [
            {**{p.name: p.values[0] for p in parameters}, "y": 0.0},
            {**{p.name: p.values[-1] for p in parameters}, "y": 1.0},
        ]
    )
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, NumericalTarget("y").to_objective(), measurements)
    assert surrogate._model is not None
    return surrogate._model.covar_module, searchspace


def test_default_factory_selector_is_preserved():
    """Partitioning preserves an explicit default-factory parameter selector."""
    kernel, _ = _resolve(
        [
            NumericalContinuousParameter("x1", (0, 1)),
            NumericalContinuousParameter("x2", (0, 1), kernel_override=RBFKernel()),
            NumericalContinuousParameter("x3", (0, 1)),
        ],
        BayBEKernelFactory(parameter_selector=NameSelector(("x1",), regex=False)),
    )

    assert {tuple(k.active_dims.tolist()) for k in _leaf_kernels(kernel)} == {
        (0,),
        (1,),
    }


@pytest.mark.parametrize(
    ("parameters", "expected_types", "expected_names"),
    [
        param(
            [
                CategoricalParameter("base", ["a", "b"]),
                NumericalDiscreteParameter(
                    "override", [0, 1, 2], kernel_override=RBFKernel()
                ),
            ],
            [gk.MaternKernel, gk.RBFKernel],
            ["base", "override"],
            id="single-dim",
        ),
        param(
            [
                CategoricalParameter("base", ["a", "b"]),
                CategoricalParameter(
                    "override", ["a", "b", "c"], kernel_override=RBFKernel()
                ),
            ],
            [gk.MaternKernel, gk.RBFKernel],
            ["base", "override"],
            id="multi-dim",
        ),
        param(
            [
                CategoricalParameter("first", ["a", "b"], kernel_override=RBFKernel()),
                CategoricalParameter(
                    "second", ["a", "b"], kernel_override=MaternKernel()
                ),
            ],
            [gk.RBFKernel, gk.MaternKernel],
            ["first", "second"],
            id="all-overridden",
        ),
        param(
            [
                CategoricalParameter("base", ["a", "b"]),
                CategoricalParameter(
                    "override", ["a", "b"], kernel_override=RBFKernel()
                ),
                TaskParameter("task", ["a", "b"]),
            ],
            [
                gk.MaternKernel,
                PositiveIndexKernel,
                gk.RBFKernel,
            ],
            ["base", "task", "override"],
            id="task-without-tl-override",
        ),
        param(
            [
                CategoricalParameter("base", ["a", "b"]),
                CategoricalParameter(
                    "override", ["a", "b", "c"], kernel_override=RBFKernel()
                ),
                TaskParameter(
                    "task",
                    ["a", "b"],
                    override_transfer_learning_mode=TransferLearningMode.INDEX_KERNEL,
                ),
            ],
            [
                gk.MaternKernel,
                gk.RBFKernel,
                gk.IndexKernel,
            ],
            ["base", "override", "task"],
            id="override-and-tl-index",
        ),
        param(
            [
                CategoricalParameter("base", ["a", "b"]),
                CategoricalParameter(
                    "override", ["a", "b", "c"], kernel_override=RBFKernel()
                ),
                TaskParameter(
                    "task",
                    ["a", "b"],
                    override_transfer_learning_mode=(
                        TransferLearningMode.POSITIVE_INDEX_KERNEL
                    ),
                ),
            ],
            [
                gk.MaternKernel,
                gk.RBFKernel,
                PositiveIndexKernel,
            ],
            ["base", "override", "task"],
            id="override-and-tl-pos-index",
        ),
    ],
)
def test_fitted_model_uses_parameter_kernel_overrides(
    parameters, expected_types, expected_names
):
    """The fitted model uses each configured kernel on the intended dimensions."""
    kernel, searchspace = _fit(parameters)
    leaves = _leaf_kernels(kernel)
    expected_dimensions = [
        searchspace.get_comp_rep_parameter_indices(name) for name in expected_names
    ]

    assert isinstance(kernel, gk.ProductKernel)
    assert [type(k) for k in leaves] == expected_types
    assert [tuple(k.active_dims.tolist()) for k in leaves] == expected_dimensions
    assert [k.ard_num_dims for k in leaves] == [len(d) for d in expected_dimensions]


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
        CategoricalParameter("override", ["a", "b", "c"], kernel_override=override),
    ]
    kernel, searchspace = _resolve(parameters)
    expected = set(searchspace.get_comp_rep_parameter_indices("override"))

    override_leaves = [
        k
        for k in _leaf_kernels(kernel)
        if set(k.active_dims.tolist())
        != set(searchspace.get_comp_rep_parameter_indices("base"))
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
        [base, CategoricalParameter("override", categories, kernel_override=override)]
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
    parameter = CategoricalParameter("p", ["a", "b", "c"], kernel_override=override)
    with pytest.raises(IncompatibleOverrideError, match="has 3 computational"):
        _resolve([parameter])


def test_parameter_equivalence_ignores_override_parameter_name():
    """Equivalent parameters may scope the same override to their own names."""
    left = NumericalContinuousParameter(
        "left", (0, 1), kernel_override=RBFKernel(parameter_names=("left",))
    )
    right = NumericalContinuousParameter(
        "right", (0, 1), kernel_override=RBFKernel(parameter_names=("right",))
    )

    assert left.is_equivalent(right)


@pytest.mark.parametrize(
    "kernel_or_factory",
    [
        param(
            AdditiveKernel([MaternKernel(), MaternKernel()]),
            id="non-reducible-baybe-kernel",
        ),
        param(gk.MaternKernel(), id="raw-gpytorch-kernel"),
        param(
            lambda s, o, m: gk.MaternKernel(),
            id="factory-returning-raw-kernel",
        ),
    ],
)
def test_incompatible_surrogate_kernel_is_rejected(kernel_or_factory):
    """Surrogate kernels that cannot exclude overridden dimensions are rejected."""
    parameters = [
        NumericalContinuousParameter("x1", (0, 1)),
        NumericalContinuousParameter("x2", (0, 1), kernel_override=RBFKernel()),
    ]

    with pytest.raises(IncompatibleOverrideError):
        _resolve(parameters, kernel_or_factory)
