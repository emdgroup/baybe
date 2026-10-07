"""Tests for symmetry-invariant Gaussian process kernels."""

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.optim.utils import get_parameters_and_bounds
from pytest import param

from baybe.constraints import SubSelectionCondition
from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.kernels import RBFKernel
from baybe.parameters import (
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process._symmetry import make_symmetric_kernel
from baybe.surrogates.gaussian_process.core import _ModelContext
from baybe.symmetries import DependencySymmetry, MirrorSymmetry, PermutationSymmetry
from baybe.targets import NumericalTarget

_PARAMETERS = {
    p.name: p
    for p in [
        NumericalDiscreteParameter("x1", (0, 1, 2, 3)),
        NumericalDiscreteParameter("x2", (0, 1, 2, 3)),
        NumericalDiscreteParameter("o1", (0, 1, 2, 3), override_kernel=RBFKernel()),
        NumericalDiscreteParameter("o2", (0, 1, 2, 3), override_kernel=RBFKernel()),
        CategoricalParameter("s1", ("a", "b", "c")),
        CategoricalParameter("s2", ("a", "b", "c")),
        NumericalDiscreteParameter("f1", (0, 50, 100)),
        NumericalDiscreteParameter("f2", (0, 50, 100)),
        NumericalContinuousParameter("m", (0, 1)),
        CategoricalParameter("c", ("off1", "off2", "on")),
        NumericalDiscreteParameter("y", (0, 5, 10)),
        NumericalContinuousParameter("yc", (0, 10)),
        NumericalContinuousParameter("z", (0, 1)),
        CustomDiscreteParameter(
            "custom",
            pd.DataFrame(
                {"a": [0.0, 0.0, 1.0], "b": [0.0, 1.0, 1.0]}, index=list("ABC")
            ),
            decorrelate=0.5,  # drops column "b", so that "A" and "B" become identical
        ),
    ]
}

_PERMUTATION = PermutationSymmetry([["x1", "x2"]])
_MIRROR = MirrorSymmetry("m", mirror_point=0.3)
_DEPENDENCY = DependencySymmetry(
    "c", SubSelectionCondition(["on"]), ["y", "yc"], n_discretization_points=3
)

_CASES = [
    param(["x1", "x2", "z"], [_PERMUTATION], id="perm_joint_kernel"),
    param(
        ["s1", "s2", "f1", "f2", "z"],
        [PermutationSymmetry([["s1", "s2"], ["f1", "f2"]])],
        id="perm_lockstep_onehot",
    ),
    param(
        ["o1", "o2", "z"], [PermutationSymmetry([["o1", "o2"]])], id="perm_overrides"
    ),
    param(["m", "z"], [_MIRROR], id="mirror"),
    param(["c", "y", "yc", "z"], [_DEPENDENCY], id="dependency"),
    param(
        ["x1", "x2", "m", "c", "y", "yc", "z"],
        [_PERMUTATION, _MIRROR, _DEPENDENCY],
        id="combined",
    ),
]


def _make_kernels(searchspace, symmetries):
    """Make the symmetric kernel, its unsymmetrized counterpart and the transform."""
    context = _ModelContext(
        searchspace, NumericalTarget("t").to_objective(), pd.DataFrame()
    )
    gp = GaussianProcessSurrogate()
    transform = gp._make_input_transform(context)
    kernel = make_symmetric_kernel(
        symmetries, searchspace, transform, gp._resolve_kernel(context)
    )
    return kernel, gp._resolve_kernel(context), transform


def _randomize(kernel):
    """Assign random hyperparameters, preserving shared ones."""
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in kernel.parameters():
            p.copy_(0.5 + torch.rand(p.shape, generator=generator, dtype=p.dtype))
    return kernel


def _make_inputs(searchspace, symmetries, transform, n_rows):
    """Make random inputs and their symmetry-equivalent counterparts.

    The counterparts are created via data augmentation, which keeps the index of the
    original row, so that all rows sharing an index are equivalent.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            p.name: rng.choice(p.values, n_rows)
            if p.is_discrete
            else rng.uniform(*p.bounds.to_tuple(), n_rows)
            for p in searchspace.parameters
        }
    )
    for symmetry in symmetries:
        df = symmetry.augment_measurements(df, searchspace)
    comp = torch.tensor(searchspace.transform(df).to_numpy(), dtype=torch.float64)
    return df.index.to_numpy(), transform.transform(comp)


def _is_invariant(K, groups):
    """Check if the kernel rows of all equivalent inputs are identical."""
    return all(
        torch.allclose(K[..., groups == g, :], K[..., [np.argmax(groups == g)], :])
        for g in np.unique(groups)
    )


@pytest.mark.parametrize(("parameter_names", "symmetries"), _CASES)
def test_symmetric_kernel_properties(parameter_names, symmetries):
    """Symmetric kernels are valid and invariant, unlike unsymmetrized kernels."""
    searchspace = SearchSpace.from_product([_PARAMETERS[n] for n in parameter_names])
    kernel, base, transform = _make_kernels(searchspace, symmetries)
    kernel, base = _randomize(kernel), _randomize(base)
    groups, x = _make_inputs(searchspace, symmetries, transform, n_rows=4)
    _, reference = _make_inputs(searchspace, [], transform, n_rows=3)

    # Batched inputs to also cover batch shapes
    x, reference = torch.stack([x, x]), torch.stack([reference, reference.flip(0)])
    with torch.no_grad():
        K = kernel(x, x).to_dense()
        assert _is_invariant(kernel(x, reference).to_dense(), groups)
        assert not _is_invariant(base(x, reference).to_dense(), groups)
        assert torch.allclose(K, K.transpose(-1, -2))
        assert torch.linalg.eigvalsh(K).min() >= -1e-8 * K.abs().max()
        assert torch.allclose(kernel(x, x, diag=True), K.diagonal(dim1=-2, dim2=-1))


def test_tied_hyperparameters_keep_bounds():
    """Hyperparameters tied for permutation invariance keep their bounds."""
    searchspace = SearchSpace.from_product([_PARAMETERS[n] for n in ["x1", "x2"]])
    kernel, _, _ = _make_kernels(searchspace, [_PERMUTATION])
    _, bounds = get_parameters_and_bounds(kernel)
    assert bounds["base_kernel.parametrizations.raw_lengthscale.original"][0] > 0


@pytest.mark.parametrize(
    ("causing", "condition", "error", "match"),
    [
        param(
            "c",
            SubSelectionCondition(["on"]),
            ValueError,
            "does not match any of the parameter's values",
            id="unknown_encoding",
        ),
        param(
            "custom",
            SubSelectionCondition(["A"]),
            IncompatibleSearchSpaceError,
            "identical computational representations",
            id="ambiguous_encoding",
        ),
    ],
)
def test_dependency_activity_must_be_identifiable(causing, condition, error, match):
    """The activity of a dependency must be determinable from the model inputs."""
    searchspace = SearchSpace.from_product([_PARAMETERS[n] for n in [causing, "y"]])
    symmetry = DependencySymmetry(causing, condition, ["y"])
    x = torch.full((2, len(searchspace.comp_rep_columns)), 0.5, dtype=torch.float64)
    with pytest.raises(error, match=match):
        kernel, _, _ = _make_kernels(searchspace, [symmetry])
        kernel(x, x).to_dense()
