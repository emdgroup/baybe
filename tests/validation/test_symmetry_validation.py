"""Validation tests for symmetry."""

from unittest.mock import Mock

import gpytorch
import numpy as np
import pytest
from pytest import param

from baybe.constraints import SubSelectionCondition, ThresholdCondition
from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.kernels import MaternKernel, RBFKernel
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.parameters.selectors import NameSelector
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets.baybe import BayBEKernelFactory
from baybe.symmetries import DependencySymmetry, MirrorSymmetry, PermutationSymmetry
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import create_fake_input

valid_config_mirror = {"parameter_name": "n1"}
valid_config_perm = {
    "permutation_groups": [["cat1", "cat2"], ["n1", "n2"]],
}
valid_config_dep = {
    "parameter_name": "n1",
    "condition": ThresholdCondition(0.0, ">="),
    "affected_parameter_names": ["n2", "cat1"],
}


@pytest.mark.parametrize(
    "cls, config, error, msg",
    [
        param(
            MirrorSymmetry,
            valid_config_mirror | {"mirror_point": np.inf},
            ValueError,
            "values containing infinity/nan to 'mirror_point': inf",
            id="mirror_nonfinite",
        ),
        param(
            PermutationSymmetry,
            {"permutation_groups": [["cat1", "cat1"]]},
            ValueError,
            r"the following group contains duplicates",
            id="perm_not_unique",
        ),
        param(
            PermutationSymmetry,
            {
                "permutation_groups": [
                    ["cat1", "cat2", "cat3"],
                    ["n1", "n2", "n3", "n4"],
                ]
            },
            ValueError,
            "must have the same length",
            id="perm_different_lengths",
        ),
        param(
            PermutationSymmetry,
            {"permutation_groups": [["cat1", "cat2"], ["cat1", "n2"]]},
            ValueError,
            r"following parameter names appear in several groups",
            id="perm_overlap",
        ),
        param(
            PermutationSymmetry,
            {"permutation_groups": [["cat1"]]},
            ValueError,
            "must be >= 2",
            id="perm_group_too_small",
        ),
        param(
            PermutationSymmetry,
            {"permutation_groups": []},
            ValueError,
            "must be >= 1",
            id="perm_no_groups",
        ),
        param(
            PermutationSymmetry,
            {"permutation_groups": [[1, 2]]},
            TypeError,
            "must be <class 'str'>",
            id="perm_not_str",
        ),
        param(
            DependencySymmetry,
            valid_config_dep | {"parameter_name": 1},
            TypeError,
            "must be <class 'str'>",
            id="dep_param_not_str",
        ),
        param(
            DependencySymmetry,
            valid_config_dep | {"condition": 1},
            TypeError,
            "must be <class 'baybe.constraints.conditions.Condition'>",
            id="dep_wrong_cond_type",
        ),
        param(
            DependencySymmetry,
            valid_config_dep | {"affected_parameter_names": []},
            ValueError,
            "Length of 'affected_parameter_names' must be >= 1",
            id="dep_affected_empty",
        ),
        param(
            DependencySymmetry,
            valid_config_dep | {"affected_parameter_names": [1]},
            TypeError,
            "must be <class 'str'>",
            id="dep_affected_wrong_type",
        ),
        param(
            DependencySymmetry,
            valid_config_dep | {"affected_parameter_names": ["a1", "a1"]},
            ValueError,
            r"Entries appearing multiple times: \['a1'\].",
            id="dep_affected_not_unique",
        ),
        param(
            PermutationSymmetry,
            {"permutation_groups": "abc"},
            ValueError,
            "must be a sequence of sequences, not a string",
            id="perm_groups_bare_string",
        ),
        param(
            PermutationSymmetry,
            {"permutation_groups": ["abc", "def"]},
            ValueError,
            "must be a sequence of parameter names, not a string",
            id="perm_groups_inner_bare_string",
        ),
        param(
            MirrorSymmetry,
            {"parameter_name": 123},
            TypeError,
            "must be <class 'str'>",
            id="mirror_param_not_str",
        ),
        param(
            DependencySymmetry,
            valid_config_dep | {"n_discretization_points": 3.5},
            TypeError,
            "must be <class 'int'>",
            id="dep_n_discretization_not_int",
        ),
        param(
            DependencySymmetry,
            valid_config_dep | {"n_discretization_points": 1},
            ValueError,
            "must be >= 2",
            id="dep_n_discretization_too_small",
        ),
        param(
            DependencySymmetry,
            valid_config_dep | {"affected_parameter_names": "abc"},
            ValueError,
            "must be a sequence but cannot be a string",
            id="dep_affected_bare_string",
        ),
        param(
            DependencySymmetry,
            valid_config_dep | {"affected_parameter_names": ["n1", "n2"]},
            ValueError,
            "causing parameter 'n1' cannot also be an affected parameter",
            id="dep_causing_in_affected",
        ),
    ],
)
def test_configuration(cls, config, error, msg):
    """Invalid configurations raise an expected error."""
    with pytest.raises(error, match=msg):
        cls(**config)


_parameters = [
    NumericalDiscreteParameter("n1", (-1, 0, 1)),
    NumericalDiscreteParameter("n2", (-1, 0, 1)),
    NumericalContinuousParameter("n1_not_discrete", (0.0, 10.0)),
    NumericalContinuousParameter("n2_not_discrete", (0.0, 10.0)),
    NumericalContinuousParameter("c1", (0.0, 10.0)),
    NumericalContinuousParameter("c2", (0.0, 10.0)),
    CategoricalParameter("cat1", ("a", "b", "c")),
    CategoricalParameter("cat1_altered", ("a", "b")),
    CategoricalParameter("cat2", ("a", "b", "c")),
    TaskParameter("task", ("a", "b")),
]


@pytest.fixture
def searchspace(parameter_names):
    ps = tuple(p for p in _parameters if p.name in parameter_names)
    return SearchSpace.from_product(ps)


@pytest.mark.parametrize(
    "parameter_names, mechanism, symmetry, error, msg",
    [
        param(
            ["cat1"],
            "augmentation",
            MirrorSymmetry(parameter_name="cat1"),
            TypeError,
            "'cat1' is of type 'CategoricalParameter' and is not numerical",
            id="mirror_not_numerical",
        ),
        param(
            ["n1"],
            "augmentation",
            MirrorSymmetry(parameter_name="n2"),
            IncompatibleSearchSpaceError,
            r"not present in the search space",
            id="mirror_param_missing",
        ),
        param(
            ["n2", "cat1"],
            "augmentation",
            DependencySymmetry(**valid_config_dep),
            IncompatibleSearchSpaceError,
            r"not present in the search space",
            id="dep_causing_missing",
        ),
        param(
            ["n1", "cat1"],
            "augmentation",
            DependencySymmetry(**valid_config_dep),
            IncompatibleSearchSpaceError,
            r"not present in the search space",
            id="dep_affected_missing",
        ),
        param(
            ["n1_not_discrete", "n2", "cat1"],
            "augmentation",
            DependencySymmetry(
                **valid_config_dep | {"parameter_name": "n1_not_discrete"}
            ),
            TypeError,
            "must be discrete. However, the parameter 'n1_not_discrete'",
            id="dep_causing_not_discrete",
        ),
        param(
            ["n1", "c1"],
            "augmentation",
            DependencySymmetry(
                parameter_name="n1",
                condition=ThresholdCondition(0.0, ">="),
                affected_parameter_names=["c1"],
            ),
            ValueError,
            r"n_discretization_points.*must be set explicitly",
            id="dep_continuous_no_discretization",
        ),
        param(
            ["cat1", "n1", "n2"],
            "augmentation",
            PermutationSymmetry(**valid_config_perm),
            IncompatibleSearchSpaceError,
            r"not present in the search space",
            id="perm_not_present",
        ),
        param(
            ["cat1", "cat2", "n1", "n2"],
            "augmentation",
            PermutationSymmetry(permutation_groups=[("cat1", "n1"), ("cat2", "n2")]),
            ValueError,
            r"differ in their specification",
            id="perm_inconsistent_types",
        ),
        param(
            ["cat1_altered", "cat2", "n1", "n2"],
            "augmentation",
            PermutationSymmetry(
                permutation_groups=[["cat1_altered", "cat2"], ["n1", "n2"]]
            ),
            ValueError,
            r"differ in their specification",
            id="perm_inconsistent_values",
        ),
        param(
            ["task", "n1"],
            "kernel",
            DependencySymmetry("task", SubSelectionCondition(["a"]), ["n1"]),
            IncompatibleSearchSpaceError,
            r"involves the special parameters \['task'\]",
            id="kernel_task_causing",
        ),
        param(
            ["task", "n1"],
            "kernel",
            DependencySymmetry("n1", ThresholdCondition(0.0, ">"), ["task"]),
            IncompatibleSearchSpaceError,
            r"involves the special parameters \['task'\]",
            id="kernel_task_affected",
        ),
    ],
)
def test_searchspace_context(searchspace, mechanism, symmetry, error, msg):
    """Incompatible symmetries raise an error for augmentation and invariant kernels."""
    recommender = (
        BotorchRecommender(symmetries=(symmetry,))
        if mechanism == "augmentation"
        else BotorchRecommender(
            surrogate_model=GaussianProcessSurrogate(symmetries=(symmetry,))
        )
    )
    t = NumericalTarget("t")
    measurements = create_fake_input(searchspace.parameters, [t])

    with pytest.raises(error, match=msg):
        recommender.recommend(
            1, searchspace, t.to_objective(), measurements=measurements
        )


@pytest.mark.parametrize(
    "parameter_names, gp_kwargs, error, msg",
    [
        param(
            ["n1"],
            {"symmetries": ["n1"]},
            TypeError,
            "must be <class 'baybe.symmetries.base.Symmetry'>",
            id="not_a_symmetry",
        ),
        param(
            ["n1", "n2"],
            {
                "symmetries": [
                    PermutationSymmetry([["n1", "n2"]]),
                    MirrorSymmetry("n1"),
                ]
            },
            ValueError,
            r"controlled by several symmetries: \['n1'\]",
            id="overlap_perm_mirror",
        ),
        param(
            ["n1", "cat1"],
            {
                "symmetries": [
                    DependencySymmetry("cat1", SubSelectionCondition(["a"]), ["n1"]),
                    MirrorSymmetry("n1"),
                ]
            },
            ValueError,
            r"controlled by several symmetries: \['n1'\]",
            id="overlap_dep_affected_mirror",
        ),
        param(
            ["n1", "n2", "c1"],
            {
                "symmetries": [
                    DependencySymmetry("n1", ThresholdCondition(0.0, ">"), ["c1"]),
                    PermutationSymmetry([["n1", "n2"]]),
                ]
            },
            ValueError,
            "causing parameter 'n1' .* cannot be controlled by another symmetry",
            id="causing_permuted",
        ),
        param(
            ["n1", "n2", "cat1"],
            {
                "symmetries": [
                    DependencySymmetry("n1", ThresholdCondition(0.0, ">"), ["n2"]),
                    DependencySymmetry("n2", ThresholdCondition(0.0, ">"), ["cat1"]),
                ]
            },
            ValueError,
            "causing parameter 'n2' .* cannot be controlled by another symmetry",
            id="dependency_chain",
        ),
        param(
            ["n1"],
            {"symmetries": [PermutationSymmetry([[f"p{i}" for i in range(6)]])]},
            ValueError,
            "at most 5 positions, but a group with 6",
            id="perm_group_too_large",
        ),
        param(
            ["n1", "n2"],
            {
                "kernel_or_factory": MaternKernel(parameter_names=["n1"])
                * RBFKernel(parameter_names=["n2"]),
                "symmetries": [PermutationSymmetry([["n1", "n2"]])],
            },
            ValueError,
            "cannot be made invariant",
            id="permuted_kernels_differ",
        ),
        param(
            ["n1", "n2", "cat1"],
            {
                "kernel_or_factory": BayBEKernelFactory(
                    parameter_selector=NameSelector(["n1", "cat1"])
                ),
                "symmetries": [PermutationSymmetry([["n1", "n2"]])],
            },
            ValueError,
            "cannot be made invariant",
            id="selector_excludes_permuted",
        ),
        param(
            ["n1", "n2"],
            {
                "mean_or_factory": gpytorch.means.LinearMean(input_size=2),
                "symmetries": [MirrorSymmetry("n1")],
            },
            ValueError,
            "require a constant mean function",
            id="input_dependent_mean",
        ),
    ],
)
def test_gp_configuration(monkeypatch, searchspace, gp_kwargs, error, msg):
    """Invalid symmetry setups of a Gaussian process raise an error before fitting."""
    fit = Mock()
    monkeypatch.setattr("botorch.fit.fit_gpytorch_mll", fit)
    t = NumericalTarget("t")
    measurements = create_fake_input(searchspace.parameters, [t], n_rows=3)

    with pytest.raises(error, match=msg):
        GaussianProcessSurrogate(**gp_kwargs).fit(
            searchspace, t.to_objective(), measurements
        )
    fit.assert_not_called()
