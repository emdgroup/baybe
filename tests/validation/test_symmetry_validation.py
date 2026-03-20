"""Validation tests for symmetry."""

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.constraints import ThresholdCondition
from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
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
            PermutationSymmetry,
            {"permutation_groups": [["a", "b"]], "use_data_augmentation": 1},
            TypeError,
            "must be <class 'bool'>",
            id="use_aug_not_bool",
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
]


@pytest.fixture
def searchspace(parameter_names):
    ps = tuple(p for p in _parameters if p.name in parameter_names)
    return SearchSpace.from_product(ps)


@pytest.mark.parametrize(
    "parameter_names, symmetry, error, msg",
    [
        param(
            ["cat1"],
            MirrorSymmetry(parameter_name="cat1"),
            TypeError,
            "'cat1' is of type 'CategoricalParameter' and is not numerical",
            id="mirror_not_numerical",
        ),
        param(
            ["n1"],
            MirrorSymmetry(parameter_name="n2"),
            IncompatibleSearchSpaceError,
            r"not present in the search space",
            id="mirror_param_missing",
        ),
        param(
            ["n2", "cat1"],
            DependencySymmetry(**valid_config_dep),
            IncompatibleSearchSpaceError,
            r"not present in the search space",
            id="dep_causing_missing",
        ),
        param(
            ["n1", "cat1"],
            DependencySymmetry(**valid_config_dep),
            IncompatibleSearchSpaceError,
            r"not present in the search space",
            id="dep_affected_missing",
        ),
        param(
            ["n1_not_discrete", "n2", "cat1"],
            DependencySymmetry(
                **valid_config_dep | {"parameter_name": "n1_not_discrete"}
            ),
            TypeError,
            "must be discrete. However, the parameter 'n1_not_discrete'",
            id="dep_causing_not_discrete",
        ),
        param(
            ["cat1", "n1", "n2"],
            PermutationSymmetry(**valid_config_perm),
            IncompatibleSearchSpaceError,
            r"not present in the search space",
            id="perm_not_present",
        ),
        param(
            ["cat1", "cat2", "n1", "n2"],
            PermutationSymmetry(permutation_groups=[("cat1", "n1"), ("cat2", "n2")]),
            ValueError,
            r"differ in their specification",
            id="perm_inconsistent_types",
        ),
        param(
            ["cat1_altered", "cat2", "n1", "n2"],
            PermutationSymmetry(
                permutation_groups=[["cat1_altered", "cat2"], ["n1", "n2"]]
            ),
            ValueError,
            r"differ in their specification",
            id="perm_inconsistent_values",
        ),
    ],
)
def test_searchspace_context(searchspace, symmetry, error, msg):
    """Configurations not compatible with the searchspace raise an expected error."""
    recommender = BotorchRecommender(
        surrogate_model=GaussianProcessSurrogate(symmetries=(symmetry,))
    )
    t = NumericalTarget("t")
    measurements = create_fake_input(searchspace.parameters, [t])

    with pytest.raises(error, match=msg):
        recommender.recommend(
            1, searchspace, t.to_objective(), measurements=measurements
        )


def test_dependency_augmentation_requires_parameters():
    """DependencySymmetry.augment_measurements raises when parameters is None."""
    s = DependencySymmetry(**valid_config_dep)
    df = pd.DataFrame({"n1": [0], "n2": [1], "cat1": ["a"]})
    with pytest.raises(ValueError, match="requires parameter objects"):
        s.augment_measurements(df)


def test_surrogate_rejects_non_symmetry():
    """Surrogate.symmetries rejects non-Symmetry members."""
    with pytest.raises(TypeError, match="must be <class"):
        GaussianProcessSurrogate(symmetries=("not_a_symmetry",))
