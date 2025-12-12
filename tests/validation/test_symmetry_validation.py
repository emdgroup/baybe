"""Validation tests for symmetry."""

import numpy as np
import pytest
from pytest import param

from baybe.constraints import ThresholdCondition
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
    "parameter_names": ["cat1", "cat2"],
    "copermuted_groups": [["n1", "n2"]],
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
            "values containing infinity/nan to attribute 'mirror_point': inf",
            id="mirror_nonfinite",
        ),
        param(
            PermutationSymmetry,
            {"parameter_names": ["cat1", "cat1"]},
            ValueError,
            r"Entries appearing multiple times: \['cat1'\].",
            id="perm_primary_not_unique",
        ),
        param(
            PermutationSymmetry,
            valid_config_perm | {"copermuted_groups": [["n1", "n1"]]},
            ValueError,
            r"the following group contains duplicates: \['n1', 'n1'\]",
            id="perm_copermuted_not_unique",
        ),
        param(
            PermutationSymmetry,
            valid_config_perm | {"copermuted_groups": [["n1"]]},
            ValueError,
            "must have the same length as the primary",
            id="perm_group_short",
        ),
        param(
            PermutationSymmetry,
            valid_config_perm | {"copermuted_groups": [["n1", "n2", "n3"]]},
            ValueError,
            "must have the same length as the primary",
            id="perm_group_long",
        ),
        param(
            PermutationSymmetry,
            valid_config_perm | {"copermuted_groups": [["cat1", "n2"]]},
            ValueError,
            r"following parameter names appear in several groups \{'cat1'\}",
            id="perm_overlap",
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
            "'cat1' is of type CategoricalParameter and is not numerical",
            id="mirror_not_numerical",
        ),
        param(
            ["n1"],
            MirrorSymmetry(parameter_name="n2"),
            ValueError,
            r"parameter which is not present in the searchspace: \{'n2'\}",
            id="mirror_param_missing",
        ),
        param(
            ["n2", "cat1"],
            DependencySymmetry(**valid_config_dep),
            ValueError,
            r"not present in the searchspace: \{'n1'\}",
            id="dep_causing_missing",
        ),
        param(
            ["n1", "cat1"],
            DependencySymmetry(**valid_config_dep),
            ValueError,
            r"not present in the searchspace: \{'n2'\}",
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
            ValueError,
            r"which is not present in the searchspace: \{'cat2'\}",
            id="perm_primary_not_present",
        ),
        param(
            ["cat1", "cat2", "n1"],
            PermutationSymmetry(**valid_config_perm),
            ValueError,
            "which is not present in the searchspace: {'n2'}",
            id="perm_copermuted_not_present",
        ),
        param(
            ["cat1", "cat2", "n1", "n2"],
            PermutationSymmetry(
                parameter_names=("cat1", "n1"), copermuted_groups=(("cat2", "n2"),)
            ),
            TypeError,
            r"multiple types were found in the permutation group \('cat1', 'n1'\):",
            id="perm_inconsistent_types",
        ),
        param(
            ["cat1_altered", "cat2", "n1", "n2"],
            PermutationSymmetry(
                **valid_config_perm | {"parameter_names": ["cat1_altered", "cat2"]}
            ),
            ValueError,
            r"group '\('cat1_altered', 'cat2'\)' contains parameters which have",
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
