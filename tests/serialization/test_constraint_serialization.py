"""Constraint serialization tests."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from tests.hypothesis_strategies.constraints import (
    continuous_linear_constraints,
    discrete_dependencies_constraints,
    discrete_excludes_constraints,
    discrete_linked_parameters_constraints,
    discrete_no_label_duplicates_constraints,
    discrete_permutation_invariance_constraints,
    discrete_product_constraints,
    discrete_sum_constraints,
)
from tests.serialization.utils import assert_roundtrip_consistency


@pytest.mark.parametrize(
    "strategy",
    [
        param(
            discrete_permutation_invariance_constraints(),
            id="DiscretePermutationInvarianceConstraint",
        ),
        param(discrete_dependencies_constraints(), id="DiscreteDependenciesConstraint"),
        param(discrete_excludes_constraints(), id="DiscreteExcludeConstraint"),
        param(discrete_sum_constraints(), id="DiscreteSumConstraint"),
        param(discrete_product_constraints(), id="DiscreteProductConstraint"),
        param(
            discrete_no_label_duplicates_constraints(),
            id="DiscreteNoLabelDuplicatesConstraint",
        ),
        param(
            discrete_linked_parameters_constraints(),
            id="DiscreteLinkedParametersConstraint",
        ),
        param(
            continuous_linear_constraints(),
            id="ContinuousLinearConstraint",
        ),
    ],
)
@given(data=st.data())
def test_roundtrip(strategy: st.SearchStrategy, data: st.DataObject):
    """A serialization roundtrip yields an equivalent object."""
    constraint = data.draw(strategy)
    assert_roundtrip_consistency(constraint)


@pytest.mark.parametrize(
    "constraint_names",
    [["Constraint_13"]],
)
def test_unsupported_serialization_of_custom_constraint(constraints):
    with pytest.raises(NotImplementedError):
        constraint = constraints[0]
        constraint.to_json()
