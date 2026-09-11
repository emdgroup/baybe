"""Validation tests for search spaces."""

import pandas as pd
import pytest
from pytest import param

from baybe.constraints import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
    DiscreteSumConstraint,
    ThresholdCondition,
)
from baybe.constraints.discrete import DiscreteLinkedParametersConstraint
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace, SubspaceContinuous, SubspaceDiscrete
from baybe.utils.dataframe import get_transform_objects

parameters = [NumericalDiscreteParameter("d1", [0, 1])]


@pytest.mark.parametrize(
    ("df", "match"),
    [
        param(
            pd.DataFrame(columns=[]),
            r"object\(s\) named \{'d1'\} cannot be matched",
            id="missing",
        ),
        param(
            pd.DataFrame(columns=["d1", "d2"]),
            r"column\(s\) \{'d2'\} cannot be matched",
            id="extra",
        ),
    ],
)
def test_invalid_transforms(df, match):
    """Transforming dataframes with incorrect columns raises an error."""
    with pytest.raises(ValueError, match=match):
        get_transform_objects(df, parameters, allow_missing=False, allow_extra=False)


@pytest.mark.parametrize(
    ("df", "missing", "extra"),
    [
        param(pd.DataFrame(columns=["d1"]), False, False, id="equal"),
        param(pd.DataFrame(columns=[]), True, False, id="missing"),
        param(pd.DataFrame(columns=["d1", "d2"]), False, True, id="extra"),
    ],
)
def test_valid_transforms(df, missing, extra):
    """When providing the appropriate flags, the columns of the dataframe to be transformed can be flexibly chosen."""  # noqa
    get_transform_objects(df, parameters, allow_missing=missing, allow_extra=extra)


def test_invalid_constraint_parameter_combos():
    """Testing invalid constraint-parameter combinations."""
    parameters = [
        CategoricalParameter("cat1", values=("c1", "c2")),
        NumericalDiscreteParameter("d1", values=[1, 2, 3]),
        NumericalDiscreteParameter("d2", values=[0, 1, 2]),
        NumericalContinuousParameter("c1", (0, 2)),
        NumericalContinuousParameter("c2", (-1, 1)),
    ]

    # Attempting continuous constraint over hybrid parameter set
    with pytest.raises(ValueError):
        SearchSpace.from_product(
            parameters=parameters,
            constraints=[ContinuousLinearConstraint(["c1", "c2", "d1"], "=")],
        )

    # Attempting discrete constraint over hybrid parameter set
    with pytest.raises(ValueError):
        SearchSpace.from_product(
            parameters=parameters,
            constraints=[
                DiscreteSumConstraint(
                    parameters=["d1", "d2", "c1"],
                    condition=ThresholdCondition(threshold=1.0, operator=">"),
                )
            ],
        )

    # Attempting constraints over parameter set where a parameter does not exist
    with pytest.raises(ValueError):
        SearchSpace.from_product(
            parameters=parameters,
            constraints=[
                DiscreteSumConstraint(
                    parameters=["d1", "e7", "c1"],
                    condition=ThresholdCondition(threshold=1.0, operator=">"),
                )
            ],
        )

    # Attempting constraints over parameter set where a parameter does not exist
    with pytest.raises(ValueError):
        SearchSpace.from_product(
            parameters=parameters,
            constraints=[ContinuousLinearConstraint(["c1", "e7", "d1"], "=")],
        )

    # Attempting constraints over parameter sets containing non-numerical discrete
    # parameters.
    with pytest.raises(
        ValueError, match="valid only for numerical discrete parameters"
    ):
        SearchSpace.from_product(
            parameters=parameters,
            constraints=[
                DiscreteSumConstraint(
                    parameters=["cat1", "d1", "d2"],
                    condition=ThresholdCondition(threshold=1.0, operator=">"),
                )
            ],
        )


def test_cardinality_constraints_with_overlapping_parameters():
    """Creating cardinality constraints with overlapping parameters raises an error."""
    parameters = (
        NumericalContinuousParameter("c1", (0, 1)),
        NumericalContinuousParameter("c2", (0, 1)),
        NumericalContinuousParameter("c3", (0, 1)),
    )
    with pytest.raises(ValueError, match="cannot share the same parameters"):
        SubspaceContinuous(
            parameters=parameters,
            constraints=(
                ContinuousCardinalityConstraint(
                    parameters=["c1", "c2"],
                    max_cardinality=1,
                ),
                ContinuousCardinalityConstraint(
                    parameters=["c2", "c3"],
                    max_cardinality=1,
                ),
            ),
        )


def test_cardinality_constraint_with_invalid_parameter_bounds():
    """Imposing a cardinality constraint on a parameter whose range does not include
    zero raises an error."""  # noqa
    parameters = (
        NumericalContinuousParameter("c1", (0, 1)),
        NumericalContinuousParameter("c2", (1, 2)),
    )
    with pytest.raises(ValueError, match="must include zero"):
        SubspaceContinuous(
            parameters=parameters,
            constraints=(
                ContinuousCardinalityConstraint(
                    parameters=["c1", "c2"],
                    max_cardinality=1,
                ),
            ),
        )


p_cont = NumericalContinuousParameter("p", (0, 1))
p_disc = NumericalDiscreteParameter("p", (0, 1))


@pytest.mark.parametrize(
    ("p1", "p2", "space"),
    [
        param(p_cont, p_cont, SubspaceContinuous, id="continuous"),
        param(p_disc, p_disc, SubspaceDiscrete, id="discrete"),
        param(p_cont, p_disc, SearchSpace, id="hybrid"),
    ],
)
def test_subspace_with_duplicate_parameter_names(p1, p2, space):
    """Creating a search space with duplicate parameter names raises an error."""
    with pytest.raises(ValueError, match="unique names"):
        space.from_product(parameters=[p1, p2])


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize(
    "referenced",
    [
        param(["nonexistent"], id="all_nonexistent"),
        param(["p1", "nonexistent"], id="partially_nonexistent"),
    ],
)
def test_continuous_subspace_constraint_with_nonexistent_params(referenced, discrete):
    """Using constraints referencing nonexistent parameters raises an error."""
    if discrete:
        parameters = [
            NumericalDiscreteParameter("p1", (0, 1)),
            NumericalDiscreteParameter("p2", (0, 1)),
        ]
        space = SubspaceDiscrete
        constraint = DiscreteLinkedParametersConstraint(referenced)
    else:
        parameters = [
            NumericalContinuousParameter("p1", (0, 1)),
            NumericalContinuousParameter("p2", (0, 1)),
        ]
        space = SubspaceContinuous
        constraint = ContinuousLinearConstraint(referenced, "=")

    with pytest.raises(ValueError, match="does not exist"):
        space.from_product(parameters=parameters, constraints=[constraint])


def test_invalid_simplex_creation_with_overlapping_parameters():
    """Creating a simplex searchspace with overlapping simplex and product parameters
    raises an error."""  # noqa
    parameters = [NumericalDiscreteParameter(name="x_1", values=(0, 1, 2))]

    with pytest.raises(
        ValueError,
        match="'simplex_parameters' and 'product_parameters' must be disjoint",
    ):
        SearchSpace(
            SubspaceDiscrete.from_simplex(
                max_sum=1.0,
                simplex_parameters=parameters,
                product_parameters=parameters,
            )
        )
