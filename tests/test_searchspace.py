"""Tests for the searchspace module."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from baybe._optional.info import POLARS_INSTALLED
from baybe.constraints import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
    DiscreteSumConstraint,
    ThresholdCondition,
)
from baybe.exceptions import EmptySearchSpaceError, IncompatibilityError
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
from baybe.searchspace.discrete import (
    parameter_cartesian_prod_pandas,
    parameter_cartesian_prod_polars,
)
from baybe.utils.basic import is_all_instance

try:
    ExceptionGroup
except NameError:
    from exceptiongroup import ExceptionGroup


def test_empty_parameters():
    """Creation of a search space with no parameters raises an exception."""
    with pytest.raises(EmptySearchSpaceError):
        SearchSpace()


def test_bounds_order():
    """Asserts that the bounds are created in the correct order.

    The correct order is discrete parameters first, continuous next.
    """
    parameters = [
        NumericalDiscreteParameter(name="A_disc", values=[1.0, 2.0, 3.0]),
        NumericalContinuousParameter(name="A_cont", bounds=(4.0, 6.0)),
        NumericalDiscreteParameter(name="B_disc", values=[7.0, 8.0, 9.0]),
        NumericalContinuousParameter(name="B_cont", bounds=(10.0, 12.0)),
    ]
    searchspace = SearchSpace.from_product(parameters=parameters)
    expected = np.array([[1.0, 7.0, 4.0, 10.0], [3.0, 9.0, 6.0, 12.0]])
    assert np.array_equal(
        searchspace.comp_rep_bounds.values,
        expected,
    )


def test_empty_parameter_bounds():
    """Asserts that the correct bounds are produced for empty search spaces.

    Also checks for the correct shapes.
    """
    parameters = []
    searchspace_discrete = SubspaceDiscrete.from_product(parameters=parameters)
    searchspace_continuous = SubspaceContinuous(parameters=parameters)
    expected = pd.DataFrame(np.empty((2, 0)), index=["min", "max"])
    pd.testing.assert_frame_equal(searchspace_discrete.comp_rep_bounds, expected)
    pd.testing.assert_frame_equal(searchspace_continuous.comp_rep_bounds, expected)


def test_discrete_searchspace_creation_from_dataframe():
    """A purely discrete search space is created from an example dataframe."""
    num_specified = NumericalDiscreteParameter(name="num_specified", values=[1, 2, 3])
    num_unspecified = NumericalDiscreteParameter(
        name="num_unspecified", values=[4, 5, 6]
    )
    cat_specified = CategoricalParameter(name="cat_specified", values=["a", "b", "c"])
    cat_unspecified = CategoricalParameter(
        name="cat_unspecified", values=["d", "e", "f"]
    )

    all_params = (cat_specified, cat_unspecified, num_specified, num_unspecified)

    df = pd.DataFrame({param.name: param.values for param in all_params})
    searchspace = SearchSpace(
        SubspaceDiscrete.from_dataframe(df, parameters=[num_specified, cat_specified])
    )

    assert searchspace.type == SearchSpaceType.DISCRETE
    assert searchspace.parameters == all_params
    assert df.equals(searchspace.discrete.exp_rep)


def test_discrete_from_dataframe_dtype_consistency():
    """Inconsistent but valid dtypes are correctly converted."""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "C": [int(1), 2.2, True],  # Valid but inconsistent dtypes
        }
    )
    subspace = SubspaceDiscrete.from_dataframe(df)

    assert isinstance(
        next(p for p in subspace.parameters if p.name == "C"),
        NumericalDiscreteParameter,
    )
    assert pd.api.types.is_float_dtype(subspace.exp_rep["C"])


def test_invalid_simplex_creating_with_overlapping_parameters():
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


def test_continuous_searchspace_creation_from_bounds():
    """A purely continuous search space is created from example bounds."""
    parameters = (
        NumericalContinuousParameter("param1", (0, 1)),
        NumericalContinuousParameter("param2", (-1, 1)),
    )
    bounds = pd.DataFrame({p.name: p.bounds.to_tuple() for p in parameters})
    searchspace = SearchSpace(continuous=SubspaceContinuous.from_bounds(bounds))

    assert searchspace.type == SearchSpaceType.CONTINUOUS
    assert searchspace.parameters == parameters


def test_hyperrectangle_searchspace_creation():
    """A purely continuous search space is created that spans a certain set of points.

    As the name suggests, this searchspace is hyperrectangle- shaped
    """
    points = pd.DataFrame(
        {
            "param1": [0, 1, 2],
            "param2": [-1, 0, 1],
        }
    )
    searchspace = SearchSpace(continuous=SubspaceContinuous.from_dataframe(points))

    parameters = (
        NumericalContinuousParameter("param1", (0, 2)),
        NumericalContinuousParameter("param2", (-1, 1)),
    )

    assert searchspace.type == SearchSpaceType.CONTINUOUS
    assert searchspace.parameters == parameters


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


@pytest.mark.parametrize(
    "parameter_names",
    [
        [
            "Categorical_1",
            "Categorical_2",
            "Frame_A",
            "Some_Setting",
            "Num_disc_1",
            "Fraction_1",
            "Solvent_1",
            "Custom_1",
        ],
        [
            "Categorical_1_subset",
            "Categorical_2",
            "Frame_A",
            "Some_Setting",
            "Num_disc_1",
            "Fraction_1",
            "Solvent_1_subset",
            "Custom_1_subset",
            "Task",
        ],
    ],
    ids=["simple", "with_active_values"],
)
def test_searchspace_memory_estimate(searchspace: SearchSpace):
    """The memory estimate doesn't differ by more than 5% from the actual memory."""
    estimate = searchspace.estimate_product_space_size(searchspace.parameters)
    estimate_exp = estimate.exp_rep_bytes
    estimate_comp = estimate.comp_rep_bytes

    actual_exp = searchspace.discrete.exp_rep.memory_usage(deep=True, index=False).sum()
    actual_comp = searchspace.discrete.comp_rep.memory_usage(
        deep=True, index=False
    ).sum()

    assert 0.95 <= estimate_exp / actual_exp <= 1.05, (
        "Exp: ",
        estimate_exp,
        actual_exp,
    )
    assert 0.95 <= estimate_comp / actual_comp <= 1.05, (
        "Comp: ",
        estimate_comp,
        actual_comp,
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
            constraints_nonlin=(
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
            constraints_nonlin=(
                ContinuousCardinalityConstraint(
                    parameters=["c1", "c2"],
                    max_cardinality=1,
                ),
            ),
        )


@pytest.mark.skipif(
    not POLARS_INSTALLED, reason="Optional polars dependency not installed."
)
@pytest.mark.parametrize(
    "parameter_names",
    [
        [
            "Categorical_1",
            "Categorical_2",
            "Custom_1",
            "Solvent_1",
            "Solvent_2",
            "Fraction_1",
        ],
        [
            "Categorical_1_subset",
            "Categorical_2",
            "Custom_1_subset",
            "Solvent_1_subset",
            "Solvent_2",
            "Fraction_1",
        ],
    ],
    ids=["simple", "with_active_values"],
)
def test_polars_pandas_equivalence(parameters):
    """Search spaces created with Polars and Pandas are identical."""
    # Do Polars product
    ldf = parameter_cartesian_prod_polars(parameters)
    df_pl = ldf.collect()

    # Do Pandas product
    df_pd = parameter_cartesian_prod_pandas(parameters)

    # Assert equality
    assert_frame_equal(df_pl.to_pandas(), df_pd)


def test_task_parameter_active_values_validation():
    """Test SearchSpace.from_dataframe with TaskParameter active_values validation."""
    df = pd.DataFrame(
        [
            {"task": "source", "method": "old", "x1": 1.0},
            {"task": "source", "method": "old", "x1": 2.0},
            {"task": "source", "method": "new", "x1": 1.0},
            {"task": "target", "method": "new", "x1": 2.0},
        ]
    )

    num_param = NumericalDiscreteParameter(name="x1", values=[1.0, 2.0])
    task_param = TaskParameter(
        name="task", values=["source1", "source2", "target"], active_values=["target"]
    )
    cat_param = CategoricalParameter(
        name="method", values=["old", "new"], active_values=["new"]
    )

    # Two parameters invalid
    with pytest.raises(ExceptionGroup) as exc_info:
        SearchSpace.from_dataframe(df, parameters=[num_param, task_param, cat_param])

    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert is_all_instance(exceptions, IncompatibilityError)
    assert all("contains the following invalid values" in str(e) for e in exceptions)

    # One parameter invalid
    single_source_df = df[df["method"] == "new"]
    with pytest.raises(ExceptionGroup) as exc_info:
        SearchSpace.from_dataframe(
            single_source_df, parameters=[num_param, task_param, cat_param]
        )

    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 1
    assert is_all_instance(exceptions, IncompatibilityError)
    assert all("contains the following invalid values" in str(e) for e in exceptions)

    # All parameters valid
    target_df = df[(df["task"] == "target") & (df["method"] == "new")]
    searchspace = SearchSpace.from_dataframe(
        target_df, parameters=[num_param, task_param, cat_param]
    )
    assert len(searchspace.discrete.exp_rep) == 1
    assert all(searchspace.discrete.exp_rep["task"] == "target")


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize(
    ("constraint_names", "calculation", "assertion_func"),
    [
        (
            ["InterConstraint_3"],
            lambda samples: samples["Conti_finite1"].sum()
            + 2 * samples["Conti_finite2"].sum(),
            lambda result: np.isclose(result, 0.3, atol=1e-6),
        ),
        (
            ["InterConstraint_4"],
            lambda samples: 2 * samples["Conti_finite1"].sum()
            - samples["Conti_finite2"].sum(),
            lambda result: result >= 0.3 - 1e-6,
        ),
    ],
    ids=["equality", "inequality"],
)
def test_sample_from_polytope_with_interpoint_constraints(
    searchspace, calculation, assertion_func
):
    """Test _sample_from_polytope method with interpoint constraints."""
    subspace = searchspace.continuous

    assert subspace.has_interpoint_constraints

    # Test batch_size=1 and batch_size>1 as those the first one is a special case
    for batch_size in [1, 42]:
        bounds = subspace.comp_rep_bounds.values
        samples = subspace._sample_from_polytope(batch_size, bounds)

        constraint_result = calculation(samples)
        assert assertion_func(constraint_result)


def test_sample_from_polytope_mixed_constraints_with_interpoint():
    """Test _sample_from_polytope method with regular and interpoint constraints."""
    # NOTE: This test does not use our fixtures as those seem to create an infeasible
    # space
    parameters = [
        NumericalContinuousParameter("Conti_finite1", (0, 1)),
        NumericalContinuousParameter("Conti_finite2", (-1, 0)),
    ]

    regular_constraint = ContinuousLinearConstraint(
        parameters=["Conti_finite2"],
        operator=">=",
        coefficients=[1.0],
        rhs=-0.8,
    )
    interpoint_constraint = ContinuousLinearConstraint(
        parameters=["Conti_finite1"],
        operator="=",
        coefficients=[1],
        rhs=0.6,
        interpoint=True,
    )

    subspace = SubspaceContinuous(
        parameters=parameters,
        constraints_lin_ineq=[regular_constraint],
        constraints_lin_eq=[interpoint_constraint],
    )

    assert subspace.has_interpoint_constraints

    # Test batch size of 1 as well as one small and one large batch size
    for batch_size in [1, 2, 42]:
        bounds = subspace.comp_rep_bounds.values
        samples = subspace._sample_from_polytope(batch_size, bounds)

        # Verify regular constraint is satisfied for each row
        assert (samples["Conti_finite2"] >= -0.8 - 1e-6).all()

        # Verify interpoint constraint is satisfied across the batch
        interpoint_constraint_result = samples["Conti_finite1"].sum()
        assert np.isclose(interpoint_constraint_result, 0.6, atol=1e-6)
