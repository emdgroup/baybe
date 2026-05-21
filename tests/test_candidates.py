"""Tests for the Candidate classes."""

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from baybe.constraints import DiscreteSumConstraint, ThresholdCondition
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.candidates import ProductCandidates, TableCandidates
from baybe.utils.interval import Interval


@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Num_disc_1", "Num_disc_2"],
        ["Categorical_1", "Num_disc_1"],
        ["Categorical_1", "Categorical_2"],
    ],
    ids=["numerical", "mixed", "categorical"],
)
@pytest.mark.parametrize(
    "dataframe_factory",
    [
        pytest.param(lambda pd_df: pd_df, id="pandas_eager"),
        pytest.param(pl.DataFrame, id="polars_eager"),
        pytest.param(
            pl.LazyFrame,
            id="polars_lazy",
        ),
        pytest.param(
            nw.from_native,
            id="narwhals_eager",
        ),
        pytest.param(
            lambda pd_df: nw.from_native(pd_df).lazy(),
            id="narwhals_lazy",
        ),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [16],
    ids=["b16"],
)
def test_table_candidates_creation(parameters, dataframe_factory, fake_measurements):
    """TableCandidates can be created with valid parameters and the compatible data."""
    df = dataframe_factory(fake_measurements)
    candidates = TableCandidates(parameters=tuple(parameters), dataframe=df)
    candidates_ldf = candidates.to_lazy()

    if isinstance(df, pl.LazyFrame) or isinstance(df, nw.LazyFrame):
        df_shape = df.collect().shape
    else:
        df_shape = df.shape
    assert isinstance(candidates_ldf, nw.LazyFrame)
    assert all([p.name in candidates_ldf.collect().columns for p in parameters])
    assert candidates_ldf.collect().shape == df_shape


@pytest.mark.parametrize(
    ("parameters", "dataframe"),
    [
        pytest.param([1], pd.DataFrame({"x": [1, 2, 3]}), id="invalid_parameter_input"),
        pytest.param(
            NumericalDiscreteParameter(
                name="Num_disc_1",
                values=(1, 2, 7),
                tolerance=0.3,
            ),
            pd.DataFrame({"Num_disc_1": [1, 2, 3]}),
            id="parameter_not_a_sequence",
        ),
        pytest.param(
            [
                NumericalDiscreteParameter(
                    name="Num_disc_1",
                    values=(1, 2, 7),
                    tolerance=0.3,
                )
            ],
            pd.DataFrame({"x": [1, 2, 3]}),
            id="unmatched_dataframe_columns",
        ),
        pytest.param(
            NumericalContinuousParameter(
                name="Conti_finite1",
                bounds=Interval(0, 1),
            ),
            pd.DataFrame({"x": [1, 2, 3]}),
            id="invalid_parameter_type_continuous",
        ),
        pytest.param([], pd.DataFrame({"x": [1, 2, 3]}), id="empty_parameter"),
        pytest.param(None, pd.DataFrame({"x": [1, 2, 3]}), id="none_parameter"),
        pytest.param(
            [
                NumericalDiscreteParameter(
                    name="Num_disc_1", values=(1, 2, 7), tolerance=0.3
                )
            ],
            123,  # Not a DataFrame or compatible type
            id="invalid_dataframe_type",
        ),
    ],
)
def test_table_candidates_invalid_input(parameters, dataframe):
    """Invalid parameter and dataframe inputs raise appropriate errors."""
    with pytest.raises((TypeError, ValueError, AttributeError)):
        TableCandidates(parameters=parameters, dataframe=dataframe)


@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Num_disc_1", "Num_disc_2", "Fraction_2"],
        ["Categorical_1", "Num_disc_1"],
        ["Categorical_1", "Categorical_2", "Categorical_1_subset"],
    ],
    ids=["numerical", "mixed", "categorical"],
)
@pytest.mark.parametrize(
    "constraint_names",
    [
        [],
        ["DiscreteSumConstraint"],
        ["DiscreteExcludeConstraint"],
    ],
    ids=["no_constraint", "sum", "exclude"],
)
def test_product_candidates_creation(parameters, constraints):
    """ProductCandidates can be created with valid parameters and constraints."""
    candidates = ProductCandidates(parameters=parameters, constraints=constraints)
    lazy_candidates = candidates.to_lazy()
    assert isinstance(lazy_candidates, nw.LazyFrame)
    for p in parameters:
        assert p.name in lazy_candidates.columns
    assert candidates.is_finite
    assert len(lazy_candidates.collect())

    ProductCandidates(parameters=parameters)


@pytest.mark.parametrize(
    ("parameters", "constraints"),
    [
        pytest.param([1], ["DiscreteSumConstraint"], id="invalid_parameter_input"),
        pytest.param(
            NumericalContinuousParameter(
                name="Conti_finite1",
                bounds=Interval(0, 1),
            ),
            ["DiscreteSumConstraint"],
            id="invalid_parameter_type",
        ),
        pytest.param([], ["DiscreteSumConstraint"], id="empty_parameter"),
        pytest.param(None, ["DiscreteSumConstraint"], id="none_parameter"),
    ],
)
def test_product_candidates_invalid_input(parameters, constraints):
    """Invalid parameter and constraint inputs raise appropriate errors."""
    with pytest.raises((TypeError, ValueError, AttributeError)):
        ProductCandidates(parameters=parameters, constraints=constraints)


@pytest.mark.parametrize(
    "parameters,expected",
    [
        pytest.param(
            (
                NumericalDiscreteParameter(name="x", values=(1, 2)),
                NumericalDiscreteParameter(name="y", values=(10, 20, 30)),
            ),
            {(x, y) for x in (1, 2) for y in (10, 20, 30)},
            id="numerical_numerical",
        ),
        pytest.param(
            (
                NumericalDiscreteParameter(name="x", values=(1, 2)),
                CategoricalParameter(name="cat", values=("a", "b")),
            ),
            {(x, c) for x in (1, 2) for c in ("a", "b")},
            id="numerical_categorical",
        ),
        pytest.param(
            (
                CategoricalParameter(name="cat1", values=("a", "b")),
                CategoricalParameter(name="cat2", values=("c", "d")),
                CategoricalParameter(name="cat3", values=("e", "f", "g")),
            ),
            {
                (c1, c2, c3)
                for c1 in ("a", "b")
                for c2 in ("c", "d")
                for c3 in ("e", "f", "g")
            },
            id="categorical_categorical",
        ),
    ],
)
def test_product_candidates_cartesian_product(parameters, expected):
    """ProductCandidates builds the correct cartesian product."""
    candidates = ProductCandidates(parameters=parameters)
    df = candidates.to_lazy().collect()
    assert df.shape[0] == len(expected)
    actual = {tuple(row) for row in df[[p.name for p in parameters]].to_numpy()}
    assert actual == expected


@pytest.mark.parametrize(
    ("parameters", "constraints", "expected_combinations"),
    [
        pytest.param(
            (
                NumericalDiscreteParameter(name="A", values=(1, 2)),
                NumericalDiscreteParameter(name="B", values=(1, 2)),
            ),
            [
                DiscreteSumConstraint(
                    parameters=["A", "B"],
                    condition=ThresholdCondition(threshold=3, operator="="),
                )
            ],
            {(1, 2), (2, 1)},
            id="sum_equals_3",
        ),
        pytest.param(
            (
                NumericalDiscreteParameter(name="A", values=(1, 2, 3)),
                NumericalDiscreteParameter(name="B", values=(1, 2, 3)),
                NumericalDiscreteParameter(name="C", values=(1, 2, 3)),
            ),
            [
                DiscreteSumConstraint(
                    parameters=["A", "B", "C"],
                    condition=ThresholdCondition(threshold=6, operator="<"),
                ),
                DiscreteSumConstraint(
                    parameters=["A", "B", "C"],
                    condition=ThresholdCondition(threshold=4, operator=">="),
                ),
            ],
            {
                (1, 2, 1),
                (2, 1, 1),
                (1, 1, 2),
                (2, 2, 1),
                (2, 1, 2),
                (1, 2, 2),
                (3, 1, 1),
                (1, 3, 1),
                (1, 1, 3),
            },
            id="sum_between_4_and_6",
        ),
    ],
)
def test_constraints_product_candidates(parameters, constraints, expected_combinations):
    """The constraints are applied correctly in ProductCandidates.to_lazy_candidates."""
    p_names = [p.name for p in parameters]
    candidates = ProductCandidates(parameters=parameters, constraints=constraints)
    df = candidates.to_lazy().collect()
    assert {tuple(row) for row in df[p_names].to_numpy()} == expected_combinations
    assert df.shape[0] == len(expected_combinations)
