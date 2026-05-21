"""Tests for candidate generators."""

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from baybe.constraints import DiscreteSumConstraint, ThresholdCondition
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.candidates import ProductCandidates, TableCandidates
from baybe.utils.dataframe import create_fake_input
from baybe.utils.interval import Interval

p_disc = NumericalDiscreteParameter("disc", (1, 2, 7))
p_cat = CategoricalParameter("cat", ("a", "b", "c"))
p_cont = NumericalContinuousParameter("cont", (3, 8))
edf = pd.DataFrame()


@pytest.mark.parametrize(
    "dataframe_factory",
    [
        pytest.param(lambda pd_df: pd_df, id="pandas_eager"),
        pytest.param(pl.DataFrame, id="polars_eager"),
        pytest.param(pl.LazyFrame, id="polars_lazy"),
        pytest.param(lambda x: nw.from_native(x, eager_only=True), id="narwhals_eager"),
        pytest.param(lambda x: nw.from_native(x).lazy(), id="narwhals_lazy"),
    ],
)
def test_table_candidates_generation(dataframe_factory):
    """TableCandidates generates the expected lazy dataframe."""
    parameters = [p_disc, p_cat]
    data = create_fake_input(parameters, [], n_rows=4)
    df = dataframe_factory(data)
    candidates = TableCandidates(parameters, df)
    candidates_ldf = candidates.to_lazy()
    candidates_df = candidates_ldf.collect()

    assert candidates.is_finite
    assert isinstance(candidates_ldf, nw.LazyFrame)
    assert set(candidates_df.columns) == {p.name for p in parameters}
    assert candidates_df.shape == data.shape
    assert_frame_equal(candidates_df.to_pandas(), data)


@pytest.mark.parametrize(
    ("parameters", "dataframe", "error"),
    [
        pytest.param([], edf, ValueError(">= 1"), id="empty_param"),
        pytest.param(None, edf, TypeError("not iterable"), id="none_param"),
        pytest.param([p_cont], edf, TypeError("be <class"), id="param_type"),
        pytest.param(p_disc, edf, TypeError("not iterable"), id="no_param_seq"),
        pytest.param([p_disc], 123, TypeError("dataframe type"), id="df_type"),
        pytest.param(
            [p_disc],
            pd.DataFrame({"x": [1]}),
            ValueError("missing columns"),
            id="missing_cols",
        ),
        pytest.param(
            [p_disc],
            pd.DataFrame({"disc": [1], "extra": [2]}),
            ValueError("extra columns"),
            id="extra_cols",
        ),
    ],
)
def test_table_candidates_validation(parameters, dataframe, error):
    """Incompatible parameter and dataframe inputs raise appropriate errors."""
    with pytest.raises(error.__class__, match=str(error)):
        TableCandidates(parameters, dataframe)


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
    df = candidates.to_lazy_candidates().collect()
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
    df = candidates.to_lazy_candidates().collect()
    assert {tuple(row) for row in df[p_names].to_numpy()} == expected_combinations
    assert df.shape[0] == len(expected_combinations)
