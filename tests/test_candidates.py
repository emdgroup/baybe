"""Tests for candidate generators."""

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from baybe.constraints import DiscreteSumConstraint, ThresholdCondition
from baybe.constraints.conditions import SubSelectionCondition
from baybe.constraints.discrete import DiscreteExcludeConstraint
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.candidates import (
    EmptyCandidates,
    ProductCandidates,
    TableCandidates,
)
from baybe.utils.dataframe import create_fake_input

p_disc = NumericalDiscreteParameter("disc", (1, 2))
p_disc2 = NumericalDiscreteParameter("disc2", (0, 10))
p_cat = CategoricalParameter("cat", ("a", "b", "c"))
p_cont = NumericalContinuousParameter("cont", (3, 8))
c_sum = DiscreteSumConstraint(["disc", "disc2"], ThresholdCondition(2, "<="))
c_sub = DiscreteExcludeConstraint(["disc"], [SubSelectionCondition([1])])
edf = pd.DataFrame()


def test_empty_candidates():
    """EmptyCandidates has no parameters, is finite, and yields an empty lazy frame."""
    candidates = EmptyCandidates()
    candidates_ldf = candidates.to_lazy()
    candidates_df = candidates_ldf.collect()

    assert candidates.parameters == ()
    assert candidates.is_finite
    assert isinstance(candidates_ldf, nw.LazyFrame)
    assert candidates_df.shape == (0, 0)


@pytest.mark.parametrize(
    "dataframe_factory",
    [
        pytest.param(lambda pd_df: pd_df, id="pandas_eager"),
        pytest.param(pl.DataFrame, id="polars_eager"),
        pytest.param(lambda x: nw.from_native(x, eager_only=True), id="narwhals_eager"),
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


def test_table_candidates_empty_rows():
    """TableCandidates accepts a zero-row dataframe with correct columns."""
    parameters = [p_disc, p_cat]
    empty_df = pd.DataFrame(columns=[p.name for p in parameters])
    candidates = TableCandidates(parameters, empty_df)
    candidates_df = candidates.to_lazy().collect()

    assert candidates.is_finite
    assert set(candidates_df.columns) == {p.name for p in parameters}
    assert len(candidates_df) == 0


def test_table_candidates_duplicate_rows():
    """TableCandidates raises an error when the dataframe contains duplicate rows."""
    parameters = [p_disc, p_cat]
    data = create_fake_input(parameters, [], n_rows=1)
    duplicate_data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate parameter configurations"):
        TableCandidates(parameters, duplicate_data)


@pytest.mark.parametrize(
    ("parameters", "dataframe", "error"),
    [
        pytest.param([], edf, ValueError(">= 1"), id="empty_param"),
        pytest.param(None, edf, TypeError("not iterable"), id="none_param"),
        pytest.param([p_cont], edf, TypeError("be <class"), id="param_type"),
        pytest.param(p_disc, edf, TypeError("not iterable"), id="no_param_seq"),
        pytest.param([p_disc], 123, TypeError("Unsupported dataframe"), id="df_type"),
        pytest.param([p_disc], pl.LazyFrame(), TypeError("eager_only"), id="pl_lazy"),
        pytest.param(
            [p_disc],
            nw.from_native(pl.LazyFrame()),
            TypeError("eager_only"),
            id="nw_lazy",
            marks=pytest.mark.xfail(
                reason="https://github.com/narwhals-dev/narwhals/pull/3677", strict=True
            ),
        ),
        pytest.param(
            [p_disc],
            pd.DataFrame({"x": [1]}),
            ValueError("missing columns"),
            id="missing_cols",
        ),
        pytest.param(
            [p_disc],
            pd.DataFrame({"disc": [1], "extra": [2]}),
            ValueError("not correspond"),
            id="extra_cols",
        ),
        pytest.param(
            [p_disc],
            pd.DataFrame(),
            ValueError("missing columns"),
            id="empty_rows_missing_cols",
        ),
        pytest.param(
            [p_disc],
            pd.DataFrame(columns=["disc", "extra"]),
            ValueError("not correspond"),
            id="empty_rows_extra_cols",
        ),
    ],
)
def test_table_candidates_validation(parameters, dataframe, error):
    """Incompatible parameter and dataframe inputs raise appropriate errors."""
    with pytest.raises(error.__class__, match=str(error)):
        TableCandidates(parameters, dataframe)


@pytest.mark.parametrize(
    ("constraints", "expected"),
    [
        ([], [[1, 0], [1, 10], [2, 0], [2, 10]]),
        ([c_sum], [[1, 0], [2, 0]]),
        ([c_sub], [[2, 0], [2, 10]]),
    ],
    ids=["no_constraint", "sum", "subselection"],
)
def test_product_candidates_generation(constraints, expected):
    """ProductCandidates generates the expected lazy dataframe."""
    parameters = [p_disc, p_disc2]
    candidates = ProductCandidates(parameters, constraints)
    candidates_ldf = candidates.to_lazy()
    candidates_df = candidates_ldf.collect()

    assert candidates.is_finite
    assert isinstance(candidates_ldf, nw.LazyFrame)
    assert set(candidates_df.columns) == {p.name for p in parameters}
    assert_frame_equal(
        candidates_df.to_pandas(),
        pd.DataFrame(expected, columns=[p.name for p in parameters]),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("parameters", "constraints", "error"),
    [
        pytest.param([], (), ValueError(">= 1"), id="empty_param"),
        pytest.param(None, (), TypeError("not iterable"), id="none_param"),
        pytest.param([p_cont], (), TypeError("be <class"), id="param_type"),
        pytest.param(p_disc, (), TypeError("not iterable"), id="no_param_seq"),
        pytest.param([p_disc], None, TypeError("not iterable"), id="none_constraint"),
        pytest.param(
            [p_disc2], [c_sum], ValueError("does not exist"), id="wrong_constraint"
        ),
    ],
)
def test_product_candidates_validation(parameters, constraints, error):
    """Incompatible parameter and constraint inputs raise appropriate errors."""
    with pytest.raises(error.__class__, match=str(error)):
        ProductCandidates(parameters, constraints)
