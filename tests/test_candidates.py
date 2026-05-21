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
from baybe.searchspace.candidates import ProductCandidates, TableCandidates
from baybe.utils.dataframe import create_fake_input

p_disc = NumericalDiscreteParameter("disc", (1, 2))
p_disc2 = NumericalDiscreteParameter("disc2", (0, 10))
p_cat = CategoricalParameter("cat", ("a", "b", "c"))
p_cont = NumericalContinuousParameter("cont", (3, 8))
c_sum = DiscreteSumConstraint(["disc", "disc2"], ThresholdCondition(2, "<="))
c_sub = DiscreteExcludeConstraint(["disc"], [SubSelectionCondition([1])])
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
            ValueError("not correspond"),
            id="extra_cols",
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
