"""Test Polars implementations of constraints."""

import pytest
from pandas.testing import assert_frame_equal

from baybe._optional.info import POLARS_INSTALLED
from baybe.searchspace.discrete import (
    _apply_constraint_filter_pandas,
    _apply_constraint_filter_polars,
    parameter_cartesian_prod_pandas,
    parameter_cartesian_prod_polars,
)

pytestmark = pytest.mark.skipif(
    not POLARS_INSTALLED, reason="Optional polars dependency not installed."
)


if POLARS_INSTALLED:
    import polars as pl


def _lazyframe_from_product(parameters):
    """Create a Polars lazyframe from the product of given parameters and return it."""
    param_frames = [pl.LazyFrame({p.name: p.values}) for p in parameters]

    # Handling edge cases
    if len(param_frames) == 1:
        return param_frames[0]

    # Cross-join parameters
    res = param_frames[0]
    for frame in param_frames[1:]:
        res = res.join(frame, how="cross", force_parallel=True)

    return res


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_8"]])
def test_polars_prodsum1(parameters, constraints):
    """Tests Polars implementation of sum constraint."""
    ldf = _lazyframe_from_product(parameters)
    ldf, _ = _apply_constraint_filter_polars(ldf, constraints)

    # Number of entries with 1,2-sum above 150
    ldf = ldf.with_columns(sum=pl.sum_horizontal(["Fraction_1", "Fraction_2"]))
    ldf = ldf.filter(pl.col("sum") > 150)
    num_entries = len(ldf.collect())

    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_9"]])
def test_polars_prodsum2(parameters, constraints):
    """Tests Polars implementation of product constrain."""
    ldf = _lazyframe_from_product(parameters)
    ldf, _ = _apply_constraint_filter_polars(ldf, constraints)

    # Number of entries with product under 30
    df = ldf.filter(
        pl.reduce(lambda acc, x: acc * x, pl.col(["Fraction_1", "Fraction_2"])).alias(
            "prod"
        )
        < 30
    ).collect()

    num_entries = len(df)
    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_10"]])
def test_polars_prodsum3(parameters, constraints):
    """Tests Polars implementation of exact sum constraint."""
    ldf = _lazyframe_from_product(parameters)
    ldf, _ = _apply_constraint_filter_polars(ldf, constraints)

    # Number of entries with sum unequal to 100
    ldf = ldf.with_columns(sum=pl.sum_horizontal(["Fraction_1", "Fraction_2"]))
    df = ldf.select(abs(pl.col("sum") - 100)).filter(pl.col("sum") > 0.01).collect()

    num_entries = len(df)

    assert num_entries == 0


@pytest.mark.parametrize(
    "parameter_names",
    [["Solvent_1", "SomeSetting", "Temperature", "Pressure"]],
)
@pytest.mark.parametrize(
    "constraint_names", [["Constraint_4", "Constraint_5", "Constraint_6"]]
)
def test_polars_exclusion(mock_substances, parameters, constraints):
    """Tests Polars implementation of exclusion constraint."""
    ldf = _lazyframe_from_product(parameters)
    ldf, _ = _apply_constraint_filter_polars(ldf, constraints)

    # Number of entries with either first/second substance and a temperature above 151
    df = ldf.filter(
        (pl.col("Temperature") > 151)
        & (pl.col("Solvent_1").is_in(list(mock_substances)[:2]))
    ).collect()
    num_entries = len(df)
    assert num_entries == 0

    # Number of entries with either last / second last substance and a pressure above 5
    df = ldf.filter(
        (pl.col("Pressure") > 5)
        & (pl.col("Solvent_1").is_in(list(mock_substances)[-2:]))
    ).collect()
    num_entries = len(df)
    assert num_entries == 0

    # Number of entries with pressure below 3 and temperature above 120
    df = ldf.filter((pl.col("Pressure") < 3) & (pl.col("Temperature") > 120)).collect()
    num_entries = len(df)
    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Solvent_1", "Solvent_2", "Solvent_3"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_7"]])
def test_polars_label_duplicates(parameters, constraints):
    """Tests Polars implementation of no-label duplicates constraint."""
    ldf = _lazyframe_from_product(parameters)
    ldf, _ = _apply_constraint_filter_polars(ldf, constraints)

    ldf = ldf.with_columns(
        pl.concat_list(pl.col(["Solvent_1", "Solvent_2", "Solvent_3"]))
        .list.eval(pl.element().n_unique())
        .explode()
        .alias("n_unique")
    )
    df = ldf.filter(pl.col("n_unique") != len(parameters)).collect()

    num_entries = len(df)
    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Solvent_1", "Solvent_2", "Solvent_3"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_15"]])
def test_polars_linked_parameters(parameters, constraints):
    """Tests Polars implementation of linked parameters constraint."""
    ldf = _lazyframe_from_product(parameters)
    ldf, _ = _apply_constraint_filter_polars(ldf, constraints)

    ldf = ldf.with_columns(
        pl.concat_list(pl.col(["Solvent_1", "Solvent_2", "Solvent_3"]))
        .list.eval(pl.element().n_unique())
        .explode()
        .alias("n_unique")
    )
    df = ldf.filter(pl.col("n_unique") != 1).collect()

    num_entries = len(df)
    assert num_entries == 0


@pytest.mark.parametrize(
    "parameter_names",
    [
        [
            "Temperature",
            "Solvent_1",
            "Solvent_2",
            "Solvent_3",
            "Fraction_1",
            "Fraction_2",
            "Fraction_3",
        ]
    ],
)
@pytest.mark.parametrize(
    "constraint_names",
    [
        ["Constraint_4"],
        ["Constraint_12"],
        ["Constraint_15", "Constraint_8", "Constraint_9"],
    ],
)
def test_polars_product(constraints, parameters):
    """Test the result of parameter product and filtering."""
    # Do Polars product
    ldf = parameter_cartesian_prod_polars(parameters)
    df_pl = ldf.collect()

    # Do Pandas product
    df_pd = parameter_cartesian_prod_pandas(parameters)

    # Assert equality of lengths before filtering
    assert_frame_equal(df_pl.to_pandas(), df_pd)

    # Apply constraints
    df_pd_filtered = _apply_constraint_filter_pandas(df_pd, constraints)
    df_pl_filtered = (
        _apply_constraint_filter_polars(ldf, constraints)[0].collect().to_pandas()
    )

    # Assert strict equality of two dataframes
    assert_frame_equal(df_pl_filtered, df_pd_filtered)
