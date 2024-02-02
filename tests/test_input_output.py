"""Tests for basic input-output and iterative loop."""
import numpy as np
import pytest

from baybe.utils.dataframe import add_fake_results

# List of tests that are expected to fail (still missing implementation etc)
param_xfails = []
target_xfails = []


@pytest.mark.parametrize(
    "bad_val",
    [1337, np.nan, "asd"],
    ids=["not_within_tol", "nan", "string_instead_float"],
)
def test_bad_parameter_input_value(campaign, good_reference_values, bad_val, request):
    """Test attempting to read in an invalid parameter value."""
    if request.node.callspec.id in param_xfails:
        pytest.xfail()

    rec = campaign.recommend(batch_size=3)
    add_fake_results(
        rec,
        campaign,
        good_reference_values=good_reference_values,
    )

    # Add an invalid value
    rec.Num_disc_1.iloc[0] = bad_val
    with pytest.raises((ValueError, TypeError)):
        campaign.add_measurements(rec)


@pytest.mark.parametrize(
    "bad_val",
    [np.nan, "asd"],
    ids=["nan", "string_instead_float"],
)
def test_bad_target_input_value(campaign, good_reference_values, bad_val, request):
    """Test attempting to read in an invalid target value."""
    if request.node.callspec.id in target_xfails:
        pytest.xfail()

    rec = campaign.recommend(batch_size=3)
    add_fake_results(
        rec,
        campaign,
        good_reference_values=good_reference_values,
    )

    # Add an invalid value
    rec.Target_max.iloc[0] = bad_val
    with pytest.raises((ValueError, TypeError)):
        campaign.add_measurements(rec)
