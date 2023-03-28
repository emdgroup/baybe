"""
Tests for basic input-output nad iterative loop.
"""
import numpy as np
import pytest

from baybe.utils import add_fake_results


# List of tests that are expected to fail (still missing implementation etc)
param_xfails = []
target_xfails = []


@pytest.mark.parametrize(
    "bad_val",
    [1337, np.nan, "asd"],
    ids=["not_within_tol", "nan", "string_instead_float"],
)
def test_bad_parameter_input_value(baybe, good_reference_values, bad_val, request):
    """
    Test attempting to read in an invalid parameter value.
    """
    if request.node.callspec.id in param_xfails:
        pytest.xfail()

    rec = baybe.recommend(batch_quantity=3)
    add_fake_results(
        rec,
        baybe,
        good_reference_values=good_reference_values,
    )

    # Add an invalid value
    rec.Num_disc_1.iloc[0] = bad_val
    with pytest.raises((ValueError, TypeError)):
        baybe.add_results(rec)


@pytest.mark.parametrize(
    "bad_val",
    [np.nan, "asd"],
    ids=["nan", "string_instead_float"],
)
def test_bad_target_input_value(baybe, good_reference_values, bad_val, request):
    """
    Test attempting to read in an invalid parameter value.
    """
    if request.node.callspec.id in target_xfails:
        pytest.xfail()

    rec = baybe.recommend(batch_quantity=3)
    add_fake_results(
        rec,
        baybe,
        good_reference_values=good_reference_values,
    )

    # Add an invalid value
    rec.Target.iloc[0] = bad_val
    with pytest.raises((ValueError, TypeError)):
        baybe.add_results(rec)
