"""Tests for basic input-output and iterative loop."""

import numpy as np
import pandas as pd
import pytest

from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements

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
    add_fake_measurements(
        rec,
        campaign.targets,
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
    add_fake_measurements(
        rec,
        campaign.targets,
        good_reference_values=good_reference_values,
    )

    # Add an invalid value
    rec.Target_max.iloc[0] = bad_val
    with pytest.raises((ValueError, TypeError)):
        campaign.add_measurements(rec)


@pytest.mark.parametrize("n_values", [5, 10, 20])
@pytest.mark.parametrize("n_parameters", [3, 5, 10])
def test_recommendation_is_not_ordered(n_values, n_parameters):
    """Test whether recommendations are unintentionally sorted.

    This is to ensure that they are not reordered according to the order they appear
    in the search space. To this end we create an example where the search space
    parameter entries are monotonically increasing and a target variable in max mode
    comprises the sum of the parameters. We feed the first and last point as training
    data. This should lead to a recommendation order that is very different from the
    original search space order (albeit not perfectly anti-ordered).
    """
    # Set up custom df with entries monotonically increasing
    values = list(range(n_values))
    df = pd.DataFrame({f"p{k+1}": values for k in range(n_parameters)})
    searchspace = SearchSpace.from_dataframe(
        df,
        parameters=[
            NumericalDiscreteParameter(name=f"p{k+1}", values=values)
            for k in range(n_parameters)
        ],
    )
    objective = NumericalTarget(name="t", mode="MAX").to_objective()
    recommender = BotorchRecommender()

    # Add first and last point as measurement, target value is the sum of all parameters
    measurements = df.iloc[[0, -1], :]
    measurements = measurements.assign(t=measurements.sum(axis=1))

    # Get recommendations and assert that they are not in ascending order regarding
    # their target value, as would be in the search space.
    rec = recommender.recommend(5, searchspace, objective, measurements)
    rec = rec.assign(t=rec.sum(axis=1))
    assert not rec["t"].is_monotonic_increasing, rec["t"].values
