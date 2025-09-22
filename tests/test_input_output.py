"""Tests for basic input-output and iterative loop."""

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements


@pytest.mark.parametrize(
    "bad_val, parameter_names",
    [
        param(1337, ["Num_disc_1"], id="num_param_outside_tol"),
        param(np.nan, ["Num_disc_1"], id="num_param_nan"),
        param("asd", ["Num_disc_1"], id="num_param_str"),
        param("asd", ["Categorical_1"], id="cat_param_invalid_cat"),
        param(np.nan, ["Categorical_1"], id="cat_param_nan"),
        param(1337, ["Categorical_1"], id="cat_param_num"),
        param("asd", ["Custom_1"], id="custom_param_invalid_cat"),
        param(np.nan, ["Custom_1"], id="custom_param_nan"),
        param(1337, ["Custom_1"], id="custom_param_num"),
        param("asd", ["Task"], id="task_param_invalid_cat"),
        param(np.nan, ["Task"], id="task_param_nan"),
        param(1337, ["Task"], id="task_param_num"),
    ],
)
def test_bad_parameter_input_value(campaign, bad_val, fake_measurements):
    """Test attempting to read in an invalid parameter value."""
    # Add an invalid value
    fake_measurements[campaign.parameters[0].name].iloc[0] = bad_val
    with pytest.raises((ValueError, TypeError)):
        campaign.add_measurements(fake_measurements)


@pytest.mark.parametrize(
    "bad_val, target_names",
    [
        param("asd", ["Target_max"], id="num_target_str"),
        param(1337, ["Target_binary"], id="binary_target_num"),
        param("asd", ["Target_binary"], id="binary_target_str"),
    ],
)
def test_bad_target_input_value(campaign, bad_val):
    """Test attempting to read in an invalid target value."""
    rec = campaign.recommend(batch_size=3)
    add_fake_measurements(rec, campaign.targets)

    # Add an invalid value
    rec[campaign.targets[0].name].iloc[0] = bad_val
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
    df = pd.DataFrame({f"p{k + 1}": values for k in range(n_parameters)})
    searchspace = SearchSpace.from_dataframe(
        df,
        parameters=[
            NumericalDiscreteParameter(name=f"p{k + 1}", values=values)
            for k in range(n_parameters)
        ],
    )
    objective = NumericalTarget(name="t").to_objective()
    recommender = BotorchRecommender()

    # Add first and last point as measurement, target value is the sum of all parameters
    measurements = df.iloc[[0, -1], :]
    measurements = measurements.assign(t=measurements.sum(axis=1))

    # Get recommendations and assert that they are not in ascending order regarding
    # their target value, as would be in the search space.
    rec = recommender.recommend(5, searchspace, objective, measurements)
    rec = rec.assign(t=rec.sum(axis=1))
    assert not rec["t"].is_monotonic_increasing, rec["t"].values


@pytest.mark.parametrize(
    "parameters",
    [
        (
            NumericalDiscreteParameter("p_disc", values=(0.0, 1.0, 2.0)),
            NumericalContinuousParameter("p_conti", bounds=(0.0, 2.0)),
        )
    ],
)
@pytest.mark.parametrize(
    "row_dict",
    [
        param(
            {
                "p_disc": np.nextafter(0.0, -np.inf),
                "p_conti": 1.0,
                "Target_max": 1.1,
            },
            id="disc_below_exact",
        ),
        param(
            {
                "p_disc": np.nextafter(0.0, np.inf),
                "p_conti": 1.0,
                "Target_max": 1.1,
            },
            id="disc_above_exact",
        ),
        param(
            {
                "p_disc": 1.0,
                "p_conti": np.nextafter(0.0, -np.inf),
                "Target_max": 1.1,
            },
            id="conti_below_bounds",
        ),
        param(
            {
                "p_disc": 1.0,
                "p_conti": np.nextafter(2.0, np.inf),
                "Target_max": 1.1,
            },
            id="conti_above_bounds",
        ),
    ],
)
def test_numerical_tolerance(campaign, row_dict):
    """Data addition is possible despite numerical inaccuracies."""
    df = pd.DataFrame.from_records([row_dict])
    campaign.add_measurements(df, numerical_measurements_must_be_within_tolerance=True)
