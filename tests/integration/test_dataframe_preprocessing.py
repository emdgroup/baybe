"""Dataframe preprocessing tests."""

from unittest.mock import patch

from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.utils.dataframe import add_fake_measurements
from baybe.utils.validation import validate_parameter_input


@patch(
    "baybe.utils.validation.validate_parameter_input", wraps=validate_parameter_input
)
def test_dataframes_are_preprocessed_only_once(mock, campaign):
    """Data preprocessing happens only once, regardless of the entry point."""
    # NOTE: The call count is tracked based on the first (unconditionally) executed
    #   statement in the fucntion when validation is active: `validate_parameter_input`

    # Get some fake data to be preprocessed
    df = campaign.recommend(1)
    add_fake_measurements(df, campaign.targets)

    # Preprocessing happens while adding the data, but not again during recommendation
    campaign.add_measurements(df)
    campaign.recommend(1)
    assert mock.call_count == 1

    # However, calling the recommender directly triggers preprocessing
    BotorchRecommender().recommend(1, campaign.searchspace, campaign.objective, df)
    assert mock.call_count == 2
