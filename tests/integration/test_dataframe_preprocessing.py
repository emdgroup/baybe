"""Dataframe preprocessing tests."""

from unittest.mock import patch

from baybe.campaign import preprocess_dataframe
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.utils.dataframe import add_fake_measurements


@patch("baybe.campaign.preprocess_dataframe", wraps=preprocess_dataframe)
@patch("baybe.recommenders.pure.base.preprocess_dataframe", wraps=preprocess_dataframe)
@patch(
    "baybe.recommenders.pure.bayesian.base.preprocess_dataframe",
    wraps=preprocess_dataframe,
)
def test_dataframes_are_preprocessed_only_once(
    mock_bayesian, mock_recommender, mock_campaign, campaign
):
    """Data preprocessing happens only once, regardless of the entry point."""
    # NOTE: This test only ensures that preprocessing happens mutually exclusively
    #   in campaigns and recommenders. Unfortunately, it does *not* verify if the
    #   preprocessing function is called in other places of the same execution chain.
    #   Testing this would require mocking at the function source level, which is
    #   hardly achievable with the current import structure. As a manual verification
    #   step: a simple reference search of `preprocess_dataframe` in the main code base
    #   should reveal imports in only these two places.

    # Get some fake data to be preprocessed
    df = campaign.recommend(1)
    add_fake_measurements(df, campaign.targets)

    # Preprocessing happens while adding the data, but not again during recommendation
    campaign.add_measurements(df)
    campaign.recommend(1)
    assert mock_campaign.call_count == 1
    assert mock_recommender.call_count == 0
    assert mock_bayesian.call_count == 0

    # However, calling the recommender directly triggers preprocessing
    BotorchRecommender().recommend(1, campaign.searchspace, campaign.objective, df)
    assert mock_bayesian.call_count + mock_recommender.call_count == 1
