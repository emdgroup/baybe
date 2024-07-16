"""Tests pending points mechanism."""

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Categorical_1", "Switch_1", "Num_disc_1"],
        ["Conti_finite1", "Conti_finite2", "Conti_finite3"],
        ["Categorical_1", "Num_disc_1", "Conti_finite1", "Conti_finite2"],
    ],
    ids=["discrete", "continuous", "hybrid"],
)
@pytest.mark.parametrize("batch_size", [1, 3, 10], ids=["b1", "b3", "b10"])
def test_pending_points(campaign, batch_size):
    """Test there is no recommendation overlap if pending points are specified."""
    rec1 = campaign.recommend(batch_size)
    rec2 = campaign.recommend(batch_size=batch_size, pending_measurements=rec1)
    overlap = pd.merge(rec1, rec2, how="inner")

    assert len(overlap) == 0, (
        f"Recommendations are overlapping!\n\nRecommendations 1:\n{rec1}\n\n"
        f"Recommendations 2:\n{rec2}\n\nOverlap:\n{overlap}"
    )
