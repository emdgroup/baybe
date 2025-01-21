"""Validation tests for meta recommenders."""

import pytest
from pytest import param

from baybe.recommenders.meta.adaptive import BatchSizeAdaptiveMetaRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender


@pytest.mark.parametrize(
    ("thresholds", "match"),
    [
        param([-1.0], "ends at -1.0", id="unused_interval"),
        param([1, 2], "equal to the number of intervals", id="inconsistent_lengths"),
    ],
)
def test_batch_size_adaptive_meta_recommender(thresholds, match):
    """Providing invalid constructor arguments raises an error."""
    rec = RandomRecommender()
    with pytest.raises(ValueError, match=match):
        BatchSizeAdaptiveMetaRecommender([rec, rec], thresholds)
