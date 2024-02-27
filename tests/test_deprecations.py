"""Deprecation tests."""

import json

import pytest

from baybe import BayBE, Campaign
from baybe.exceptions import DeprecationError
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.nonpredictive.sampling import (
    FPSRecommender,
    RandomRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.strategies import (
    SequentialStrategy,
    Strategy,
    StreamingSequentialStrategy,
    TwoPhaseStrategy,
)
from baybe.targets import Objective
from baybe.targets.base import Target
from baybe.utils.interval import Interval


def test_deprecated_baybe_class(parameters, objective):
    """Using the deprecated ``BayBE`` class raises a warning."""
    with pytest.warns(DeprecationWarning):
        BayBE(SearchSpace.from_product(parameters), objective)


def test_moved_objective(targets):
    """Importing ``Objective`` from ``baybe.targets`` raises a warning."""
    with pytest.warns(DeprecationWarning):
        Objective(mode="SINGLE", targets=targets)


def test_renamed_surrogate():
    """Importing from ``baybe.surrogate`` raises a warning."""
    with pytest.warns(DeprecationWarning):
        from baybe.surrogate import GaussianProcessSurrogate  # noqa: F401


def test_missing_recommender_type(config):
    """Specifying a recommender without a corresponding type raises a warning."""
    dict_ = json.loads(config)
    dict_["recommender"].pop("type")
    config_without_strategy_type = json.dumps(dict_)
    with pytest.warns(DeprecationWarning):
        Campaign.from_config(config_without_strategy_type)


# Create some recommenders of different class for better differentiation after roundtrip
RECOMMENDERS = [RandomRecommender(), FPSRecommender()]
assert len(RECOMMENDERS) == len(set(rec.__class__.__name__ for rec in RECOMMENDERS))


@pytest.mark.parametrize(
    "test_objects",
    [
        (Strategy, {}),
        (TwoPhaseStrategy, {}),
        (SequentialStrategy, {"recommenders": RECOMMENDERS}),
        (StreamingSequentialStrategy, {"recommenders": RECOMMENDERS}),
    ],
)
def test_deprecated_strategies(test_objects):
    """Using the deprecated strategy classes raises a warning."""
    strategy, arguments = test_objects
    with pytest.warns(DeprecationWarning):
        strategy(**arguments)


def test_deprecated_interval_is_finite():
    """Using the deprecated ``Interval.is_finite`` property raises a warning."""
    with pytest.warns(DeprecationWarning):
        Interval(0, 1).is_finite


def test_missing_target_type():
    """Specifying a target without a corresponding type raises a warning."""
    with pytest.warns(DeprecationWarning):
        Target.from_json(
            json.dumps(
                {
                    "name": "missing_type",
                    "mode": "MAX",
                }
            )
        )


deprecated_config = """
{
    "parameters": [
        {
            "type": "CategoricalParameter",
            "name": "Granularity",
            "values": ["coarse", "fine", "ultra-fine"]
        }
    ],
    "objective": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Yield",
                "mode": "MAX"
            }
        ]
    }
}
"""


def test_deprecated_config():
    """Using the deprecated config format raises a warning."""
    with pytest.warns(UserWarning):
        Campaign.from_config(deprecated_config)


@pytest.mark.parametrize("flag", [False, True])
def test_deprecated_campaign_tolerance_flag(flag):
    """Constructing a Campaign with the deprecated tolerance flag raises an error."""
    with pytest.raises(DeprecationError):
        Campaign(None, None, None, numerical_measurements_must_be_within_tolerance=flag)


def test_deprecated_batch_quantity_keyword(campaign):
    """Using the deprecated batch_quantity keyword raises an error."""
    with pytest.raises(DeprecationError):
        campaign.recommend(batch_quantity=5)


@pytest.mark.parametrize("flag", (True, False))
def test_deprecated_strategy_allow_flags(flag):
    """Using the deprecated recommender "allow" flags raises an error."""
    with pytest.raises(DeprecationError):
        TwoPhaseMetaRecommender(allow_recommending_already_measured=flag)
    with pytest.raises(DeprecationError):
        TwoPhaseMetaRecommender(allow_repeated_recommendations=flag)


def test_deprecated_strategy_campaign_flag(recommender):
    """Using the deprecated strategy keyword raises an error."""
    with pytest.raises(DeprecationError):
        Campaign(None, None, None, strategy=recommender)
