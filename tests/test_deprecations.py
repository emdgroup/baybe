"""Deprecation tests."""

import json

import pytest

from baybe import BayBE, Campaign
from baybe.searchspace import SearchSpace
from baybe.strategies import Strategy
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


def test_missing_strategy_type(config):
    """Specifying a strategy without a corresponding type raises a warning."""
    dict_ = json.loads(config)
    dict_["strategy"].pop("type")
    config_without_strategy_type = json.dumps(dict_)
    with pytest.warns(DeprecationWarning):
        Campaign.from_config(config_without_strategy_type)


def test_deprecated_strategy_class():
    """Using the deprecated ``Strategy`` class raises a warning."""
    with pytest.warns(DeprecationWarning):
        Strategy()


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
            "values": [
                "coarse",
                "fine",
                "ultra-fine"
            ],
            "encoding": "OHE"
        },
        {
            "type": "NumericalDiscreteParameter",
            "name": "Pressure[bar]",
            "values": [
                1,
                5,
                10
            ],
            "tolerance": 0.2
        },
        {
            "type": "SubstanceParameter",
            "name": "Solvent",
            "data": {
                "Solvent A": "COC",
                "Solvent B": "CCCCC",
                "Solvent C": "COCOC",
                "Solvent D": "CCOCCOCCN"
            },
            "decorrelate": true,
            "encoding": "MORDRED"
        }
    ],
    "constraints": [],
    "objective": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Yield",
                "mode": "MAX"
            }
        ]
    },
    "strategy": {
        "type": "TwoPhaseStrategy",
        "initial_recommender": {
            "type": "FPSRecommender"
        },
        "recommender": {
            "type": "SequentialGreedyRecommender",
            "surrogate_model": {
                "type": "GaussianProcessSurrogate"
            },
            "acquisition_function_cls": "qEI"
        },
        "switch_after": 1,
        "allow_repeated_recommendations": false,
        "allow_recommending_already_measured": false
    }
}
"""


def test_deprecated_config():
    """Using the deprecated config format raises a warning."""
    with pytest.warns(UserWarning):
        Campaign.from_config(deprecated_config)
