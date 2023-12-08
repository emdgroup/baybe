"""Deprecation tests."""

import json

import pytest

from baybe import BayBE, Campaign
from baybe.searchspace import SearchSpace
from baybe.strategies import Strategy
from baybe.targets import Objective
from baybe.utils.interval import Interval


def test_deprecated_baybe_class(parameters, objective):
    """Using the deprecated ``BayBE`` class should raise a warning."""
    with pytest.warns(DeprecationWarning):
        BayBE(SearchSpace.from_product(parameters), objective)


def test_moved_objective(targets):
    """Importing ``Objective`` from ``baybe.targets`` should raise a warning."""
    with pytest.warns(DeprecationWarning):
        Objective(mode="SINGLE", targets=targets)


def test_renamed_surrogate():
    """Importing from ``baybe.surrogate`` should raise a warning."""
    with pytest.warns(DeprecationWarning):
        from baybe.surrogate import GaussianProcessSurrogate  # noqa: F401


def test_missing_strategy_type(config):
    """Specifying a strategy without a corresponding type should trigger a warning."""
    dict_ = json.loads(config)
    dict_["strategy"].pop("type")
    config_without_strategy_type = json.dumps(dict_)
    with pytest.warns(DeprecationWarning):
        Campaign.from_config(config_without_strategy_type)


def test_deprecated_strategy_class():
    """Using the deprecated ``Strategy`` class should raise a warning."""
    with pytest.warns(DeprecationWarning):
        Strategy()


def test_deprecated_interval_is_finite():
    """Using the deprecated ``Interval.is_finite`` property should raise a warning."""
    with pytest.warns(DeprecationWarning):
        Interval(0, 1).is_finite


def test_deprecated_interval_is_bounded():
    """Using the deprecated ``Interval.is_bounded`` property should raise a warning."""
    with pytest.warns(DeprecationWarning):
        Interval(0, 1).is_bounded
