"""Surrogate tests."""

from unittest.mock import patch

from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.dataframe import add_fake_measurements


@patch.object(GaussianProcessSurrogate, "_fit")
def test_caching(patched, searchspace, objective):
    """A second fit call with the same context does not trigger retraining."""
    # Prepare the setting
    measurements = RandomRecommender().recommend(3, searchspace, objective)
    add_fake_measurements(measurements, objective.targets)
    surrogate = GaussianProcessSurrogate()

    # First call
    surrogate.fit(searchspace, objective, measurements)
    patched.assert_called()

    patched.reset_mock()

    # Second call
    surrogate.fit(searchspace, objective, measurements)
    patched.assert_not_called()
