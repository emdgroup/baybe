"""Surrogate tests."""

from unittest.mock import patch

from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate


@patch.object(GaussianProcessSurrogate, "_fit")
def test_caching(patched, searchspace, objective, fake_measurements):
    """A second fit call with the same context does not trigger retraining."""
    # Prepare the setting
    surrogate = GaussianProcessSurrogate()

    # First call
    surrogate.fit(searchspace, objective, fake_measurements)
    patched.assert_called()

    patched.reset_mock()

    # Second call
    surrogate.fit(searchspace, objective, fake_measurements)
    patched.assert_not_called()
