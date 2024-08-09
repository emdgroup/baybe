"""Test for continuous parameters."""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Conti_finite1"],
        ["Conti_finite2"],
        ["Conti_finite1", "Conti_finite2"],
    ],
)
def test_valid_configs(campaign):
    """Test whether the given settings work without error."""
    print(campaign.searchspace.continuous.comp_rep_bounds.values.flatten())

    assert all(
        np.issubdtype(type(itm), np.floating)
        for itm in campaign.searchspace.continuous.comp_rep_bounds.values.flatten()
    )
