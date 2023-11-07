"""Test for continuous parameters."""
import pytest
import torch


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
    print(campaign.searchspace.continuous.param_bounds_comp.flatten())

    assert all(
        torch.is_floating_point(itm)
        for itm in campaign.searchspace.continuous.param_bounds_comp.flatten()
    )
