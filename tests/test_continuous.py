"""
Test for continuous parameters
"""
import pytest
import torch


@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Conti_finite1"],
        ["Conti_finite2"],
        ["Conti_infinite1"],
        ["Conti_infinite2"],
        ["Conti_infinite3"],
        ["Conti_infinite4"],
        ["Conti_infinite5"],
        ["Conti_finite1", "Conti_finite2", "Conti_infinite5"],
    ],
)
def test_valid_configs(baybe_one_maximization_target):
    """
    Test whether the given settings work without error
    """

    baybe_obj = baybe_one_maximization_target
    print(baybe_obj.searchspace.continuous.param_bounds_comp.flatten())

    assert all(
        torch.is_floating_point(itm)
        for itm in baybe_obj.searchspace.continuous.param_bounds_comp.flatten()
    )
