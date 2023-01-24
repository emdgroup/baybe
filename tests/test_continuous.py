"""
Test for continuous parameters
"""
import numpy as np
import pytest
from baybe.core import BayBE, BayBEConfig

param_updates = {
    "conti_finite": {
        "name": "Conti_Param",
        "type": "NUM_CONTINUOUS",
        "bounds": [0, 1],
    },
    "conti_infinite1": {
        "name": "Conti_Param",
        "type": "NUM_CONTINUOUS",
        "bounds": [None, 1],
    },
    "conti_infinite2": {
        "name": "Conti_Param",
        "type": "NUM_CONTINUOUS",
        "bounds": [0, None],
    },
    "conti_infinite3": {
        "name": "Conti_Param",
        "type": "NUM_CONTINUOUS",
        "bounds": [0, np.inf],
    },
    "conti_infinite4": {
        "name": "Conti_Param",
        "type": "NUM_CONTINUOUS",
        "bounds": [-np.inf, 1],
    },
    "conti_infinite5": {
        "name": "Conti_Param",
        "type": "NUM_CONTINUOUS",
        "bounds": [None, None],
    },
}


@pytest.mark.parametrize("config_update_key", param_updates.keys())
def test_valid_configs(
    config_basic_1target,
    config_update_key,
):
    """
    Test whether the given settings work without error
    """
    config_basic_1target["parameters"].append(param_updates[config_update_key])

    config = BayBEConfig(**config_basic_1target)
    baybe_obj = BayBE(config)

    assert baybe_obj is not None  # needed to avoid non-used lint error
