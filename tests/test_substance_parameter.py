"""
Tests for teh generic substance parameter.
"""
from typing import get_args, get_type_hints

import pytest

from baybe.core import BayBE, BayBEConfig
from baybe.parameters import GenericSubstance
from baybe.utils import add_fake_results, add_parameter_noise

valid_encodings = get_args(get_type_hints(GenericSubstance)["encoding"])


@pytest.mark.parametrize("encoding", valid_encodings)
def test_run_iterations(
    config_discrete_1target,
    mock_substances,
    encoding,
    batch_quantity,
    n_iterations,
    good_reference_values,
):
    """
    Test running some iterations with fake results and a substance parameter.
    """
    config_discrete_1target["parameters"].append(
        {
            "name": "Substance_1",
            "type": "SUBSTANCE",
            "data": mock_substances,
            "encoding": encoding,
        },
    )

    config = BayBEConfig(**config_discrete_1target)
    baybe_obj = BayBE(config)

    for k in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)
