"""
Tests for teh generic substance parameter.
"""

import pytest

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise


@pytest.mark.parametrize("encoding", ["MORDRED", "RDKIT", "MORGAN_FP"])
def test_run_iterations(
    config_basic_1target,
    mock_substances,
    encoding,
    batch_quantity,
    n_iterations,
    good_reference_values,
):
    """
    Test running some iterations with fake results and a substance parameter.
    """
    config_basic_1target["parameters"].append(
        {
            "name": "Substance_1",
            "type": "SUBSTANCE",
            "data": mock_substances,
            "encoding": encoding,
        },
    )

    config = BayBEConfig(**config_basic_1target)
    baybe_obj = BayBE(config)

    for k in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)
