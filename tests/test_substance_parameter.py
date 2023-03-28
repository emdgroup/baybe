"""
Tests for the generic substance parameter.
"""

import pytest

from baybe.parameters import SUBSTANCE_ENCODINGS
from baybe.utils import add_fake_results, add_parameter_noise


@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", f"Substance_1_{enc}"] for enc in SUBSTANCE_ENCODINGS],
    ids=SUBSTANCE_ENCODINGS,
)
def test_run_iterations(
    baybe_one_maximization_target,
    batch_quantity,
    n_iterations,
):
    """
    Test running some iterations with fake results and a substance parameter.
    """

    baybe_obj = baybe_one_maximization_target

    for k in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe_obj)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)
