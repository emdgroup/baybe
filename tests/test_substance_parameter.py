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
def test_run_iterations(baybe, batch_quantity, n_iterations):
    """
    Test running some iterations with fake results and a substance parameter.
    """
    for k in range(n_iterations):
        rec = baybe.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe)
        if k % 2:
            add_parameter_noise(rec, baybe, noise_level=0.1)

        baybe.add_results(rec)
