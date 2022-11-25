"""
Tests for basic input-output nad iterative loop.
"""
import numpy as np
import pytest

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise


def test_run_iterations(
    config_basic_1target, n_iterations, good_reference_values, batch_quantity
):
    """
    Test running some iterations with fake results and basic parameters.
    """
    config = BayBEConfig(**config_basic_1target)
    baybe_obj = BayBE(config)

    for k in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)


@pytest.mark.parametrize(
    "bad_val",
    [1337, np.nan, "asd"],
    ids=["not_within_tol", "nan", "string_instead_float"],
)
def test_bad_input_value(
    baybe_object_batch3_iterations2, good_reference_values, bad_val
):
    """
    Test attempting to read in an invalid parameter value.
    """

    rec = baybe_object_batch3_iterations2.recommend(batch_quantity=3)
    add_fake_results(
        rec,
        baybe_object_batch3_iterations2,
        good_reference_values=good_reference_values,
    )

    # Add an invalid value
    rec.Num_disc_1.iloc[0] = bad_val
    with pytest.raises(ValueError):
        baybe_object_batch3_iterations2.add_results(rec)
