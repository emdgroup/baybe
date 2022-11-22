"""
Tests for basic input-output nad iterative loop.
"""
import pytest

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise


@pytest.mark.parametrize("batch_quantity", [1, 3])
def test_basic_io_run_iterations(
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


def test_bad_input_value(config_basic_1target, good_reference_values):
    """
    Test attempting to read in an invalid parameter value.
    """
    config = BayBEConfig(**config_basic_1target)
    baybe_obj = BayBE(config)

    for k in range(1):
        rec = baybe_obj.recommend(batch_quantity=3)

        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        # Add an invalid value
        rec.Num_disc_1.iloc[0] = 1337
        with pytest.raises(ValueError):
            baybe_obj.add_results(rec)


# # ToDo Add these tests once the duplciate fix is included in branch
# def test_basic_dulpicated_parameter_value_categorical(config_basic_1target):
#     """
#     Test attempting to create a numeric parameter with duplciated values.
#     """
#     config_basic_1target["parameters"].append(
#         {
#             "name": "Categorical_New",
#             "type": "CAT",
#             "values": ["very bad", "bad", "OK", "bad"],
#         },
#     )
#
#     with pytest.raises(ValueError):
#         config = BayBEConfig(**config_basic_1target)
#
#
# def test_basic_duplicated_parameter_value_numeric(config_basic_1target):
#     """
#     Test attempting to create a numeric parameter with duplicated values.
#     """
#     config_basic_1target["parameters"].append(
#         {
#             "name": "Numerical_New",
#             "type": "NUM_DISCRETE",
#             "values": [1, 2, 3, 2],
#         },
#     )
#
#     with pytest.raises(ValueError):
#         config = BayBEConfig(**config_basic_1target)
