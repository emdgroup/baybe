"""
Tests for basic input-output nad iterative loop.
"""
import pytest

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

dict_surrogate_variants = {
    "gaussian_process": {
        "surrogate_model_cls": "GP",
    },
    "mean_prediction": {
        "surrogate_model_cls": "MP",
    },
    "random_forest": {
        "surrogate_model_cls": "RF",
    },
    "ngboost": {
        "surrogate_model_cls": "NG",
    },
    "bayesian_linear": {
        "surrogate_model_cls": "BL",
    },
}


@pytest.mark.parametrize("config_update_key", dict_surrogate_variants.keys())
def test_run_iterations(
    config_basic_1target,
    n_iterations,
    good_reference_values,
    batch_quantity,
    config_update_key,
):
    """
    Test running some iterations with fake results and basic parameters.
    """
    config_basic_1target["strategy"] = dict_surrogate_variants[config_update_key]
    config = BayBEConfig(**config_basic_1target)
    baybe_obj = BayBE(config)

    for k in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)
