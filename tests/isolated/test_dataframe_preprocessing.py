"""Test module for dataframe preprocessing behavior."""

import sys
from unittest.mock import patch

import pandas as pd


def patch_with_import_reset(path: str):
    """Decorator that patches a function and resets module imports for fresh testing."""
    module = ".".join(path.split(".")[:-1])

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Clear baybe modules from sys.modules to ensure fresh imports

            # Apply the patch and run the test
            with patch(path) as mock:
                modules_to_remove = [
                    name
                    for name in sys.modules
                    if name.startswith("baybe") and name != module
                ]
                for module_name in modules_to_remove:
                    del sys.modules[module_name]
                return func(mock, *args, **kwargs)

        return wrapper

    return decorator


@patch_with_import_reset("baybe.utils.validation.preprocess_dataframe")
def test_dataframes_are_preprocessed_only_once(mock_preprocess):
    # Configure the mock to return the input DataFrame (side_effect passes through)
    mock_preprocess.side_effect = lambda df, *args, **kwargs: df

    # # Clear baybe modules from sys.modules to ensure fresh imports
    # import sys

    # modules_to_remove = [name for name in sys.modules if name.startswith("baybe")]
    # for module_name in modules_to_remove:
    #     if module_name != "baybe.utils.validation":
    #         del sys.modules[module_name]

    from baybe.campaign import Campaign
    from baybe.parameters.numerical import NumericalDiscreteParameter
    from baybe.targets.numerical import NumericalTarget

    searchspace = NumericalDiscreteParameter("p", [0, 1]).to_searchspace()
    objective = NumericalTarget("t").to_objective()
    campaign = Campaign(searchspace, objective)
    campaign.add_measurements(pd.DataFrame({"p": [0], "t": [0]}))

    # Assert that preprocess_dataframe was called exactly once
    campaign.recommend(1)
    assert mock_preprocess.call_count == 3
