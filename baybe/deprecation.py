"""Temporary name aliases for backward compatibility."""

import warnings

from attrs import define

from baybe import Campaign


@define
class BayBE(Campaign):
    """A :class:`baybe.campaign.Campaign` alias for backward compatibility."""

    def __attrs_pre_init__(self):
        warnings.warn(
            "The 'BayBE' class is deprecated and will be removed in a future version. "
            "Please use the 'Campaign' class instead.",
            DeprecationWarning,
        )


def compatibilize_config(config: dict) -> dict:
    """Turn a legacy-format config into the new format."""
    if "parameters" not in config:
        return config

    if "searchspace" in config:
        raise ValueError(
            "Something is wrong with your campaign config. "
            "It neither adheres to the deprecated nor the new format."
        )

    warnings.warn(
        '''
        Specifying parameters/constraints at the top level of the
        campaign configuration JSON is deprecated and will not be
        supported in future releases.
        Instead, use a dedicated "searchspace" field that can be
        used to customize the creation of the search space,
        offering the possibility to specify a desired constructor.

        To replicate the old behavior, use
        """
        ...
        "searchspace": {
            "constructor": "from_product",
            "parameters": <your parameter configuration>,
            "constraints": <your constraints configuration>
        }
        ...
        """

        For the available constructors and the parameters they expect,
        see `baybe.searchspace.core.SearchSpace`.''',
        UserWarning,
    )

    config = config.copy()
    config["searchspace"] = {
        "constructor": "from_product",
        "parameters": config.pop("parameters"),
        "constraints": config.pop("constraints", None),
    }

    return config
