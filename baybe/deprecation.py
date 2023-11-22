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
