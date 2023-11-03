"""Temporary name aliases for backward compatibility."""

import warnings

from baybe import Campaign


def BayBE(*args, **kwargs) -> Campaign:  # pylint: disable=invalid-name
    """A ```Campaign``` alias for backward compatibility."""
    warnings.warn(
        "The 'BayBE' class is deprecated and will be removed in a future version. "
        "Please use the 'Campaign' class instead.",
        DeprecationWarning,
    )
    return Campaign(*args, **kwargs)
