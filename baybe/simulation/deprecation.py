"""Temporary functionality for backward compatibility."""

import warnings

import pandas as pd

from baybe.simulation import simulate_campaign


def simulate_experiment(*args, **kwargs) -> pd.DataFrame:
    """A :func:`simulate_campaign` alias for backward compatibility."""  # noqa: D401
    warnings.warn(
        "Using 'simulate_experiment' is deprecated and will be removed in a future "
        "version. Please use the 'simulate_campaign' instead.",
        DeprecationWarning,
    )
    return simulate_campaign(*args, **kwargs)
