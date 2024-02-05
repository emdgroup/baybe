"""Temporary functionality for backward compatibility."""

import warnings

from baybe.strategies.composite import TwoPhaseStrategy


def Strategy(*args, **kwargs) -> TwoPhaseStrategy:
    """A ``Strategy`` alias for backward compatibility."""  # noqa: D401 (imperative mood)
    from baybe.strategies.composite import TwoPhaseStrategy

    warnings.warn(
        f"Using 'Strategy' directly is deprecated and will be removed in a future "
        f"version. Please use '{TwoPhaseStrategy.__name__}' class instead.",
        DeprecationWarning,
    )

    return TwoPhaseStrategy(*args, **kwargs)
