"""A temporary namespace alias for backward compatibility."""

import warnings

warnings.warn(
    "The use of `baybe.surrogate` is deprecated and will be disabled in "
    "a future version. Use `baybe.surrogates` instead.",
    DeprecationWarning,
)

from baybe.surrogates import *  # noqa: F401, E402, F403
