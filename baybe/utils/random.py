"""Utilities targeting random number generation."""

import contextlib
import random
import warnings

import numpy as np
from typing_extensions import deprecated


@deprecated(
    "Using 'set_random_seed' is deprecated and support will be removed in a future "
    "release. Use the new settings management system instead. For details: "
    "https://emdgroup.github.io/baybe/stable/userguide/settings.html)",
)
def set_random_seed(seed: int):
    """Set the global random seed.

    Args:
        seed: The chosen global random seed.
    """
    import torch

    # Ensure seed limits
    seed %= 2**32

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@deprecated(
    "Using 'temporary_seed' is deprecated and support will be removed in a future "
    "release. Use the new settings management system instead. For details: "
    "https://emdgroup.github.io/baybe/stable/userguide/settings.html)",
)
@contextlib.contextmanager
def temporary_seed(seed: int):  # noqa: DOC402, DOC404
    """Context manager for setting a temporary random seed.

    Args:
        seed: The chosen random seed.
    """
    import torch

    # Ensure seed limits
    seed %= 2**32

    # Collect the current RNG states
    state_builtin = random.getstate()
    state_np = np.random.get_state()
    state_torch = torch.get_rng_state()

    # Set the requested seed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        set_random_seed(seed)

    # Run the context-specific code
    try:
        yield

    # Restore the original RNG states
    finally:
        random.setstate(state_builtin)
        np.random.set_state(state_np)
        torch.set_rng_state(state_torch)
