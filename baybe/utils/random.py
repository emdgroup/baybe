"""Utilities targeting random number generation."""

from __future__ import annotations

import contextlib
import random
import warnings
from typing import TYPE_CHECKING

import numpy as np
from attrs import cmp_using, define, field
from typing_extensions import Self, deprecated

if TYPE_CHECKING:
    from torch import Tensor


@deprecated(
    "Using 'set_random_seed' is deprecated and support will be removed in a future "
    "release. Use the new settings management system instead. For details: "
    "https://emdgroup.github.io/baybe/stable/userguide/settings.html",
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
    "https://emdgroup.github.io/baybe/stable/userguide/settings.html",
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
    state_python = random.getstate()
    state_np = np.random.get_state()
    state_torch = torch.get_rng_state()

    # Set the requested seed
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using 'set_random_seed' is deprecated",
            category=DeprecationWarning,
        )
        set_random_seed(seed)

    # Run the context-specific code
    try:
        yield

    # Restore the original RNG states
    finally:
        random.setstate(state_python)
        np.random.set_state(state_np)
        torch.set_rng_state(state_torch)


def _lazy_torch_equal(a: Tensor, b: Tensor, /) -> bool:
    """Equality check for tensors with lazy torch import."""
    import torch

    return torch.equal(a, b)


@define(frozen=True)
class _RandomState:
    """Container for the random states of all managed numeric libraries."""

    state_python = field(init=False, factory=random.getstate)
    """The state of the Python random number generator."""

    state_numpy = field(
        init=False,
        factory=np.random.get_state,
        eq=cmp_using(
            eq=lambda s1, s2: all(np.array_equal(a, b) for a, b in zip(s1, s2))
        ),
    )
    """The state of the Numpy random number generator."""

    state_torch: Tensor = field(init=False, eq=cmp_using(eq=_lazy_torch_equal))
    """The state of the Torch random number generator."""
    # Note: initialized by attrs default method below (for lazy torch loading)

    @state_torch.default
    def _default_state_torch(self) -> Tensor:
        """Get the current Torch random state using a lazy import."""
        import torch

        return torch.get_rng_state()

    def activate(self) -> None:
        """Activate the random state."""
        import torch

        random.setstate(self.state_python)
        np.random.set_state(self.state_numpy)
        torch.set_rng_state(self.state_torch)

    @staticmethod
    def _activate_seed(seed: int) -> None:
        """Seed all random number generators."""
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def from_seed(cls, seed: int, *, activate: bool = False) -> Self:
        """Create a random state corresponding to a given seed."""
        if activate:
            cls._activate_seed(seed)
            return cls()

        backup = cls()
        cls._activate_seed(seed)
        state = cls()
        backup.activate()
        return state
