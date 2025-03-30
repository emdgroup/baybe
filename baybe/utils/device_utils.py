"""Utilities for device management across BayBE."""

import gc
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from gpytorch.settings import debug, fast_computations


class single_device_mode:
    """Context manager that forces all operations to happen on a single device."""

    _global_value = False

    def __init__(self, state: bool = True):
        self.prev = self.__class__._global_value
        self.state = state

    def __enter__(self):
        self.__class__._global_value = self.state
        return self

    def __exit__(self, *args):
        self.__class__._global_value = self.prev
        return False

    @classmethod
    def on(cls) -> bool:
        """Return whether single device mode is currently enabled."""
        return cls._global_value


@contextmanager
def device_mode(state: bool = True) -> Generator[None, None, None]:
    """Context manager that forces all operations to happen on a single device.

    This combines multiple GPyTorch settings to ensure consistent device usage:
    - single_device_mode: Forces operations to stay on one device
    - debug: Enables additional device checks
    - fast_computations(solves=False): Prevents some caching that can lead
    to device mismatches

    Args:
        state: If True, enable single device mode. If False, disable it.

    Yields:
        None: This context manager doesn't yield a value.
    """
    with single_device_mode(state), debug(state), fast_computations(solves=False):
        yield


def get_default_device() -> torch.device:
    """Get the default device (CUDA if available, otherwise CPU).

    Returns:
        torch.device: The default device for computations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(obj: Any, device: torch.device | str | None = None) -> Any:
    """Move an object to the specified device.

    Args:
        obj: The object to move (tensor, model, or other torch object with
          a 'to' method).
        device: The target device. If None, uses the default device.

    Returns:
        The object on the specified device.
    """
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def clear_gpu_memory() -> None:
    """Clear CUDA cache and force garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def device_context(
    device: torch.device | str | None = None,
) -> Generator[torch.device, None, None]:
    """Context manager for performing operations on a specific device.

    Combines device_mode with memory management and consistent device handling.

    Args:
        device: The device to use. If None, uses the default device.

    Yields:
        torch.device: The active device for use within the context.
    """
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    with device_mode(True):
        # Clear GPU memory before operations
        clear_gpu_memory()

        try:
            yield device
        finally:
            # Clear GPU memory after operations
            clear_gpu_memory()
