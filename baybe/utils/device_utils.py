"""Utilities for device management across BayBE."""

import gc
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from gpytorch.settings import debug, fast_computations


class _SingleDeviceMode:
    """Internal context manager that forces operations to happen on a single device."""

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


def get_default_device() -> torch.device:
    """Get the default device (CUDA if explicitly enabled, otherwise CPU).

    Respects the BAYBE_USE_GPU environment variable:
    - If set to "true", attempts to use CUDA when available
    - If set to "false" or unset, uses CPU

    Returns:
        torch.device: The default device for computations.
    """
    # Check if GPU usage is explicitly enabled via environment variable
    use_gpu_env = os.environ.get("BAYBE_USE_GPU", "").lower()

    # Only try to use CUDA if explicitly requested and available
    if use_gpu_env == "true" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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
    manage_memory: bool = True,
    enforce_single_device: bool = True,
) -> Generator[torch.device, None, None]:
    """Context manager for performing operations on a specific device.

    Args:
        device: The device to use. If None, uses the default device.
        manage_memory: If True, clears GPU memory before and after operations.
        enforce_single_device: If True, enforces single device usage with additional
                               GPyTorch settings.

    Yields:
        torch.device: The active device for use within the context.
    """
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Handle GPyTorch settings for consistent device usage
    managers = []
    if enforce_single_device:
        managers.extend(
            [_SingleDeviceMode(True), debug(True), fast_computations(solves=False)]
        )

    # Clear memory if requested
    if manage_memory:
        clear_gpu_memory()

    try:
        # Apply all context managers
        for manager in managers:
            manager.__enter__()

        yield device
    finally:
        # Exit all context managers in reverse order
        for manager in reversed(managers):
            manager.__exit__(None, None, None)

        # Clean up memory after operations if requested
        if manage_memory:
            clear_gpu_memory()
