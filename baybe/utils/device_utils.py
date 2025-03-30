"""Utilities for device management across BayBE."""

import gc
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
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


def to_tensor(
    *args: Any,
    device: torch.device | str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Convert one or multiple inputs to PyTorch tensors.

    Args:
        *args: One or multiple inputs to convert to tensors.
        device: The device to move the tensor(s) to.

    Returns:
        The provided array(s)/dataframe(s) represented as tensor(s).
    """
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    def _convert_single(data: Any) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return to_device(data, device)
        elif isinstance(data, pd.DataFrame):
            tensor = torch.tensor(data.values.astype(np.float64))
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data.astype(np.float64))
        else:
            tensor = torch.tensor(data, dtype=torch.float64)

        return to_device(tensor, device)

    # Handle single or multiple inputs
    if len(args) == 1:
        return _convert_single(args[0])
    else:
        return tuple(_convert_single(arg) for arg in args)
