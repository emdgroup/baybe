"""Device mode utilities."""

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


@contextmanager
def cpu_only_mode() -> Generator[None, None, None]:
    """Force PyTorch to use CPU only, even if CUDA is available.

    This context manager temporarily patches torch.Tensor.cuda and other CUDA
    related functions to prevent any operations from using CUDA. This is useful
    when you want to ensure that all operations happen on CPU.

    Yields:
        None
    """
    # Save original functions
    original_tensor_cuda = torch.Tensor.cuda
    original_is_available = torch.cuda.is_available

    try:
        # Patch functions to force CPU usage
        def dummy_cuda(self, *args, **kwargs):
            return self.to("cpu")

        def dummy_is_available():
            return False

        # Apply patches
        torch.Tensor.cuda = dummy_cuda
        torch.cuda.is_available = dummy_is_available

        # Run with patched functions
        with device_mode(True):
            yield

    finally:
        # Restore original functions
        torch.Tensor.cuda = original_tensor_cuda
        torch.cuda.is_available = original_is_available


def force_device_recursive(obj: Any, device: torch.device) -> None:
    """Recursively forces all tensors in an object to a specific device.

    This function traverses the object structure and any nested attributes,
    moving all tensors to the specified device.

    Args:
        obj: The object to traverse and modify
        device: The device to move tensors to
    """
    if obj is None:
        return

    # Handle tensors directly
    if isinstance(obj, torch.Tensor):
        obj.data = obj.data.to(device)
        return

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        for item in obj:
            force_device_recursive(item, device)
        return

    # Handle dictionaries
    if isinstance(obj, dict):
        for value in obj.values():
            force_device_recursive(value, device)
        return

    # Handle objects with attributes
    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in obj.__dict__.items():
            # Skip private attributes and methods
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                continue

            try:
                # Try to move the attribute to the device
                if isinstance(attr_value, torch.Tensor):
                    setattr(obj, attr_name, attr_value.to(device))
                elif hasattr(attr_value, "to") and callable(attr_value.to):
                    try:
                        setattr(obj, attr_name, attr_value.to(device))
                    except Exception:
                        # If we can't move the entire object, recurse its attributes
                        force_device_recursive(attr_value, device)
                else:
                    # For other types, recurse
                    force_device_recursive(attr_value, device)
            except Exception:
                # Skip attributes that can't be processed
                pass


@contextmanager
def force_to_device(model: Any, device: torch.device) -> Generator[None, None, None]:
    """Context manager that forces a model and all its tensors to a specific device.

    This is a more aggressive approach than device_mode as it recursively
    traverses the object structure and moves all tensors to the specified device.

    Args:
        model: The model to force to device
        device: The target device

    Yields:
        None: This context manager doesn't yield a value.
    """
    # First, use the standard device_mode to get GPyTorch's help
    with device_mode(True):
        # Force move to CPU first to avoid mixed device operations
        if device != torch.device("cpu"):
            force_device_recursive(model, torch.device("cpu"))

        # Now move to the target device
        force_device_recursive(model, device)

        # Force empty all caches if available
        if (
            hasattr(model, "prediction_strategy")
            and model.prediction_strategy is not None
        ):
            if hasattr(model.prediction_strategy, "_memoize_cache"):
                model.prediction_strategy._memoize_cache = {}
            if hasattr(model.prediction_strategy, "_mean_cache"):
                model.prediction_strategy._mean_cache = None
            if hasattr(model.prediction_strategy, "_covar_cache"):
                model.prediction_strategy._covar_cache = None

        yield
