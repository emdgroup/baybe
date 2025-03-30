"""Device mode utilities."""

from collections.abc import Generator
from contextlib import contextmanager

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
