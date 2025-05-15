"""Tests for device utilities and GPU/CPU selection."""

import os

import torch

from baybe.utils.device_utils import get_default_device


def test_device_selection():
    """Test that device selection respects BAYBE_USE_GPU environment variable."""
    device = get_default_device()
    print(f"\nActive device: {device}")
    print(f"BAYBE_USE_GPU: {os.environ.get('BAYBE_USE_GPU', 'Not set')}")

    # If BAYBE_USE_GPU is false, we should always get CPU
    if os.environ.get("BAYBE_USE_GPU", "").lower() == "false":
        assert device == torch.device("cpu")

    # If CUDA is available and BAYBE_USE_GPU is true, we should get CUDA
    elif (
        torch.cuda.is_available()
        and os.environ.get("BAYBE_USE_GPU", "").lower() == "true"
    ):
        assert device == torch.device("cuda")
