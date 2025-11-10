"""Torch utilities shipped as separate module for lazy-loading."""

import os
from typing import Any

import numpy as np
import torch

from baybe.utils.boolean import strtobool

VARNAME_TORCH_USE_SINGLE_PRECISION = "BAYBE_TORCH_USE_SINGLE_PRECISION"
"""Environment variable name for enforcing single precision in torch."""

DTypeFloatTorch = (
    torch.float32
    if strtobool(os.environ.get(VARNAME_TORCH_USE_SINGLE_PRECISION, "False"))
    else torch.float64
)
"""Floating point data type used for torch tensors."""

torch_to_numpy_dtype_mapping: dict[torch.dtype, np.dtype[Any]] = {
    torch.float32: np.dtype("float32"),
    torch.float64: np.dtype("float64"),
}
"""Mapping from Torch to NumPy dtypes."""
