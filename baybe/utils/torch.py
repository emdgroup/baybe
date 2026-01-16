"""Torch utilities shipped as separate module for lazy-loading."""

from typing import Any

import numpy as np
import torch

torch_to_numpy_dtype_mapping: dict[torch.dtype, np.dtype[Any]] = {
    torch.float32: np.dtype("float32"),
    torch.float64: np.dtype("float64"),
}
"""Mapping from Torch to NumPy dtypes."""
