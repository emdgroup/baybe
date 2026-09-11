"""Torch utilities shipped as separate module for lazy-loading."""

from typing import Any

import numpy as np
import torch

torch_to_numpy_dtype_mapping: dict[torch.dtype, np.dtype[Any]] = {
    torch.float32: np.dtype("float32"),
    torch.float64: np.dtype("float64"),
}
"""Mapping from Torch to NumPy dtypes."""


def index_in_tensor(needle: torch.Tensor, haystack: torch.Tensor) -> list[int]:
    """Find the position of each row of one tensor within another tensor.

    Args:
        needle: A ``(n, d)`` tensor whose rows are to be located.
        haystack: A ``(m, d)`` tensor to search in.

    Raises:
        ValueError: If any row of ``needle`` has no exact match in ``haystack``.

    Returns:
        The row indices into ``haystack`` for each row of ``needle``.
    """
    match_mask = (haystack.unsqueeze(0) == needle.unsqueeze(1)).all(dim=-1)
    if not match_mask.any(dim=1).all():
        raise ValueError(
            "Could not find all rows of the search tensor in the target tensor."
        )
    return match_mask.int().argmax(dim=1).tolist()
